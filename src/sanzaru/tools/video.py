# SPDX-License-Identifier: MIT
"""Video generation on Higgsfield, plus the local video library.

OpenAI removed the Videos API and every Sora model on 2026-09-24; sanzaru now
generates video through Higgsfield (api.higgsfield.ai). Jobs are asynchronous:
`create_video` / `edit_video` / `extend_video` return an `hf_<uuid>` id at
once, `wait_for` (or `get_video_status`) follows it, and `download_video`
copies the finished file into sanzaru's own storage — Higgsfield keeps outputs
for about a week, so the copy is the durable one.

Every submit is priced first. A curated model is validated locally before any
request; the job is then estimated (free, and done with placeholder URLs, so a
refused job uploads nothing), `max_cost_usd` is enforced, and only then are
the inputs uploaded and the job submitted. Seedance's estimate endpoint
answers in prose, so its price comes from the local table in
`higgsfield/pricing.py`.

A source video or reference image may be a local filename *or* the `hf_…` id
of an earlier completed job, whose output URL is used directly — chaining an
extend onto a render needs no download and re-upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict
from urllib.parse import urlsplit

import anyio

from ..config import get_higgsfield_client, logger
from ..higgsfield.errors import HiggsfieldAPIError
from ..higgsfield.ids import JOB_PREFIX, to_job_id, to_request_id
from ..higgsfield.limits import make_limiter, max_upload_bytes
from ..higgsfield.media import content_type_for, probe_duration
from ..higgsfield.pricing import VideoCost, enforce_cap, local_estimate, resolve_cost
from ..higgsfield.types import TERMINAL_STATES, RequestStatus
from ..storage import get_storage
from ..storage.protocol import PathType
from ..types import DownloadResult, VideoFile
from ..utils import generate_filename
from ..video_models import DEFAULT_VIDEO_MODEL, Operation, build_arguments, build_raw_arguments, resolve_model

_PLACEHOLDER = "https://placeholder.invalid/{}"
_OUTPUT_EXTENSIONS = frozenset({".mp4", ".mov"})


class VideoJob(TypedDict):
    """A submitted (or dry-run) video job."""

    id: str | None
    """`hf_<uuid>`; None for a dry run, which submits nothing."""
    status: str
    """`queued` on submit, `dry_run` when nothing was submitted."""
    model: str
    slug: str
    operation: Operation
    cost: VideoCost
    arguments: dict[str, object]
    """The request body, with uploaded inputs shown as `<upload:name>`."""


class VideoStatus(TypedDict):
    id: str
    status: str
    done: bool
    """Terminal: completed, failed, nsfw or canceled. Nothing left to wait for."""
    error: str | None
    video_url: str | None
    """Higgsfield's output URL (kept about 7 days); download_video copies it."""


class CancelResult(TypedDict):
    id: str
    canceled: bool


# ==================== inputs ====================


@dataclass(frozen=True, slots=True)
class _Input:
    """One media input: a stored file to upload, or a prior job's output URL."""

    role: Literal["image", "end_image", "video", "reference"]
    name: str
    kind: Literal["image", "video"]


def _is_job_id(name: str) -> bool:
    return name.startswith(JOB_PREFIX)


async def _job_output_url(job_id: str) -> str:
    """The output URL of a completed earlier job, for use as an input."""
    request_id = to_request_id(job_id, "input")  # validate before building a client
    status = await get_higgsfield_client().status(request_id)
    video = status.get("video")
    if status.get("status") != "completed" or not video:
        raise ValueError(f"{job_id} cannot be used as an input: it is {status.get('status')}, not completed")
    return video["url"]


async def _read_input(item: _Input) -> tuple[bytes, str]:
    """Read a stored input, refusing oversize files from their metadata first."""
    path_type: PathType = "video" if item.kind == "video" else "reference"
    content_type = content_type_for(item.name, item.kind)
    storage = get_storage()
    info = await storage.stat(path_type, item.name)
    limit = max_upload_bytes()
    if info.size_bytes > limit:
        raise ValueError(
            f"{item.name} is {info.size_bytes / 1e6:.0f} MB, over the {limit / 1e6:.0f} MB upload limit "
            "(SANZARU_HIGGSFIELD_MAX_UPLOAD_MB)"
        )
    return await storage.read(path_type, item.name), content_type


# ==================== pipeline ====================


async def _submit_job(
    *,
    operation: Operation,
    model: str,
    prompt: str | None,
    duration: int | None,
    aspect_ratio: str | None,
    resolution: str | None,
    audio: bool | None,
    inputs: list[_Input],
    extra: dict[str, object] | None,
    max_cost_usd: float | None,
    dry_run: bool,
) -> VideoJob:
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValueError(f"max_cost_usd must be positive, got {max_cost_usd}")
    spec = resolve_model(model)
    if spec is None and operation in ("edit", "extend"):
        raise ValueError(
            f"{operation}_video needs a curated model (e.g. {DEFAULT_VIDEO_MODEL}); raw model ids are create_video only"
        )

    def build(urls: dict[str, list[str]]) -> tuple[str, dict[str, object]]:
        image = urls.get("image", [None])[0]
        end_image = urls.get("end_image", [None])[0]
        if spec is None:
            return build_raw_arguments(
                model,
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                audio=audio,
                image_url=image,
                end_image_url=end_image,
                extra=extra,
            )
        return build_arguments(
            spec,
            operation,
            prompt=prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            audio=audio,
            image_url=image,
            end_image_url=end_image,
            video_url=urls.get("video", [None])[0],
            reference_image_urls=urls.get("reference") or None,
            extra=extra,
        )

    # 1. Validate the request shape with placeholder URLs: a bad argument fails
    #    here, before anything is read, uploaded or priced.
    placeholders: dict[str, list[str]] = {}
    for index, item in enumerate(inputs):
        placeholders.setdefault(item.role, []).append(_PLACEHOLDER.format(index))
    slug, draft = build(placeholders)

    # 2. Read stored inputs (job-id inputs resolve later, to their output URL)
    #    and measure the source video, which edit/extend bill for.
    data: dict[int, tuple[bytes, str]] = {}
    input_seconds: float | None = None
    for index, item in enumerate(inputs):
        if _is_job_id(item.name):
            continue
        data[index] = await _read_input(item)
        if item.role == "video":
            input_seconds = await probe_duration("video", item.name, data[index][0])

    # 3. Price with the placeholders (the estimate endpoint does not fetch
    #    inputs), so an over-budget job is refused before any upload.
    client = get_higgsfield_client()
    local = local_estimate(spec.family, operation, draft, input_seconds) if spec else None
    try:
        remote = await client.estimate(slug, draft)
        estimate_failed = False
    except HiggsfieldAPIError as exc:
        if exc.kind in ("invalid", "validation"):
            raise ValueError(f"Higgsfield rejected the arguments for {slug}: {exc.detail}") from exc
        logger.warning("Higgsfield estimate failed for %s: %s", slug, exc)
        remote, estimate_failed = None, True
    cost = resolve_cost(remote, local, estimate_failed=estimate_failed)
    enforce_cap(cost, max_cost_usd, model)

    shown: dict[str, list[str]] = {}
    for item in inputs:
        label = item.name if _is_job_id(item.name) else f"<upload:{item.name}>"
        shown.setdefault(item.role, []).append(label)
    shown_arguments = build(shown)[1]

    if dry_run:
        return VideoJob(
            id=None,
            status="dry_run",
            model=model,
            slug=slug,
            operation=operation,
            cost=cost,
            arguments=shown_arguments,
        )

    # 4. Upload (bounded) and resolve job-id inputs, then submit exactly once.
    urls: dict[int, str] = {}
    limiter = make_limiter()

    async def resolve(index: int, item: _Input) -> None:
        async with limiter:
            if _is_job_id(item.name):
                urls[index] = await _job_output_url(item.name)
            else:
                payload, content_type = data[index]
                urls[index] = await client.upload(payload, content_type)

    async with anyio.create_task_group() as tg:
        for index, item in enumerate(inputs):
            tg.start_soon(resolve, index, item)
    real: dict[str, list[str]] = {}
    for index, item in enumerate(inputs):
        real.setdefault(item.role, []).append(urls[index])
    slug, arguments = build(real)

    submitted = await client.submit(slug, arguments)
    job_id = to_job_id(submitted["request_id"])
    logger.info("Video job %s submitted (%s, %s)", job_id, slug, cost["basis"])
    return VideoJob(
        id=job_id,
        status=submitted.get("status", "queued"),
        model=model,
        slug=slug,
        operation=operation,
        cost=cost,
        arguments=shown_arguments,
    )


# ==================== tools ====================


async def create_video(
    prompt: str,
    model: str = DEFAULT_VIDEO_MODEL,
    duration: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    audio: bool | None = None,
    reference_image: str | None = None,
    end_image: str | None = None,
    extra: dict[str, object] | None = None,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
) -> VideoJob:
    """Start a text-to-video job, or image-to-video when `reference_image` is given.

    Args:
        prompt: What happens. With a reference image, describe only the motion.
        model: A curated id (`seedance-2.5`, `kling-3.0-std|pro|4k|turbo`) or any
            Higgsfield catalog slug (`vendor/model/operation`).
        duration: Seconds; the model's range applies (Seedance 4-30, Kling 3-15).
        aspect_ratio: Text-to-video only — with an image, framing follows the image.
        resolution: `480p` / `720p` where the model offers a choice.
        audio: Generate a soundtrack (model default when None).
        reference_image: Start frame — a file in the reference images directory,
            or the `hf_…` id of a completed job.
        end_image: Optional last frame (needs `reference_image`).
        extra: Model-specific parameters, merged into the request.
        max_cost_usd: Refuse (submitting nothing) if the estimate is higher.
        dry_run: Price and validate only; nothing is uploaded or submitted.
    """
    if end_image and not reference_image:
        raise ValueError("end_image needs reference_image (the start frame)")
    inputs = []
    if reference_image:
        inputs.append(_Input("image", reference_image, "image"))
    if end_image:
        inputs.append(_Input("end_image", end_image, "image"))
    return await _submit_job(
        operation="image" if reference_image else "text",
        model=model,
        prompt=prompt,
        duration=duration,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        audio=audio,
        inputs=inputs,
        extra=extra,
        max_cost_usd=max_cost_usd,
        dry_run=dry_run,
    )


async def edit_video(
    prompt: str,
    source_video: str,
    model: str = DEFAULT_VIDEO_MODEL,
    resolution: str | None = None,
    audio: bool | None = None,
    reference_images: list[str] | None = None,
    extra: dict[str, object] | None = None,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
) -> VideoJob:
    """Re-render `source_video` per `prompt`; the output keeps the source's length and framing.

    The source is billed as well as the output, so an edit costs roughly twice
    the source's duration (at Seedance's 0.6x video-input rate).
    """
    inputs = [_Input("video", source_video, "video")]
    inputs += [_Input("reference", name, "image") for name in reference_images or []]
    return await _submit_job(
        operation="edit",
        model=model,
        prompt=prompt,
        duration=None,
        aspect_ratio=None,
        resolution=resolution,
        audio=audio,
        inputs=inputs,
        extra=extra,
        max_cost_usd=max_cost_usd,
        dry_run=dry_run,
    )


async def extend_video(
    prompt: str,
    source_video: str,
    duration: int | None = None,
    model: str = DEFAULT_VIDEO_MODEL,
    resolution: str | None = None,
    audio: bool | None = None,
    reference_images: list[str] | None = None,
    extra: dict[str, object] | None = None,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
) -> VideoJob:
    """Continue `source_video` by `duration` seconds. The source is billed too."""
    inputs = [_Input("video", source_video, "video")]
    inputs += [_Input("reference", name, "image") for name in reference_images or []]
    return await _submit_job(
        operation="extend",
        model=model,
        prompt=prompt,
        duration=duration,
        aspect_ratio=None,
        resolution=resolution,
        audio=audio,
        inputs=inputs,
        extra=extra,
        max_cost_usd=max_cost_usd,
        dry_run=dry_run,
    )


def _video_status(job_id: str, status: RequestStatus) -> VideoStatus:
    state = status.get("status", "unknown")
    video = status.get("video")
    error = status.get("error")
    if state == "nsfw" and not error:
        error = "rejected by content moderation (not charged) — rephrase the prompt or change the inputs"
    return VideoStatus(
        id=job_id,
        status=state,
        done=state in TERMINAL_STATES,
        error=error,
        video_url=video["url"] if video else None,
    )


async def get_video_status(video_id: str) -> VideoStatus:
    """One status check. To wait for completion use wait_for, not a loop over this."""
    request_id = to_request_id(video_id)  # a bad id is a usage error, even without HF_KEY
    status = await get_higgsfield_client().status(request_id)
    return _video_status(video_id, status)


def _output_extension(url: str) -> str:
    path = urlsplit(url).path.lower()
    dot = path.rfind(".")
    ext = path[dot:] if dot != -1 else ""
    return ext if ext in _OUTPUT_EXTENSIONS else ".mp4"


async def download_video(video_id: str, filename: str | None = None) -> DownloadResult:
    """Copy a completed job's output into the video directory."""
    status = await get_video_status(video_id)
    if status["status"] != "completed" or not status["video_url"]:
        detail = f": {status['error']}" if status["error"] else ""
        raise ValueError(f"{video_id} is {status['status']}, not completed{detail}")
    extension = _output_extension(status["video_url"])
    name = filename or generate_filename(video_id, extension.lstrip("."))
    async with get_higgsfield_client().stream_output(status["video_url"]) as chunks:
        written = await get_storage().write_stream("video", name, chunks)
    logger.info("Downloaded %s to %s", video_id, written)
    return DownloadResult(filename=name, format=extension.lstrip("."))


async def cancel_video(video_id: str) -> CancelResult:
    """Cancel a job that is still queued (refunded). A started job cannot be canceled."""
    request_id = to_request_id(video_id)
    try:
        await get_higgsfield_client().cancel(request_id)
    except HiggsfieldAPIError as exc:
        if exc.status_code == 400:
            raise ValueError(
                f"{video_id} has already started; Higgsfield only cancels queued jobs (and refunds them)"
            ) from exc
        raise
    return CancelResult(id=video_id, canceled=True)


async def list_local_videos(
    pattern: str | None = None,
    file_type: Literal["mp4", "webm", "mov", "all"] = "all",
    sort_by: Literal["name", "size", "modified"] = "modified",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
) -> dict:
    """List locally downloaded video files.

    Args:
        pattern: Glob pattern to filter filenames (e.g., "*.mp4", "hf_*")
        file_type: Filter by video type
        sort_by: Sort criterion (name, size, or modified timestamp)
        order: Sort order (asc or desc)
        limit: Maximum number of results to return

    Returns:
        Dict with "data" key containing list of VideoFile objects

    Raises:
        RuntimeError: If VIDEO_PATH not configured
    """
    storage = get_storage()

    # Map file_type to extensions
    type_to_extensions: dict[str, set[str]] = {
        "mp4": {".mp4"},
        "webm": {".webm"},
        "mov": {".mov"},
        "all": {".mp4", ".webm", ".mov"},
    }
    allowed_extensions = type_to_extensions[file_type]

    # Collect matching files via storage backend
    glob_pattern = pattern if pattern else "*"
    file_infos = await storage.list_files("video", pattern=glob_pattern, extensions=allowed_extensions)

    # Sort files
    if sort_by == "name":
        file_infos.sort(key=lambda x: x.name, reverse=(order == "desc"))
    elif sort_by == "size":
        file_infos.sort(key=lambda x: x.size_bytes, reverse=(order == "desc"))
    elif sort_by == "modified":
        file_infos.sort(key=lambda x: x.modified_timestamp, reverse=(order == "desc"))

    # Build result list
    results: list[VideoFile] = []
    for info in file_infos[:limit]:
        # Determine file type from extension
        ext = ("." + info.name.rsplit(".", 1)[-1].lower()) if "." in info.name else ""
        if ext == ".mp4":
            vid_type = "mp4"
        elif ext == ".webm":
            vid_type = "webm"
        else:
            vid_type = "mov"

        results.append(
            {
                "filename": info.name,
                "size_bytes": info.size_bytes,
                "modified_timestamp": int(info.modified_timestamp),
                "file_type": vid_type,
            }
        )

    logger.info("Listed %d local videos (pattern=%s, type=%s)", len(results), glob_pattern, file_type)
    return {"data": results}
