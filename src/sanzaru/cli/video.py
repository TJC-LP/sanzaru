# SPDX-License-Identifier: MIT
"""`sanzaru video` — Higgsfield video jobs.

OpenAI retired the Videos API (and every Sora model) on 2026-09-24; video is
now generated on Higgsfield. Workflow: create/edit/extend → wait → download.
`create -o out.mp4` composes all three (`-o` implies `--download` implies
`--wait`). Waits are idempotent and resumable: on exit 4 (timeout) the job
keeps running server-side — re-run the `resume` command from the envelope.

Every submit is priced first (`cost` in the envelope); `--max-cost` refuses an
over-budget job before anything is uploaded, and `--dry-run` prices without
submitting.
"""

from __future__ import annotations

import json
import pathlib
import shlex
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal, cast

import anyio
import click

from ..higgsfield.ids import JOB_PREFIX
from ..video_models import ALL_ASPECT_RATIOS, DEFAULT_VIDEO_MODEL, VIDEO_MODEL_IDS, VIDEO_MODELS
from ._io import (
    OutputPlan,
    PathSession,
    finalize_output,
    install_overrides,
    plan_output,
    read_content_arg,
    resolve_input,
)
from ._output import (
    EXIT_JOB_FAILED,
    EXIT_TIMEOUT,
    EXIT_USAGE,
    aggregate_exit_code,
    emit,
    emit_line,
    error_envelope,
    note,
    success_envelope,
)
from ._runtime import CLIError, _classify, get_state, make_progress_printer, parse_duration, run_async

if TYPE_CHECKING:
    from ..tools.video import VideoJob

DEFAULT_TIMEOUT_S = 1800.0  # a long Seedance render; parity with polling.DEFAULT_VIDEO_TIMEOUT

_RESOLUTIONS = ["480p", "720p", "1080p"]
_ASPECT_RATIOS = sorted(ALL_ASPECT_RATIOS)

# --retry-busy backoff: Higgsfield's per-account concurrency cap rejects a
# submit outright (nothing created, nothing charged), so resubmitting after a
# running job finishes is safe — unlike any other submit failure.
_BUSY_INITIAL_S = 10.0
_BUSY_FACTOR = 1.5
_BUSY_MAX_S = 60.0


class _ModelType(click.ParamType):
    """A curated model id, or any Higgsfield catalog slug (`vendor/model/operation`)."""

    name = "model"

    def convert(self, value: object, param: click.Parameter | None, ctx: click.Context | None) -> str:
        text = str(value)
        if text in VIDEO_MODELS or "/" in text:
            return text
        self.fail(
            f"{text!r} is not a curated model ({', '.join(VIDEO_MODEL_IDS)}) or a catalog slug containing '/'",
            param,
            ctx,
        )


_MODEL = _ModelType()


@click.group()
def video() -> None:
    """Higgsfield video jobs. Workflow: create → wait → download (create -o does all three)."""


def _resume_command(video_id: str, *, download: bool, output: str | None) -> str:
    parts = ["sanzaru", "video", "wait", video_id]
    if download:
        parts.append("--download")
    if output is not None:
        parts += ["-o", output]
    return " ".join(shlex.quote(p) for p in parts)


def _file_payload(final_path: str, fmt: str) -> dict[str, object]:
    payload: dict[str, object] = {"path": final_path, "format": fmt}
    fp = pathlib.Path(final_path)
    if fp.is_file():
        payload["bytes"] = fp.stat().st_size
    return payload


def _parse_extra(arg_pairs: tuple[str, ...], args_file: str | None) -> dict[str, object] | None:
    """Merge `--args @file.json` (an object) with repeated `--arg KEY=JSON` (later wins).

    A `--arg` value that is not valid JSON is taken as a plain string, so
    `--arg bitrate_mode=standard` works without shell-quoting the quotes.
    """
    extra: dict[str, object] = {}
    if args_file is not None:
        raw = read_content_arg(args_file, "--args")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CLIError("usage", f"--args is not valid JSON: {exc}", exit_code=EXIT_USAGE) from exc
        if not isinstance(parsed, dict):
            raise CLIError("usage", "--args must be a JSON object", exit_code=EXIT_USAGE)
        extra.update(cast("dict[str, object]", parsed))
    for pair in arg_pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise CLIError("usage", f"--arg expects KEY=VALUE, got {pair!r}", exit_code=EXIT_USAGE)
        try:
            extra[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            extra[key.strip()] = value
    return extra or None


def _merge_alias(primary: object, alias: object, names: str) -> object:
    if primary is not None and alias is not None and primary != alias:
        raise CLIError("usage", f"{names} disagree; pass one", exit_code=EXIT_USAGE)
    return primary if primary is not None else alias


def _resolve_media(session: PathSession, value: str, path_type: Literal["video", "reference"], arg: str) -> str:
    """A local file resolved for the tool layer, or an `hf_…` job id passed through."""
    if value.startswith(JOB_PREFIX):
        return value
    return resolve_input(session, value, path_type, arg)


async def _submit_with_retry_busy(
    submit: Callable[[], Awaitable[VideoJob]], retry_busy: float | None, quiet: bool
) -> VideoJob:
    """Submit; with `--retry-busy`, retry only the account's concurrency rejection."""
    if retry_busy is None:
        return await submit()
    from ..higgsfield.errors import HiggsfieldConcurrencyError

    deadline = time.monotonic() + retry_busy
    delay = _BUSY_INITIAL_S
    while True:
        try:
            return await submit()
        except HiggsfieldConcurrencyError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            wait_s = min(delay, remaining)
            if not quiet:
                note(f"Higgsfield concurrency limit reached; retrying in {wait_s:.0f}s")
            await anyio.sleep(wait_s)
            delay = min(delay * _BUSY_FACTOR, _BUSY_MAX_S)


async def _wait_one(
    command: str,
    video_id: str,
    *,
    session: PathSession,
    plan: OutputPlan | None,
    download: bool,
    output: str | None,
    timeout: float,
    interval: float | None,
    quiet: bool,
    started_at: float | None = None,
) -> tuple[int, dict[str, object]]:
    """Wait for one job (optionally downloading); never raises — returns (exit_code, envelope)."""
    from ..polling import WaitTimeoutError, wait_for_video
    from ..tools import video as video_tools

    printer = make_progress_printer(video_id, quiet=quiet)
    resume = _resume_command(video_id, download=download, output=output)

    try:
        result = await wait_for_video(
            video_id,
            timeout=timeout,
            interval=interval,
            on_progress=lambda s: printer(str(s["status"]), None),
        )
    except WaitTimeoutError as exc:
        extra: dict[str, object] = {"id": video_id}
        if isinstance(exc.last, dict):
            extra["last_status"] = str(exc.last.get("status"))
        return EXIT_TIMEOUT, error_envelope(command, "timeout", str(exc), resume=resume, extra=extra)
    except Exception as exc:  # noqa: BLE001 — fan-out callers must not lose sibling jobs
        error = _classify(exc)
        return error.exit_code, error_envelope(
            command, error.error_type, str(error), resume=error.resume, extra={"id": video_id, **(error.extra or {})}
        )

    status = str(result["status"])
    if status != "completed":
        if status == "nsfw":
            message = (
                f"Video job {video_id} was rejected by content moderation (not charged) — "
                "rephrase the prompt or change the inputs"
            )
        else:
            message = f"Video job {video_id} ended with status {status}"
        return EXIT_JOB_FAILED, error_envelope(
            command,
            "job_failed",
            message,
            extra={"id": video_id, "status": status, "details": result["error"]},
        )

    payload: dict[str, object] = dict(result)
    if download and plan is not None:
        try:
            dl = await video_tools.download_video(video_id, filename=plan.filename)
            final_path = await finalize_output(session, plan, dl["filename"])
        except Exception as exc:  # noqa: BLE001 — job completed; only the fetch failed
            error = _classify(exc)
            return error.exit_code, error_envelope(
                command,
                "download_error",
                f"Job completed but download failed: {error}",
                resume=resume,
                extra={"id": video_id, "status": "completed"},
            )
        payload["file"] = _file_payload(final_path, dl["format"])

    elapsed = None if started_at is None else time.monotonic() - started_at
    return 0, success_envelope(command, payload, elapsed_s=elapsed)


async def _submit_flow(
    command: str,
    submit: Callable[[], Awaitable[VideoJob]],
    *,
    session: PathSession,
    wait_flag: bool,
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
    retry_busy_arg: str | None,
    dry_run: bool,
    quiet: bool,
) -> int:
    """Shared create/edit/extend flow: submit, then optionally wait and download."""
    download_flag = download_flag or output is not None
    wait_flag = wait_flag or download_flag
    timeout = parse_duration(timeout_arg, "--timeout")
    interval = parse_duration(poll_arg, "--poll-interval")
    retry_busy = parse_duration(retry_busy_arg, "--retry-busy")

    plan: OutputPlan | None = None
    if download_flag and not dry_run:
        plan = plan_output(session, output, "video", quiet=quiet)
    install_overrides(session)

    started_at = time.monotonic()
    job = await _submit_with_retry_busy(submit, retry_busy, quiet)

    if dry_run or job["id"] is None:
        if wait_flag and not quiet:
            note("dry run: nothing was submitted, so there is nothing to wait for or download")
        emit(success_envelope(command, dict(job)))
        return 0

    if not quiet:
        cost = job["cost"]
        price = f" ~${cost['usd']:.2f}" if cost.get("usd") is not None else ""
        note(f"{job['id']} submitted ({job['status']}{price})")

    if not wait_flag:
        emit(success_envelope(command, dict(job)))
        return 0

    code, envelope = await _wait_one(
        command,
        job["id"],
        session=session,
        plan=plan,
        download=download_flag,
        output=output,
        timeout=DEFAULT_TIMEOUT_S if timeout is None else timeout,
        interval=interval,
        quiet=quiet,
        started_at=started_at,
    )
    if envelope.get("ok"):
        result = envelope.get("result")
        if isinstance(result, dict):
            result["cost"] = job["cost"]
    emit(envelope)
    return code


def _job_options(fn: Callable[..., object]) -> Callable[..., object]:
    """Options shared by create/edit/extend: pricing, extras, wait/download."""
    options = [
        click.option("--arg", "arg_pairs", multiple=True, help="Model parameter KEY=JSON (repeatable)."),
        click.option("--args", "args_file", default=None, help="Model parameters as a JSON object: @file or -."),
        click.option("--max-cost", "max_cost", type=float, default=None, help="Refuse if the estimate exceeds USD."),
        click.option("--dry-run", is_flag=True, help="Validate and price only; submit nothing."),
        click.option(
            "--retry-busy",
            "retry_busy_arg",
            default=None,
            help="On the account's concurrency limit, keep retrying the submit for this long (e.g. 5m).",
        ),
        click.option("--wait", "wait_flag", is_flag=True, help="Poll to a terminal state before exiting."),
        click.option("--download", "download_flag", is_flag=True, help="Save the file on completion (implies --wait)."),
        click.option("-o", "--output", default=None, help="Output file or directory (implies --download)."),
        click.option("--timeout", "timeout_arg", default=None, help="Wait deadline (90, 90s, 5m). Default 30m."),
        click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 2s→10s."),
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


@video.command("create")
@click.argument("prompt")
@click.option("--model", type=_MODEL, default=DEFAULT_VIDEO_MODEL, show_default=True, help="Curated id or slug.")
@click.option("--duration", type=int, default=None, help="Seconds (Seedance 4-30, Kling 3-15).")
@click.option("--seconds", "seconds_alias", type=int, default=None, hidden=True)
@click.option("--aspect-ratio", type=click.Choice(_ASPECT_RATIOS), default=None, help="Text-to-video only.")
@click.option("--resolution", type=click.Choice(_RESOLUTIONS), default=None)
@click.option("--audio/--no-audio", default=None, help="Generate a soundtrack (default: the model's).")
@click.option("--image", default=None, help="Start frame: path, bare filename in the media dir, or an hf_… id.")
@click.option("--input-ref", "input_ref_alias", default=None, hidden=True)
@click.option("--end-image", default=None, help="Last frame (needs --image).")
@_job_options
@click.pass_context
@run_async("video.create")
async def video_create(
    ctx: click.Context,
    prompt: str,
    model: str,
    duration: int | None,
    seconds_alias: int | None,
    aspect_ratio: str | None,
    resolution: str | None,
    audio: bool | None,
    image: str | None,
    input_ref_alias: str | None,
    end_image: str | None,
    arg_pairs: tuple[str, ...],
    args_file: str | None,
    max_cost: float | None,
    dry_run: bool,
    retry_busy_arg: str | None,
    wait_flag: bool,
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Submit a Higgsfield video job (async). PROMPT is inline text, @file, or - (stdin).

    \b
    Text-to-video, or image-to-video with --image. With an image, prompt only
    the motion: the image already carries subject, framing and style.
    One-shot: sanzaru video create "..." --duration 5 --max-cost 3 -o ./out/clip.mp4
    Price first: sanzaru video create "..." --resolution 480p --dry-run
    """
    from ..tools import video as video_tools

    state = get_state(ctx)
    prompt_text = read_content_arg(prompt, "PROMPT")
    seconds = cast("int | None", _merge_alias(duration, seconds_alias, "--duration and --seconds"))
    image_arg = cast("str | None", _merge_alias(image, input_ref_alias, "--image and --input-ref"))
    extra = _parse_extra(arg_pairs, args_file)
    session = PathSession()
    image_name = _resolve_media(session, image_arg, "reference", "--image") if image_arg else None
    end_name = _resolve_media(session, end_image, "reference", "--end-image") if end_image else None

    return await _submit_flow(
        "video.create",
        lambda: video_tools.create_video(
            prompt=prompt_text,
            model=model,
            duration=seconds,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            audio=audio,
            reference_image=image_name,
            end_image=end_name,
            extra=extra,
            max_cost_usd=max_cost,
            dry_run=dry_run,
        ),
        session=session,
        wait_flag=wait_flag,
        download_flag=download_flag,
        output=output,
        timeout_arg=timeout_arg,
        poll_arg=poll_arg,
        retry_busy_arg=retry_busy_arg,
        dry_run=dry_run,
        quiet=state.quiet,
    )


def _source_options(fn: Callable[..., object]) -> Callable[..., object]:
    options = [
        click.option("--model", type=_MODEL, default=DEFAULT_VIDEO_MODEL, show_default=True),
        click.option("--resolution", type=click.Choice(_RESOLUTIONS), default=None),
        click.option("--audio/--no-audio", default=None, help="Regenerate a soundtrack (default: the model's)."),
        click.option("--ref", "refs", multiple=True, help="Reference image to guide the result (repeatable)."),
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


@video.command("edit")
@click.argument("source")
@click.argument("prompt")
@_source_options
@_job_options
@click.pass_context
@run_async("video.edit")
async def video_edit(
    ctx: click.Context,
    source: str,
    prompt: str,
    model: str,
    resolution: str | None,
    audio: bool | None,
    refs: tuple[str, ...],
    arg_pairs: tuple[str, ...],
    args_file: str | None,
    max_cost: float | None,
    dry_run: bool,
    retry_busy_arg: str | None,
    wait_flag: bool,
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Re-render SOURCE per PROMPT, keeping its length and framing (async).

    \b
    SOURCE is an MP4 (path or media-dir filename) or the hf_… id of a completed
    job. The source is billed as well as the output — about twice its length.
    """
    from ..tools import video as video_tools

    state = get_state(ctx)
    prompt_text = read_content_arg(prompt, "PROMPT")
    extra = _parse_extra(arg_pairs, args_file)
    session = PathSession()
    source_name = _resolve_media(session, source, "video", "SOURCE")
    ref_names = [_resolve_media(session, ref, "reference", "--ref") for ref in refs] or None

    return await _submit_flow(
        "video.edit",
        lambda: video_tools.edit_video(
            prompt=prompt_text,
            source_video=source_name,
            model=model,
            resolution=resolution,
            audio=audio,
            reference_images=ref_names,
            extra=extra,
            max_cost_usd=max_cost,
            dry_run=dry_run,
        ),
        session=session,
        wait_flag=wait_flag,
        download_flag=download_flag,
        output=output,
        timeout_arg=timeout_arg,
        poll_arg=poll_arg,
        retry_busy_arg=retry_busy_arg,
        dry_run=dry_run,
        quiet=state.quiet,
    )


@video.command("extend")
@click.argument("source")
@click.argument("prompt")
@click.option("--duration", type=int, default=None, help="Seconds to add (Seedance 4-30; default 5).")
@_source_options
@_job_options
@click.pass_context
@run_async("video.extend")
async def video_extend(
    ctx: click.Context,
    source: str,
    prompt: str,
    duration: int | None,
    model: str,
    resolution: str | None,
    audio: bool | None,
    refs: tuple[str, ...],
    arg_pairs: tuple[str, ...],
    args_file: str | None,
    max_cost: float | None,
    dry_run: bool,
    retry_busy_arg: str | None,
    wait_flag: bool,
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Continue SOURCE by --duration seconds (async). The source is billed too.

    \b
    SOURCE is an MP4 (path or media-dir filename) or the hf_… id of a completed job.
    """
    from ..tools import video as video_tools

    state = get_state(ctx)
    prompt_text = read_content_arg(prompt, "PROMPT")
    extra = _parse_extra(arg_pairs, args_file)
    session = PathSession()
    source_name = _resolve_media(session, source, "video", "SOURCE")
    ref_names = [_resolve_media(session, ref, "reference", "--ref") for ref in refs] or None

    return await _submit_flow(
        "video.extend",
        lambda: video_tools.extend_video(
            prompt=prompt_text,
            source_video=source_name,
            duration=duration,
            model=model,
            resolution=resolution,
            audio=audio,
            reference_images=ref_names,
            extra=extra,
            max_cost_usd=max_cost,
            dry_run=dry_run,
        ),
        session=session,
        wait_flag=wait_flag,
        download_flag=download_flag,
        output=output,
        timeout_arg=timeout_arg,
        poll_arg=poll_arg,
        retry_busy_arg=retry_busy_arg,
        dry_run=dry_run,
        quiet=state.quiet,
    )


@video.command("status")
@click.argument("video_id")
@run_async("video.status")
async def video_status(video_id: str) -> int:
    """One-shot job status (never blocks; use `wait` to block)."""
    from ..tools import video as video_tools

    result = await video_tools.get_video_status(video_id)
    emit(success_envelope("video.status", dict(result)))
    return 0


@video.command("wait")
@click.argument("video_ids", nargs=-1, required=True)
@click.option("--download", "download_flag", is_flag=True, help="Download each video as it completes.")
@click.option("-o", "--output", default=None, help="Output file (single id) or directory.")
@click.option("--timeout", "timeout_arg", default=None, help="Deadline across all ids (default 30m).")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 2s→10s.")
@click.pass_context
@run_async("video.wait")
async def video_wait(
    ctx: click.Context,
    video_ids: tuple[str, ...],
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Poll job(s) to a terminal state (idempotent — safe to re-run after timeout).

    Multiple ids poll concurrently; one JSONL envelope per job, in completion order.
    """
    state = get_state(ctx)
    download_flag = download_flag or output is not None
    timeout = parse_duration(timeout_arg, "--timeout")
    interval = parse_duration(poll_arg, "--poll-interval")

    if (
        len(video_ids) > 1
        and output is not None
        and not output.endswith(("/", "\\"))
        and not pathlib.Path(output).expanduser().is_dir()
    ):
        raise CLIError("usage", "-o must be a directory when waiting on multiple jobs", exit_code=EXIT_USAGE)

    session = PathSession()
    plan: OutputPlan | None = None
    if download_flag:
        plan = plan_output(session, output, "video", quiet=state.quiet)
    install_overrides(session)

    single = len(video_ids) == 1
    codes: list[int] = []

    async def worker(vid: str) -> None:
        code, envelope = await _wait_one(
            "video.wait",
            vid,
            session=session,
            plan=plan,
            download=download_flag,
            output=output,
            timeout=DEFAULT_TIMEOUT_S if timeout is None else timeout,
            interval=interval,
            quiet=state.quiet,
        )
        codes.append(code)
        if single:
            emit(envelope)
        else:
            emit_line(envelope)

    async with anyio.create_task_group() as tg:
        for vid in video_ids:
            tg.start_soon(worker, vid)

    return aggregate_exit_code(codes)


@video.command("download")
@click.argument("video_id")
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("video.download")
async def video_download(ctx: click.Context, video_id: str, output: str | None) -> int:
    """Save a completed job's video (Higgsfield keeps outputs about 7 days)."""
    from ..tools import video as video_tools

    state = get_state(ctx)
    session = PathSession()
    plan = plan_output(session, output, "video", quiet=state.quiet)
    install_overrides(session)

    result = await video_tools.download_video(video_id, filename=plan.filename)
    final_path = await finalize_output(session, plan, result["filename"])
    emit(success_envelope("video.download", {"id": video_id, "file": _file_payload(final_path, result["format"])}))
    return 0


@video.command("cancel")
@click.argument("video_id")
@run_async("video.cancel")
async def video_cancel(video_id: str) -> int:
    """Cancel a job that is still queued (refunded). A started job cannot be canceled."""
    from ..tools import video as video_tools

    result = await video_tools.cancel_video(video_id)
    emit(success_envelope("video.cancel", dict(result)))
    return 0


@video.command("models")
@click.option("--catalog", is_flag=True, help="Also list every video model on the Higgsfield API (network call).")
@run_async("video.models")
async def video_models(catalog: bool) -> int:
    """The curated video models (validated and priced locally), and optionally the full catalog."""
    curated: list[dict[str, object]] = []
    for spec in VIDEO_MODELS.values():
        curated.append(
            {
                "id": spec.id,
                "default": spec.id == DEFAULT_VIDEO_MODEL,
                "family": spec.family,
                "operations": list(spec.operations),
                "slugs": dict(spec.slugs),
                "durations": list(spec.durations),
                "default_duration": spec.default_duration,
                "aspect_ratios": sorted(spec.text_aspect_ratios),
                "resolutions": sorted(spec.resolutions),
                "default_resolution": spec.default_resolution,
                "audio_param": spec.audio_param,
                "end_image_param": spec.end_image_param,
                "price_note": spec.price_note,
                "verified": spec.verified,
            }
        )
    result: dict[str, object] = {"curated": curated}
    if catalog:
        from ..config import get_higgsfield_client

        models = await get_higgsfield_client().models()
        result["catalog"] = [
            {"slug": m["slug"], "title": m.get("title"), "operations": m.get("operation_type")}
            for m in models
            if m.get("output_type") == "video"
        ]
    emit(success_envelope("video.models", result))
    return 0


@video.command("files")
@click.option("--pattern", default=None, help='Glob filter, e.g. "hf_*".')
@click.option("--type", "file_type", type=click.Choice(["mp4", "webm", "mov", "all"]), default="all", show_default=True)
@click.option(
    "--sort", "sort_by", type=click.Choice(["name", "size", "modified"]), default="modified", show_default=True
)
@click.option("--order", type=click.Choice(["asc", "desc"]), default="desc", show_default=True)
@click.option("--limit", type=int, default=50, show_default=True)
@run_async("video.files")
async def video_files(pattern: str | None, file_type: str, sort_by: str, order: str, limit: int) -> int:
    """List locally downloaded videos in the media dir."""
    from ..tools import video as video_tools

    result = await video_tools.list_local_videos(
        pattern=pattern,
        file_type=cast(Literal["mp4", "webm", "mov", "all"], file_type),
        sort_by=cast(Literal["name", "size", "modified"], sort_by),
        order=cast(Literal["asc", "desc"], order),
        limit=limit,
    )
    emit(success_envelope("video.files", result))
    return 0
