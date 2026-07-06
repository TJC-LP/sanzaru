# SPDX-License-Identifier: MIT
"""`sanzaru image` — image generation.

Two paths, mirroring the MCP tools:
- `generate` / `edit`: synchronous Images API — blocks until rendered,
  returns the file + token usage. RECOMMENDED for one-off images.
- `create` → `wait` → `download`: async Responses API job — use for
  refinement chains (--previous-id) and firing many jobs in parallel.
"""

from __future__ import annotations

import pathlib
import shlex
import time
from typing import TYPE_CHECKING, Literal, cast

import anyio
import click

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
    EXIT_PARTIAL,
    EXIT_TIMEOUT,
    EXIT_USAGE,
    emit,
    emit_line,
    error_envelope,
    note,
    success_envelope,
)
from ._runtime import CLIError, _classify, get_state, make_progress_printer, parse_duration, run_async

if TYPE_CHECKING:
    from openai.types.responses.tool_param import ImageGeneration

DEFAULT_TIMEOUT_S = 600.0  # Responses-API image jobs typically finish in 10-60s

_GENERATE_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "2560x1440",
    "1440x2560",
    "3840x2160",
    "2160x3840",
]
_QUALITIES = ["auto", "low", "medium", "high"]
_BACKGROUNDS = ["auto", "transparent", "opaque"]
_FORMATS = ["png", "jpeg", "webp"]
_VIDEO_SIZES = ["720x1280", "1280x720", "1024x1792", "1792x1024"]

GenerateSize = Literal[
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "2560x1440",
    "1440x2560",
    "3840x2160",
    "2160x3840",
]
Quality = Literal["auto", "low", "medium", "high"]
Background = Literal["auto", "transparent", "opaque"]
OutputFormat = Literal["png", "jpeg", "webp"]


@click.group()
def image() -> None:
    """Image generation.

    \b
    generate   Synchronous; returns the finished file + token usage.
               RECOMMENDED for one-off images.
    create     Async job; use for refinement chains (--previous-id) and
               firing many jobs in parallel. Then: wait / download.
    """


def _resume_command(response_id: str, *, download: bool, output: str | None) -> str:
    parts = ["sanzaru", "image", "wait", response_id]
    if download:
        parts.append("--download")
    if output is not None:
        parts += ["-o", output]
    return " ".join(shlex.quote(p) for p in parts)


def _file_payload(final_path: str) -> dict[str, object]:
    payload: dict[str, object] = {"path": final_path}
    fp = pathlib.Path(final_path)
    if fp.is_file():
        payload["bytes"] = fp.stat().st_size
    return payload


async def wait_one_image(
    command: str,
    response_id: str,
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
    """Wait for one Responses-API image job; never raises — returns (exit_code, envelope)."""
    from ..polling import WaitTimeoutError, wait_for_image
    from ..tools import image as image_tools

    printer = make_progress_printer(response_id, quiet=quiet)
    resume = _resume_command(response_id, download=download, output=output)

    try:
        result = await wait_for_image(
            response_id,
            timeout=timeout,
            interval=interval,
            on_progress=lambda r: printer(r["status"], None),
        )
    except WaitTimeoutError as exc:
        extra: dict[str, object] = {"id": response_id}
        if isinstance(exc.last, dict):
            extra["last_status"] = exc.last["status"]
        return EXIT_TIMEOUT, error_envelope(command, "timeout", str(exc), resume=resume, extra=extra)
    except Exception as exc:  # noqa: BLE001 — fan-out callers must not lose sibling jobs
        error = _classify(exc)
        return error.exit_code, error_envelope(
            command, error.error_type, str(error), resume=error.resume, extra={"id": response_id}
        )

    if result["status"] != "completed":
        return EXIT_JOB_FAILED, error_envelope(
            command,
            "job_failed",
            f"Image job {response_id} ended with status {result['status']}",
            extra={"id": response_id, "status": result["status"]},
        )

    payload: dict[str, object] = dict(result)
    if download and plan is not None:
        try:
            dl = await image_tools.download_image(response_id, filename=plan.filename)
            final_path = await finalize_output(session, plan, dl["filename"])
        except Exception as exc:  # noqa: BLE001 — job completed; only the fetch failed
            error = _classify(exc)
            return error.exit_code, error_envelope(
                command,
                "download_error",
                f"Job completed but download failed: {error}",
                resume=resume,
                extra={"id": response_id, "status": "completed"},
            )
        payload["size"] = dl["size"]
        payload["format"] = dl["format"]
        payload["file"] = _file_payload(final_path)

    elapsed = None if started_at is None else time.monotonic() - started_at
    return 0, success_envelope(command, payload, elapsed_s=elapsed)


@image.command("create")
@click.argument("prompt")
@click.option("--model", default="gpt-5.2", show_default=True, help="Mainline model driving the image tool.")
@click.option("--image-model", default=None, help="Image model in the tool config (default gpt-image-2).")
@click.option("--size", default=None, help="e.g. 1536x1024 (gpt-image-2 allows most 16-multiples).")
@click.option("--quality", type=click.Choice(_QUALITIES), default=None)
@click.option("--background", type=click.Choice(_BACKGROUNDS), default=None)
@click.option("--output-format", type=click.Choice(_FORMATS), default=None)
@click.option("--moderation", type=click.Choice(["auto", "low"]), default=None)
@click.option("--input-fidelity", type=click.Choice(["high", "low"]), default=None, help="gpt-image-1.5 only.")
@click.option("--previous-id", default=None, help="Refine a previous response (iterative chains).")
@click.option("--input-image", "input_images", multiple=True, help="Reference image (path or media-dir filename).")
@click.option("--mask", default=None, help="PNG mask with alpha channel (requires --input-image).")
@click.option("--wait", "wait_flag", is_flag=True, help="Poll to a terminal state before exiting.")
@click.option("--download", "download_flag", is_flag=True, help="Fetch the image on completion (implies --wait).")
@click.option("-o", "--output", default=None, help="Output file or directory (implies --download).")
@click.option("--timeout", "timeout_arg", default=None, help="Wait deadline (90, 90s, 5m). Default 10m.")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 2s→10s.")
@click.pass_context
@run_async("image.create")
async def image_create(
    ctx: click.Context,
    prompt: str,
    model: str,
    image_model: str | None,
    size: str | None,
    quality: str | None,
    background: str | None,
    output_format: str | None,
    moderation: str | None,
    input_fidelity: str | None,
    previous_id: str | None,
    input_images: tuple[str, ...],
    mask: str | None,
    wait_flag: bool,
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Submit a Responses-API image job (async). PROMPT is inline text, @file, or - (stdin).

    \b
    Refinement chain: create → wait → create --previous-id <id> → ...
    One-shot: sanzaru image create "..." -o ./art/hero.png
    """
    from ..tools import image as image_tools

    state = get_state(ctx)
    download_flag = download_flag or output is not None
    wait_flag = wait_flag or download_flag
    timeout = parse_duration(timeout_arg, "--timeout")
    interval = parse_duration(poll_arg, "--poll-interval")

    prompt_text = read_content_arg(prompt, "PROMPT")
    session = PathSession()
    ref_names = [resolve_input(session, img, "reference", "--input-image") for img in input_images]
    mask_name = resolve_input(session, mask, "reference", "--mask") if mask else None

    plan: OutputPlan | None = None
    if download_flag:
        plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    config: dict[str, object] = {"type": "image_generation"}
    if image_model is not None:
        config["model"] = image_model
    if size is not None:
        config["size"] = size
    if quality is not None:
        config["quality"] = quality
    if background is not None:
        config["background"] = background
    if output_format is not None:
        config["output_format"] = output_format
    if moderation is not None:
        config["moderation"] = moderation
    if input_fidelity is not None:
        config["input_fidelity"] = input_fidelity

    started_at = time.monotonic()
    job = await image_tools.create_image(
        prompt=prompt_text,
        model=model,
        tool_config=cast("ImageGeneration", config),
        previous_response_id=previous_id,
        input_images=ref_names or None,
        mask_filename=mask_name,
    )
    if not state.quiet:
        note(f"{job['id']} submitted ({job['status']})")

    if not wait_flag:
        emit(success_envelope("image.create", job))
        return 0

    code, envelope = await wait_one_image(
        "image.create",
        job["id"],
        session=session,
        plan=plan,
        download=download_flag,
        output=output,
        timeout=DEFAULT_TIMEOUT_S if timeout is None else timeout,
        interval=interval,
        quiet=state.quiet,
        started_at=started_at,
    )
    emit(envelope)
    return code


@image.command("status")
@click.argument("response_id")
@run_async("image.status")
async def image_status(response_id: str) -> int:
    """One-shot job status (never blocks; use `wait` to block)."""
    from ..tools import image as image_tools

    result = await image_tools.get_image_status(response_id)
    emit(success_envelope("image.status", result))
    return 0


@image.command("wait")
@click.argument("response_ids", nargs=-1, required=True)
@click.option("--download", "download_flag", is_flag=True, help="Download each image as it completes.")
@click.option("-o", "--output", default=None, help="Output file (single id) or directory.")
@click.option("--timeout", "timeout_arg", default=None, help="Deadline across all ids (default 10m).")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 2s→10s.")
@click.pass_context
@run_async("image.wait")
async def image_wait(
    ctx: click.Context,
    response_ids: tuple[str, ...],
    download_flag: bool,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Poll image job(s) to a terminal state (idempotent — safe to re-run).

    Multiple ids poll concurrently; one JSONL envelope per job, in completion order.
    """
    state = get_state(ctx)
    download_flag = download_flag or output is not None
    timeout = parse_duration(timeout_arg, "--timeout")
    interval = parse_duration(poll_arg, "--poll-interval")

    if (
        len(response_ids) > 1
        and output is not None
        and not output.endswith(("/", "\\"))
        and not pathlib.Path(output).expanduser().is_dir()
    ):
        raise CLIError("usage", "-o must be a directory when waiting on multiple jobs", exit_code=EXIT_USAGE)

    session = PathSession()
    plan: OutputPlan | None = None
    if download_flag:
        plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    single = len(response_ids) == 1
    codes: list[int] = []

    async def worker(rid: str) -> None:
        code, envelope = await wait_one_image(
            "image.wait",
            rid,
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
        for rid in response_ids:
            tg.start_soon(worker, rid)

    if all(code == 0 for code in codes):
        return 0
    if any(code == 0 for code in codes):
        return EXIT_PARTIAL
    return codes[0]


@image.command("download")
@click.argument("response_id")
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("image.download")
async def image_download(ctx: click.Context, response_id: str, output: str | None) -> int:
    """Download a completed image job's result."""
    from ..tools import image as image_tools

    state = get_state(ctx)
    session = PathSession()
    plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    result = await image_tools.download_image(response_id, filename=plan.filename)
    final_path = await finalize_output(session, plan, result["filename"])
    payload: dict[str, object] = dict(result)
    payload["file"] = _file_payload(final_path)
    emit(success_envelope("image.download", payload))
    return 0


@image.command("generate")
@click.argument("prompts", nargs=-1, required=True)
@click.option("--model", default=None, help="Image model (default gpt-image-2).")
@click.option("--size", type=click.Choice(_GENERATE_SIZES), default="auto", show_default=True)
@click.option("--quality", type=click.Choice(_QUALITIES), default="auto", show_default=True)
@click.option("--background", type=click.Choice(_BACKGROUNDS), default="auto", show_default=True)
@click.option("--output-format", type=click.Choice(_FORMATS), default="png", show_default=True)
@click.option("--moderation", type=click.Choice(["auto", "low"]), default="auto", show_default=True)
@click.option("--count", type=click.IntRange(1, 16), default=1, show_default=True, help="Images per prompt.")
@click.option("--concurrency", type=click.IntRange(1, 16), default=4, show_default=True)
@click.option("-o", "--output", default=None, help="Output file (single image) or directory.")
@click.pass_context
@run_async("image.generate")
async def image_generate(
    ctx: click.Context,
    prompts: tuple[str, ...],
    model: str | None,
    size: str,
    quality: str,
    background: str,
    output_format: str,
    moderation: str,
    count: int,
    concurrency: int,
    output: str | None,
) -> int:
    """Generate image(s) synchronously (Images API) — no polling, returns files + usage.

    \b
    Multiple prompts (and --count > 1) fan out concurrently; one JSONL
    envelope per image in completion order. RECOMMENDED for one-off images.
    Batch: sanzaru image generate "icon" "banner" --quality high -o ./art/
    """
    from ..config import DEFAULT_IMAGE_MODEL
    from ..tools import images_api

    state = get_state(ctx)
    prompt_texts = [read_content_arg(p, "PROMPT") for p in prompts]
    jobs = [(prompt, n) for prompt in prompt_texts for n in range(count)]
    total = len(jobs)

    if (
        total > 1
        and output is not None
        and not output.endswith(("/", "\\"))
        and not pathlib.Path(output).expanduser().is_dir()
    ):
        raise CLIError("usage", "-o must be a directory when generating multiple images", exit_code=EXIT_USAGE)

    session = PathSession()
    plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    model_name = model or DEFAULT_IMAGE_MODEL
    single = total == 1
    batch_stamp = int(time.time())
    limiter = anyio.CapacityLimiter(concurrency)
    codes: list[int] = []

    async def worker(index: int, prompt: str) -> None:
        # Explicit per-image filenames: the tool's timestamped autogen name
        # can collide inside a same-second batch.
        if plan.filename is not None:
            filename: str | None = plan.filename
        elif single:
            filename = None
        else:
            filename = f"img_{batch_stamp}_{index}.{output_format}"
        envelope: dict[str, object]
        async with limiter:
            started = time.monotonic()
            try:
                result = await images_api.generate_image(
                    prompt=prompt,
                    model=model_name,
                    size=cast(GenerateSize, size),
                    quality=cast(Quality, quality),
                    background=cast(Background, background),
                    output_format=cast(OutputFormat, output_format),
                    moderation=cast(Literal["auto", "low"], moderation),
                    filename=filename,
                )
                final_path = await finalize_output(session, plan, result.filename)
                payload: dict[str, object] = result.model_dump(mode="json")
                payload["file"] = _file_payload(final_path)
                code, envelope = 0, success_envelope("image.generate", payload, elapsed_s=time.monotonic() - started)
            except Exception as exc:  # noqa: BLE001 — batch siblings must keep going
                error = _classify(exc)
                code, envelope = (
                    error.exit_code,
                    error_envelope("image.generate", error.error_type, str(error), extra=error.extra),
                )
        envelope["input"] = {"index": index, "prompt": prompt}
        codes.append(code)
        if single:
            emit(envelope)
        else:
            emit_line(envelope)

    async with anyio.create_task_group() as tg:
        for index, (prompt, _) in enumerate(jobs):
            tg.start_soon(worker, index, prompt)

    if all(code == 0 for code in codes):
        return 0
    if any(code == 0 for code in codes):
        return EXIT_PARTIAL
    return codes[0]


@image.command("edit")
@click.argument("prompt")
@click.option("--input-image", "input_images", multiple=True, required=True, help="Image(s) to edit/compose.")
@click.option("--mask", default=None, help="PNG mask with alpha channel.")
@click.option("--model", default=None, help="Image model (default gpt-image-2).")
@click.option("--size", type=click.Choice(_GENERATE_SIZES), default="auto", show_default=True)
@click.option("--quality", type=click.Choice(_QUALITIES), default="auto", show_default=True)
@click.option("--background", type=click.Choice(_BACKGROUNDS), default="auto", show_default=True)
@click.option("--output-format", type=click.Choice(_FORMATS), default="png", show_default=True)
@click.option("--input-fidelity", type=click.Choice(["high", "low"]), default=None, help="gpt-image-1.5 only.")
@click.option("-o", "--output", default=None, help="Output file or directory.")
@click.pass_context
@run_async("image.edit")
async def image_edit(
    ctx: click.Context,
    prompt: str,
    input_images: tuple[str, ...],
    mask: str | None,
    model: str | None,
    size: str,
    quality: str,
    background: str,
    output_format: str,
    input_fidelity: str | None,
    output: str | None,
) -> int:
    """Edit/compose existing images synchronously (Images API)."""
    from ..config import DEFAULT_IMAGE_MODEL
    from ..tools import images_api

    state = get_state(ctx)
    prompt_text = read_content_arg(prompt, "PROMPT")
    session = PathSession()
    ref_names = [resolve_input(session, img, "reference", "--input-image") for img in input_images]
    mask_name = resolve_input(session, mask, "reference", "--mask") if mask else None
    plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    started = time.monotonic()
    result = await images_api.edit_image(
        prompt=prompt_text,
        input_images=ref_names,
        model=model or DEFAULT_IMAGE_MODEL,
        mask_filename=mask_name,
        size=cast(GenerateSize, size),
        quality=cast(Quality, quality),
        background=cast(Background, background),
        output_format=cast(OutputFormat, output_format),
        input_fidelity=cast("Literal['high', 'low'] | None", input_fidelity),
        filename=plan.filename,
    )
    final_path = await finalize_output(session, plan, result.filename)
    payload: dict[str, object] = result.model_dump(mode="json")
    payload["file"] = _file_payload(final_path)
    emit(success_envelope("image.edit", payload, elapsed_s=time.monotonic() - started))
    return 0


@image.command("prepare")
@click.argument("input_image")
@click.option("--size", type=click.Choice(_VIDEO_SIZES), required=True, help="Target Sora dimensions.")
@click.option("--mode", type=click.Choice(["crop", "pad", "rescale"]), default="crop", show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (default: alongside input).")
@click.pass_context
@run_async("image.prepare")
async def image_prepare(ctx: click.Context, input_image: str, size: str, mode: str, output: str | None) -> int:
    """Resize an image to Sora dimensions (crop preserves ratio; pad letterboxes)."""
    from openai.types import VideoSize

    from ..tools import reference as reference_tools

    state = get_state(ctx)
    session = PathSession()
    input_name = resolve_input(session, input_image, "reference", "INPUT_IMAGE")
    plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    result = await reference_tools.prepare_reference_image(
        input_filename=input_name,
        target_size=cast(VideoSize, size),
        output_filename=plan.filename,
        resize_mode=cast(Literal["crop", "pad", "rescale"], mode),
    )
    final_path = await finalize_output(session, plan, result["output_filename"])
    payload: dict[str, object] = dict(result)
    payload["file"] = _file_payload(final_path)
    emit(success_envelope("image.prepare", payload))
    return 0


@image.command("files")
@click.option("--pattern", default=None, help='Glob filter, e.g. "hero*".')
@click.option(
    "--type", "file_type", type=click.Choice(["jpeg", "png", "webp", "all"]), default="all", show_default=True
)
@click.option(
    "--sort", "sort_by", type=click.Choice(["name", "size", "modified"]), default="modified", show_default=True
)
@click.option("--order", type=click.Choice(["asc", "desc"]), default="desc", show_default=True)
@click.option("--limit", type=int, default=50, show_default=True)
@run_async("image.files")
async def image_files(pattern: str | None, file_type: str, sort_by: str, order: str, limit: int) -> int:
    """List images in the media dir (references + downloaded generations)."""
    from ..tools import reference as reference_tools

    result = await reference_tools.list_reference_images(
        pattern=pattern,
        file_type=cast(Literal["jpeg", "png", "webp", "all"], file_type),
        sort_by=cast(Literal["name", "size", "modified"], sort_by),
        order=cast(Literal["asc", "desc"], order),
        limit=limit,
    )
    emit(success_envelope("image.files", result))
    return 0
