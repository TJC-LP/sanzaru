# SPDX-License-Identifier: MIT
"""`sanzaru video` — Sora video jobs.

Workflow: create → wait → download. `create -o out.mp4` composes all three
(`-o` implies `--download` implies `--wait`). Waits are idempotent and
resumable: on exit 4 (timeout) the job keeps running server-side — re-run
the `resume` command from the error envelope.
"""

from __future__ import annotations

import pathlib
import shlex
import time
from collections.abc import Awaitable, Callable
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
    from openai.types import Video

VideoVariant = Literal["video", "thumbnail", "spritesheet"]

DEFAULT_TIMEOUT_S = 1800.0  # Sora jobs run minutes; parity with polling.DEFAULT_VIDEO_TIMEOUT

_SIZES = ["720x1280", "1280x720", "1024x1792", "1792x1024"]
_VARIANTS = ["video", "thumbnail", "spritesheet"]


@click.group()
def video() -> None:
    """Sora video jobs. Workflow: create → wait → download (create -o does all three)."""


def _resume_command(video_id: str, *, download: bool, output: str | None, variant: str) -> str:
    parts = ["sanzaru", "video", "wait", video_id]
    if download:
        parts.append("--download")
    if variant != "video":
        parts += ["--variant", variant]
    if output is not None:
        parts += ["-o", output]
    return " ".join(shlex.quote(p) for p in parts)


def _file_payload(final_path: str, variant: str) -> dict[str, object]:
    payload: dict[str, object] = {"path": final_path, "variant": variant}
    fp = pathlib.Path(final_path)
    if fp.is_file():
        payload["bytes"] = fp.stat().st_size
    return payload


async def _wait_one(
    command: str,
    video_id: str,
    *,
    session: PathSession,
    plan: OutputPlan | None,
    download: bool,
    variant: str,
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
    resume = _resume_command(video_id, download=download, output=output, variant=variant)

    try:
        result = await wait_for_video(
            video_id,
            timeout=timeout,
            interval=interval,
            on_progress=lambda v: printer(str(v.status), v.progress),
        )
    except WaitTimeoutError as exc:
        extra: dict[str, object] = {"id": video_id}
        if exc.last is not None and not isinstance(exc.last, dict):
            extra["last_status"] = str(exc.last.status)
            extra["last_progress"] = exc.last.progress
        return EXIT_TIMEOUT, error_envelope(command, "timeout", str(exc), resume=resume, extra=extra)
    except Exception as exc:  # noqa: BLE001 — fan-out callers must not lose sibling jobs
        error = _classify(exc)
        return error.exit_code, error_envelope(
            command, error.error_type, str(error), resume=error.resume, extra={"id": video_id}
        )

    if str(result.status) != "completed":
        details = result.error.model_dump(mode="json") if result.error is not None else None
        return EXIT_JOB_FAILED, error_envelope(
            command,
            "job_failed",
            f"Video job {video_id} ended with status {result.status}",
            extra={"id": video_id, "status": str(result.status), "details": details},
        )

    payload: dict[str, object] = result.model_dump(mode="json")
    if download and plan is not None:
        try:
            dl = await video_tools.download_video(video_id, filename=plan.filename, variant=cast(VideoVariant, variant))
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
        payload["file"] = _file_payload(final_path, variant)

    elapsed = None if started_at is None else time.monotonic() - started_at
    return 0, success_envelope(command, payload, elapsed_s=elapsed)


async def _submit_flow(
    command: str,
    submit: Callable[[], Awaitable[Video]],
    *,
    session: PathSession,
    wait_flag: bool,
    download_flag: bool,
    variant: str,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
    quiet: bool,
) -> int:
    """Shared create/remix flow: submit, then optionally wait and download."""
    download_flag = download_flag or output is not None
    wait_flag = wait_flag or download_flag
    timeout = parse_duration(timeout_arg, "--timeout")
    interval = parse_duration(poll_arg, "--poll-interval")

    plan: OutputPlan | None = None
    if download_flag:
        plan = plan_output(session, output, "video", quiet=quiet)
    install_overrides(session)

    started_at = time.monotonic()
    job = await submit()
    if not quiet:
        note(f"{job.id} submitted ({job.status})")

    if not wait_flag:
        emit(success_envelope(command, job))
        return 0

    code, envelope = await _wait_one(
        command,
        job.id,
        session=session,
        plan=plan,
        download=download_flag,
        variant=variant,
        output=output,
        timeout=DEFAULT_TIMEOUT_S if timeout is None else timeout,
        interval=interval,
        quiet=quiet,
        started_at=started_at,
    )
    emit(envelope)
    return code


@video.command("create")
@click.argument("prompt")
@click.option("--model", type=click.Choice(["sora-2", "sora-2-pro"]), default="sora-2", show_default=True)
@click.option("--seconds", type=click.Choice(["4", "8", "12"]), default=None, help="Clip duration.")
@click.option("--size", type=click.Choice(_SIZES), default=None, help="Resolution (1024x1792/1792x1024 are pro-only).")
@click.option("--input-ref", default=None, help="Reference image: path, or bare filename in the media dir.")
@click.option("--wait", "wait_flag", is_flag=True, help="Poll to a terminal state before exiting.")
@click.option("--download", "download_flag", is_flag=True, help="Fetch the artifact on completion (implies --wait).")
@click.option("--variant", type=click.Choice(_VARIANTS), default="video", show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (implies --download).")
@click.option("--timeout", "timeout_arg", default=None, help="Wait deadline (90, 90s, 5m). Default 30m.")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 5s→20s.")
@click.pass_context
@run_async("video.create")
async def video_create(
    ctx: click.Context,
    prompt: str,
    model: str,
    seconds: str | None,
    size: str | None,
    input_ref: str | None,
    wait_flag: bool,
    download_flag: bool,
    variant: str,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Submit a Sora video job (async). PROMPT is inline text, @file, or - (stdin).

    \b
    With --input-ref, keep the prompt motion-focused: the image already
    carries character/setting/style; describe only what happens next.
    One-shot: sanzaru video create "..." --seconds 8 -o ./out/clip.mp4
    """
    from openai.types import VideoModel, VideoSeconds, VideoSize

    from ..tools import video as video_tools

    state = get_state(ctx)
    prompt_text = read_content_arg(prompt, "PROMPT")
    session = PathSession()
    ref_name = resolve_input(session, input_ref, "reference", "--input-ref") if input_ref else None

    return await _submit_flow(
        "video.create",
        lambda: video_tools.create_video(
            prompt=prompt_text,
            model=cast(VideoModel, model),
            seconds=cast("VideoSeconds | None", seconds),
            size=cast("VideoSize | None", size),
            input_reference_filename=ref_name,
        ),
        session=session,
        wait_flag=wait_flag,
        download_flag=download_flag,
        variant=variant,
        output=output,
        timeout_arg=timeout_arg,
        poll_arg=poll_arg,
        quiet=state.quiet,
    )


@video.command("remix")
@click.argument("video_id")
@click.argument("prompt")
@click.option("--wait", "wait_flag", is_flag=True, help="Poll to a terminal state before exiting.")
@click.option("--download", "download_flag", is_flag=True, help="Fetch the artifact on completion (implies --wait).")
@click.option("--variant", type=click.Choice(_VARIANTS), default="video", show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (implies --download).")
@click.option("--timeout", "timeout_arg", default=None, help="Wait deadline (90, 90s, 5m). Default 30m.")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 5s→20s.")
@click.pass_context
@run_async("video.remix")
async def video_remix(
    ctx: click.Context,
    video_id: str,
    prompt: str,
    wait_flag: bool,
    download_flag: bool,
    variant: str,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Submit a remix of a completed video (async, new job id). Same flags as create."""
    from ..tools import video as video_tools

    state = get_state(ctx)
    prompt_text = read_content_arg(prompt, "PROMPT")

    return await _submit_flow(
        "video.remix",
        lambda: video_tools.remix_video(video_id, prompt_text),
        session=PathSession(),
        wait_flag=wait_flag,
        download_flag=download_flag,
        variant=variant,
        output=output,
        timeout_arg=timeout_arg,
        poll_arg=poll_arg,
        quiet=state.quiet,
    )


@video.command("status")
@click.argument("video_id")
@run_async("video.status")
async def video_status(video_id: str) -> int:
    """One-shot job status + progress (never blocks; use `wait` to block)."""
    from ..tools import video as video_tools

    result = await video_tools.get_video_status(video_id)
    emit(success_envelope("video.status", result))
    return 0


@video.command("wait")
@click.argument("video_ids", nargs=-1, required=True)
@click.option("--download", "download_flag", is_flag=True, help="Download each artifact as it completes.")
@click.option("--variant", type=click.Choice(_VARIANTS), default="video", show_default=True)
@click.option("-o", "--output", default=None, help="Output file (single id) or directory.")
@click.option("--timeout", "timeout_arg", default=None, help="Deadline across all ids (default 30m).")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts 5s→20s.")
@click.pass_context
@run_async("video.wait")
async def video_wait(
    ctx: click.Context,
    video_ids: tuple[str, ...],
    download_flag: bool,
    variant: str,
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
            variant=variant,
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

    if all(code == 0 for code in codes):
        return 0
    if any(code == 0 for code in codes):
        return EXIT_PARTIAL
    return codes[0]


@video.command("download")
@click.argument("video_id")
@click.option("--variant", type=click.Choice(_VARIANTS), default="video", show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("video.download")
async def video_download(ctx: click.Context, video_id: str, variant: str, output: str | None) -> int:
    """Download a completed job's artifact (video, thumbnail, or spritesheet)."""
    from ..tools import video as video_tools

    state = get_state(ctx)
    session = PathSession()
    plan = plan_output(session, output, "video", quiet=state.quiet)
    install_overrides(session)

    result = await video_tools.download_video(video_id, filename=plan.filename, variant=cast(VideoVariant, variant))
    final_path = await finalize_output(session, plan, result["filename"])
    emit(success_envelope("video.download", {"id": video_id, "file": _file_payload(final_path, variant)}))
    return 0


@video.command("list")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--after", default=None, help="Pagination cursor (last id from the previous page).")
@click.option("--order", type=click.Choice(["asc", "desc"]), default="desc", show_default=True)
@run_async("video.list")
async def video_list(limit: int, after: str | None, order: str) -> int:
    """List video jobs in OpenAI's cloud (newest first by default)."""
    from ..tools import video as video_tools

    result = await video_tools.list_videos(limit=limit, after=after, order=cast(Literal["asc", "desc"], order))
    emit(success_envelope("video.list", result))
    return 0


@video.command("delete")
@click.argument("video_id")
@run_async("video.delete")
async def video_delete(video_id: str) -> int:
    """Permanently delete a video from OpenAI storage."""
    from ..tools import video as video_tools

    result = await video_tools.delete_video(video_id)
    emit(success_envelope("video.delete", result))
    return 0


@video.command("files")
@click.option("--pattern", default=None, help='Glob filter, e.g. "sora*".')
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
