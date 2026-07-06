# SPDX-License-Identifier: MIT
"""Top-level `wait` (cross-type job poller) and `capabilities` (introspection)."""

from __future__ import annotations

import pathlib

import anyio
import click

from ._io import OutputPlan, PathSession, install_overrides, plan_output
from ._output import EXIT_USAGE, aggregate_exit_code, emit, emit_line, success_envelope
from ._runtime import CLIError, get_state, parse_duration, run_async

DEFAULT_TIMEOUT_S = 1800.0  # generous cross-type default (Sora pace)


def _job_kind(job_id: str, forced: str) -> str:
    if forced != "auto":
        return forced
    if job_id.startswith("video_"):
        return "video"
    if job_id.startswith("resp_"):
        return "image"
    raise CLIError(
        "usage",
        f"cannot infer job type from id {job_id!r} (expected video_* or resp_*); pass --type video|image",
        exit_code=EXIT_USAGE,
    )


@click.command("wait")
@click.argument("job_ids", nargs=-1, required=True)
@click.option("--type", "forced_type", type=click.Choice(["auto", "video", "image"]), default="auto", show_default=True)
@click.option("--download", "download_flag", is_flag=True, help="Download each artifact as it completes.")
@click.option("--variant", type=click.Choice(["video", "thumbnail", "spritesheet"]), default="video", show_default=True)
@click.option("-o", "--output", default=None, help="Output file (single id) or directory.")
@click.option("--timeout", "timeout_arg", default=None, help="Deadline across all ids (default 30m).")
@click.option("--poll-interval", "poll_arg", default=None, help="Fixed poll interval; default adapts per type.")
@click.pass_context
@run_async("wait")
async def wait_command(
    ctx: click.Context,
    job_ids: tuple[str, ...],
    forced_type: str,
    download_flag: bool,
    variant: str,
    output: str | None,
    timeout_arg: str | None,
    poll_arg: str | None,
) -> int:
    """Poll mixed job(s) to a terminal state — video_* and resp_* ids in one call.

    Type is inferred from the id prefix (video_* → video, resp_* → image);
    one JSONL envelope per job in completion order. Idempotent and resumable.
    One lightweight poll loop per id (a few requests/minute each) — dozens of
    ids in one call are fine.
    """
    from .image import wait_one_image
    from .video import _wait_one as wait_one_video

    state = get_state(ctx)
    download_flag = download_flag or output is not None
    timeout = parse_duration(timeout_arg, "--timeout")
    interval = parse_duration(poll_arg, "--poll-interval")

    kinds = {job_id: _job_kind(job_id, forced_type) for job_id in job_ids}

    if (
        len(job_ids) > 1
        and output is not None
        and not output.endswith(("/", "\\"))
        and not pathlib.Path(output).expanduser().is_dir()
    ):
        raise CLIError("usage", "-o must be a directory when waiting on multiple jobs", exit_code=EXIT_USAGE)

    session = PathSession()
    video_plan: OutputPlan | None = None
    image_plan: OutputPlan | None = None
    if download_flag:
        if "video" in kinds.values():
            video_plan = plan_output(session, output, "video", quiet=state.quiet)
        if "image" in kinds.values():
            image_plan = plan_output(session, output, "reference", quiet=state.quiet)
    install_overrides(session)

    single = len(job_ids) == 1
    codes: list[int] = []

    async def worker(job_id: str) -> None:
        if kinds[job_id] == "video":
            code, envelope = await wait_one_video(
                "wait",
                job_id,
                session=session,
                plan=video_plan,
                download=download_flag,
                variant=variant,
                output=output,
                timeout=DEFAULT_TIMEOUT_S if timeout is None else timeout,
                interval=interval,
                quiet=state.quiet,
            )
        else:
            code, envelope = await wait_one_image(
                "wait",
                job_id,
                session=session,
                plan=image_plan,
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
        for job_id in job_ids:
            tg.start_soon(worker, job_id)

    return aggregate_exit_code(codes)


@click.command("capabilities")
@run_async("capabilities")
async def capabilities() -> int:
    """Machine-readable environment report: version, features, paths, commands.

    Needs no API key — safe as an agent's first call to discover what works here.
    """
    import os
    from importlib.metadata import PackageNotFoundError, version

    from ..config import is_path_configured
    from ..features import get_available_features
    from . import cli

    try:
        pkg_version = version("sanzaru")
    except PackageNotFoundError:
        pkg_version = "unknown"

    features: dict[str, object] = {}
    for name, available in get_available_features().items():
        entry: dict[str, object] = {"available": available}
        if not available:
            path_type = "reference" if name == "image" else name
            if not is_path_configured(path_type):  # type: ignore[arg-type]
                entry["reason"] = "media path not configured (set SANZARU_MEDIA_PATH or the individual path var)"
            else:
                entry["reason"] = f"optional dependencies missing (install sanzaru[{name}])"
        features[name] = entry

    paths = {
        path_type: is_path_configured(path_type)  # type: ignore[arg-type]
        for path_type in ("video", "reference", "audio")
    }

    commands: dict[str, list[str]] = {}
    for name, command in sorted(cli.commands.items()):
        if isinstance(command, click.Group):
            commands[name] = sorted(command.commands)
        else:
            commands[name] = []

    result: dict[str, object] = {
        "version": pkg_version,
        "features": features,
        "paths_configured": paths,
        "storage_backend": os.getenv("STORAGE_BACKEND", "local"),
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "commands": commands,
    }
    emit(success_envelope("capabilities", result))
    return 0
