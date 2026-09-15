# SPDX-License-Identifier: MIT
"""wait_for: block server-side on a batch of long-running job ids.

`create_video`, `remix_video` and `create_image` return immediately with an id,
and until now the MCP surface offered only `get_*_status` to find out when the
job finished — so the model ran the poll loop itself, one round trip per check.
This tool moves the loop into the server, where `polling.py` already knows the
right cadence (adaptive backoff, transient-error retries), and answers in one
call with every job's terminal state.

Two properties matter more than the loop itself:

- **A deadline returns, it does not raise.** The job keeps running on OpenAI's
  side whether or not anyone is waiting, so an expired wait is not a failure —
  it is "not yet". Each job comes back with `timed_out=True` and its last-seen
  status, and the same call with the same ids resumes. That is the CLI's exit-4
  contract (`sanzaru wait`), carried over.
- **Progress is reported on every poll**, not just at the end. Clients abort a
  tool call that is silent for too long (Claude Code: 5 minutes over HTTP), and
  a progress notification is what resets that clock — so the wait can safely
  run past the idle window, and clients that render progress show it.

Per-job errors (a 404 for an unknown id, a non-retryable API error) are
reported on that job, not raised, so one bad id does not discard the others.
Malformed ids are rejected before any request, as everywhere else.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import Literal

import anyio
from openai import APIStatusError

from ..config import logger
from ..polling import WaitTimeoutError, wait_for_image, wait_for_video
from ..types import WaitJob, WaitResult
from . import image as image_tools
from . import video as video_tools

JobKind = Literal["video", "image"]

DEFAULT_WAIT_TIMEOUT = 240.0
"""Under every known client idle window (Claude Code aborts a silent HTTP tool
call at 300 s), so a wait that sends no progress still returns before the
client gives up. With progress flowing the cap below applies instead."""

MAX_WAIT_TIMEOUT = 1800.0
"""Sora's own worst case (`polling.DEFAULT_VIDEO_TIMEOUT`). Longer waits should
be re-issued, which also re-validates that the caller is still there."""

MAX_WAIT_IDS = 20

ProgressCallback = Callable[[int, int, str], Awaitable[None]]
"""(settled, total, message) — called on every poll and every completion."""


def job_kind(job_id: str) -> JobKind:
    """Infer the job type from the id prefix, as the CLI's `wait` does."""
    if job_id.startswith("video_"):
        return "video"
    if job_id.startswith("resp_"):
        return "image"
    raise ValueError(f"cannot infer job type from id {job_id!r}: expected a video_* or resp_* id")


def _describe(exc: APIStatusError) -> str:
    return f"HTTP {exc.status_code}: {exc.message}" if getattr(exc, "message", None) else f"HTTP {exc.status_code}"


async def wait_for(
    ids: list[str],
    *,
    timeout: float = DEFAULT_WAIT_TIMEOUT,
    download: bool = False,
    on_progress: ProgressCallback | None = None,
) -> WaitResult:
    """Wait for video (`video_*`) and image (`resp_*`) jobs to reach a terminal state.

    Args:
        ids: One to `MAX_WAIT_IDS` job ids, mixed types allowed, waited on concurrently.
        timeout: Overall deadline in seconds, 1..`MAX_WAIT_TIMEOUT`. Jobs still
            running at the deadline come back with `timed_out=True`; nothing raises.
        download: Download each job that completes within the deadline to the
            media directory (video: the mp4; image: the png), in the same call.
        on_progress: Awaited with (settled, total, message) on every poll result
            and every completion. Exceptions from it are logged and ignored — a
            client's progress quirk must not end the wait.

    Returns:
        A `WaitResult`: one `WaitJob` per id in input order, plus `all_done`
        (every job terminal), `timed_out` (any job still running) and the
        deadline used.

    Raises:
        ValueError: Empty/too many/duplicate ids, an id with an unknown prefix,
            or a timeout outside its range — all before any request is made.
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    if not ids:
        raise ValueError("wait_for needs at least one job id")
    if len(ids) > MAX_WAIT_IDS:
        raise ValueError(f"wait_for accepts at most {MAX_WAIT_IDS} ids per call (got {len(ids)})")
    if len(set(ids)) != len(ids):
        raise ValueError("wait_for ids must be unique")
    if not 1.0 <= timeout <= MAX_WAIT_TIMEOUT:
        raise ValueError(f"timeout must be between 1 and {MAX_WAIT_TIMEOUT:.0f} seconds (got {timeout})")
    kinds: dict[str, JobKind] = {job_id: job_kind(job_id) for job_id in ids}

    jobs: dict[str, WaitJob] = {
        job_id: WaitJob(
            id=job_id,
            kind=kinds[job_id],
            status="unknown",
            done=False,
            timed_out=False,
            progress=None,
            error=None,
            download=None,
        )
        for job_id in ids
    }
    settled = 0
    total = len(ids)

    # polling.py reports progress through a *sync* callback; the client-facing
    # report is async. A memory stream bridges them: polls push a message, one
    # reporter task awaits `on_progress` for each. Dropping a message when the
    # buffer is full is fine — the next poll sends another.
    send, receive = anyio.create_memory_object_stream[str](max_buffer_size=64)

    def note(job_id: str, status: str, progress: int | None = None) -> None:
        job = jobs[job_id]
        job["status"] = status
        if progress is not None:
            job["progress"] = progress
        suffix = f" {progress}%" if progress is not None else ""
        # A full buffer just skips one message; the next poll sends another.
        with contextlib.suppress(anyio.WouldBlock):
            send.send_nowait(f"{job_id}: {status}{suffix}")

    async def report(message: str) -> None:
        if on_progress is None:
            return
        try:
            await on_progress(settled, total, message)
        except Exception:  # noqa: BLE001 - the wait must survive a client's progress quirk
            logger.debug("wait_for: progress report failed", exc_info=True)

    async def reporter() -> None:
        async with receive:
            async for message in receive:
                await report(message)

    async def one(job_id: str) -> None:
        nonlocal settled
        job = jobs[job_id]
        try:
            if kinds[job_id] == "video":
                video = await wait_for_video(
                    job_id,
                    timeout=timeout,
                    on_progress=lambda v: note(job_id, v.status, v.progress),
                )
                job["status"] = video.status
                job["progress"] = video.progress
                job["done"] = True
                if download and video.status == "completed":
                    job["download"] = await video_tools.download_video(job_id)
            else:
                response = await wait_for_image(
                    job_id,
                    timeout=timeout,
                    on_progress=lambda r: note(job_id, r["status"]),
                )
                job["status"] = response["status"]
                job["done"] = True
                if download and response["status"] == "completed":
                    job["download"] = await image_tools.download_image(job_id)
        except WaitTimeoutError as exc:
            job["timed_out"] = True
            last = exc.last
            if isinstance(last, dict):
                job["status"] = last["status"]
            elif last is not None:
                job["status"] = last.status
        except APIStatusError as exc:
            # Non-retryable (polling already retried the transient ones): an
            # unknown id, a permission problem. Reported on the job so the other
            # ids in the batch still get their answer.
            job["error"] = _describe(exc)
            job["done"] = True
        finally:
            settled += 1
            note(job_id, job["status"], job["progress"])

    async with anyio.create_task_group() as tg:
        tg.start_soon(reporter)
        async with anyio.create_task_group() as waits:
            for job_id in ids:
                waits.start_soon(one, job_id)
        await send.aclose()

    ordered = [jobs[job_id] for job_id in ids]
    return WaitResult(
        jobs=ordered,
        all_done=all(job["done"] for job in ordered),
        timed_out=any(job["timed_out"] for job in ordered),
        timeout_s=timeout,
    )
