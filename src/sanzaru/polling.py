# SPDX-License-Identifier: MIT
"""Neutral polling helpers for long-running OpenAI jobs.

The MCP tools expose the raw create → status → download primitives and leave
polling to the caller; these helpers implement the blocking wait loop for
callers that need a terminal state (the CLI today, potentially MCP wait tools
later). Pure async — no CLI/framework imports.

Behavior:
- Adaptive backoff by default (video: 5s ×1.5 → 20s cap; image: 2s ×1.5 → 10s
  cap), with ±10% jitter; passing ``interval`` fixes the cadence instead.
- Transient API errors (connection/timeout errors, 408/409/429, 5xx) are
  retried in place until the deadline; anything else (e.g. 404) propagates.
- On deadline expiry, :class:`WaitTimeoutError` carries the last-seen payload.
  The job keeps running server-side — waiting again with the same ID resumes.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio
from openai import APIConnectionError, APIStatusError
from openai.types import Video

from .tools import image as image_tools
from .tools import video as video_tools
from .types import ImageResponse

DEFAULT_VIDEO_TIMEOUT = 1800.0
DEFAULT_IMAGE_TIMEOUT = 600.0

_VIDEO_INITIAL_INTERVAL = 5.0
_VIDEO_MAX_INTERVAL = 20.0
_IMAGE_INITIAL_INTERVAL = 2.0
_IMAGE_MAX_INTERVAL = 10.0

_BACKOFF_FACTOR = 1.5
_JITTER = 0.1

# Statuses that mean "keep polling". Everything else is terminal: video ends at
# completed|failed; Responses jobs can also end cancelled|incomplete. "unknown"
# (a null status early in a Responses job's life) is treated as still-active.
_ACTIVE_STATUSES = frozenset({"queued", "in_progress", "unknown"})

# HTTP statuses retried while the deadline allows (connection errors likewise).
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

_T = TypeVar("_T", Video, ImageResponse)


class WaitTimeoutError(TimeoutError):
    """Deadline expired while the job was still running (it keeps running server-side).

    Attributes:
        last: Most recent status payload seen before the deadline, if any.
    """

    def __init__(self, message: str, last: Video | ImageResponse | None = None) -> None:
        super().__init__(message)
        self.last = last


def _is_retryable(exc: APIStatusError) -> bool:
    return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500


async def _wait_until_terminal(
    fetch: Callable[[], Awaitable[_T]],
    is_terminal: Callable[[_T], bool],
    *,
    describe: str,
    timeout: float,
    interval: float | None,
    initial_interval: float,
    max_interval: float,
    on_progress: Callable[[_T], None] | None,
) -> _T:
    deadline = anyio.current_time() + timeout
    delay = initial_interval if interval is None else interval
    last: _T | None = None

    while True:
        try:
            last = await fetch()
        except APIConnectionError:  # noqa: S110 — includes timeouts; retry until deadline
            pass
        except APIStatusError as exc:
            if not _is_retryable(exc):
                raise
        else:
            if on_progress is not None:
                on_progress(last)
            if is_terminal(last):
                return last

        remaining = deadline - anyio.current_time()
        if remaining <= 0:
            raise WaitTimeoutError(f"{describe} still running after {timeout:.0f}s", last)

        # A caller-fixed interval is used verbatim; jitter and backoff growth
        # apply only to the adaptive schedule.
        if interval is None:
            sleep_for = delay * (1.0 + random.uniform(-_JITTER, _JITTER))
            delay = min(delay * _BACKOFF_FACTOR, max_interval)
        else:
            sleep_for = interval
        await anyio.sleep(min(sleep_for, remaining))


async def wait_for_video(
    video_id: str,
    *,
    timeout: float = DEFAULT_VIDEO_TIMEOUT,
    interval: float | None = None,
    on_progress: Callable[[Video], None] | None = None,
) -> Video:
    """Poll a Sora video job until it reaches a terminal state.

    Args:
        video_id: The video ID from create_video/remix_video
        timeout: Overall deadline in seconds (default 30 minutes)
        interval: Fixed poll interval in seconds; None enables adaptive backoff
        on_progress: Called with each successfully fetched Video (for progress display)

    Returns:
        The terminal Video — status "completed" OR "failed" (callers decide how
        to surface failure; the wait itself succeeded)

    Raises:
        WaitTimeoutError: Deadline expired; carries the last-seen Video
        openai.APIStatusError: Non-retryable API error (e.g. 404 unknown ID)
        RuntimeError: If OPENAI_API_KEY not set
    """
    return await _wait_until_terminal(
        lambda: video_tools.get_video_status(video_id),
        lambda video: video.status not in _ACTIVE_STATUSES,
        describe=f"Video job {video_id}",
        timeout=timeout,
        interval=interval,
        initial_interval=_VIDEO_INITIAL_INTERVAL,
        max_interval=_VIDEO_MAX_INTERVAL,
        on_progress=on_progress,
    )


async def wait_for_image(
    response_id: str,
    *,
    timeout: float = DEFAULT_IMAGE_TIMEOUT,
    interval: float | None = None,
    on_progress: Callable[[ImageResponse], None] | None = None,
) -> ImageResponse:
    """Poll a Responses-API image job until it reaches a terminal state.

    Args:
        response_id: The response ID from create_image
        timeout: Overall deadline in seconds (default 10 minutes)
        interval: Fixed poll interval in seconds; None enables adaptive backoff
        on_progress: Called with each successfully fetched ImageResponse

    Returns:
        The terminal ImageResponse — status "completed" OR "failed"/"cancelled"/
        "incomplete" (callers decide how to surface failure)

    Raises:
        WaitTimeoutError: Deadline expired; carries the last-seen ImageResponse
        openai.APIStatusError: Non-retryable API error (e.g. 404 unknown ID)
        RuntimeError: If OPENAI_API_KEY not set
    """
    return await _wait_until_terminal(
        lambda: image_tools.get_image_status(response_id),
        lambda resp: resp["status"] not in _ACTIVE_STATUSES,
        describe=f"Image job {response_id}",
        timeout=timeout,
        interval=interval,
        initial_interval=_IMAGE_INITIAL_INTERVAL,
        max_interval=_IMAGE_MAX_INTERVAL,
        on_progress=on_progress,
    )
