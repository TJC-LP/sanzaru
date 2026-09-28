# SPDX-License-Identifier: MIT
"""Neutral polling helpers for long-running jobs, whatever provider runs them.

The MCP tools expose the raw create → status → download primitives; these
helpers implement the blocking wait loop for callers that need a terminal state
(`wait_for`, the CLI). Pure async — no CLI/framework imports.

The loop itself knows nothing about a provider. What differs per provider is
passed in: how to fetch a status, which statuses are terminal, and which
exceptions are transient (`is_transient`) — OpenAI signals those with
`APIConnectionError` and 408/409/429/5xx, Higgsfield with
`HiggsfieldAPIError.transient`. A status payload may be a pydantic model, a
TypedDict, or anything else; the loop only hands it back.

Behavior:
- Adaptive backoff by default (image: 2s ×1.5 → 10s cap; Higgsfield video: the
  same, which is its documented cadence), with ±10% jitter; passing
  ``interval`` fixes the cadence instead.
- Transient errors are retried in place until the deadline; anything else
  (e.g. 404) propagates.
- On deadline expiry, :class:`WaitTimeoutError` carries the last-seen payload.
  The job keeps running server-side — waiting again with the same ID resumes.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

import anyio
from openai import APIConnectionError, APIStatusError

from .higgsfield.errors import HiggsfieldAPIError
from .tools import image as image_tools
from .tools import video as video_tools
from .types import ImageResponse

DEFAULT_VIDEO_TIMEOUT = 1800.0
DEFAULT_IMAGE_TIMEOUT = 600.0

_IMAGE_INITIAL_INTERVAL = 2.0
_IMAGE_MAX_INTERVAL = 10.0

# Higgsfield's documented polling guidance: start at 2s, grow ×1.5, cap at 10s.
DEFAULT_HF_VIDEO_INTERVAL_INITIAL = 2.0
DEFAULT_HF_VIDEO_INTERVAL_MAX = 10.0

_BACKOFF_FACTOR = 1.5
_JITTER = 0.1

# Statuses that mean "keep polling". Everything else is terminal: video ends at
# completed|failed; Responses jobs can also end cancelled|incomplete. "unknown"
# (a null status early in a Responses job's life) is treated as still-active.
_ACTIVE_STATUSES = frozenset({"queued", "in_progress", "unknown"})

# HTTP statuses retried while the deadline allows (connection errors likewise).
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

_T = TypeVar("_T")


class WaitTimeoutError(TimeoutError, Generic[_T]):
    """Deadline expired while the job was still running (it keeps running server-side).

    Generic over the status payload, which is whatever the provider's fetch
    returns (a pydantic model, a TypedDict, ...) — the loop never inspects it.

    Attributes:
        last: Most recent status payload seen before the deadline, if any.
    """

    def __init__(self, message: str, last: _T | None = None) -> None:
        super().__init__(message)
        self.last = last


def _openai_transient(exc: BaseException) -> bool:
    """OpenAI's transient failures: connection/timeout errors, 408/409/429, 5xx."""
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500
    return False


def _higgsfield_transient(exc: BaseException) -> bool:
    """Higgsfield's transient failures (5xx, 429, transport) — for status reads only."""
    return isinstance(exc, HiggsfieldAPIError) and exc.transient


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
    is_transient: Callable[[BaseException], bool],
) -> _T:
    deadline = anyio.current_time() + timeout
    delay = initial_interval if interval is None else interval
    last: _T | None = None

    while True:
        try:
            last = await fetch()
        except Exception as exc:
            # Transient failures are retried until the deadline; the job is
            # still running server-side whether or not this poll got through.
            if not is_transient(exc):
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
    on_progress: Callable[[video_tools.VideoStatus], None] | None = None,
) -> video_tools.VideoStatus:
    """Poll a Higgsfield video job (`hf_…`) until it reaches a terminal state.

    Args:
        video_id: The `hf_…` id from create_video / edit_video / extend_video
        timeout: Overall deadline in seconds (default 30 minutes)
        interval: Fixed poll interval in seconds; None enables adaptive backoff
            (Higgsfield's documented 2 s x1.5 up to 10 s)
        on_progress: Called with each successfully fetched status

    Returns:
        The terminal status — completed, failed, nsfw or canceled (callers decide
        how to surface a non-completed job; the wait itself succeeded)

    Raises:
        WaitTimeoutError: Deadline expired; carries the last-seen status
        HiggsfieldAPIError: Non-retryable API error (e.g. 404 unknown id)
        ValueError: A malformed id, or a retired Sora `video_…` id
        RuntimeError: If HF_KEY is not set
    """
    return await _wait_until_terminal(
        lambda: video_tools.get_video_status(video_id),
        lambda status: status["done"],
        describe=f"Video job {video_id}",
        timeout=timeout,
        interval=interval,
        initial_interval=DEFAULT_HF_VIDEO_INTERVAL_INITIAL,
        max_interval=DEFAULT_HF_VIDEO_INTERVAL_MAX,
        on_progress=on_progress,
        is_transient=_higgsfield_transient,
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
        is_transient=_openai_transient,
    )


async def wait_until_done(
    fetch: Callable[[], Awaitable[_T]],
    is_terminal: Callable[[_T], bool],
    *,
    describe: str,
    timeout: float,
    interval: float | None = None,
    initial_interval: float = DEFAULT_HF_VIDEO_INTERVAL_INITIAL,
    max_interval: float = DEFAULT_HF_VIDEO_INTERVAL_MAX,
    on_progress: Callable[[_T], None] | None = None,
    is_transient: Callable[[BaseException], bool] = _higgsfield_transient,
) -> _T:
    """Poll any job until `is_terminal` says it is done.

    The provider-agnostic entry point: the caller supplies the status fetch,
    the terminal test and the transient-error rule. Defaults are Higgsfield's
    (2s ×1.5 → 10s, `HiggsfieldAPIError.transient`).

    Args:
        fetch: Awaited once per poll; returns the current status payload
        is_terminal: True when the payload is a final state (success or failure)
        describe: Job description used in the timeout message
        timeout: Overall deadline in seconds
        interval: Fixed poll interval in seconds; None enables adaptive backoff
        initial_interval: First adaptive delay
        max_interval: Adaptive delay cap
        on_progress: Called with each successfully fetched payload
        is_transient: True for exceptions worth retrying until the deadline

    Returns:
        The terminal payload (callers decide how to surface failure)

    Raises:
        WaitTimeoutError: Deadline expired; carries the last-seen payload
        Exception: Whatever `fetch` raised, when `is_transient` rejects it
    """
    return await _wait_until_terminal(
        fetch,
        is_terminal,
        describe=describe,
        timeout=timeout,
        interval=interval,
        initial_interval=initial_interval,
        max_interval=max_interval,
        on_progress=on_progress,
        is_transient=is_transient,
    )
