# SPDX-License-Identifier: MIT
"""Local bounds on Higgsfield traffic.

`make_limiter()` bounds the HTTP work *one* call fans out — uploads of several
reference images, `wait_for(download=true)` over up to 20 ids. It is built per
call and never at module scope: an `anyio.CapacityLimiter` binds to the event
loop it is first used on.

It does not enforce Higgsfield's per-account cap of 4 in-flight jobs, and
cannot: that cap is counted server-side across every client of the account (the
Higgsfield app and connector included), and a submit returns in milliseconds,
long before the job it started stops counting. The cap is surfaced as
`HiggsfieldConcurrencyError` instead.
"""

from __future__ import annotations

import logging
import os

import anyio

logger = logging.getLogger("sanzaru")

MAX_CONCURRENCY_ENV = "SANZARU_HIGGSFIELD_MAX_CONCURRENCY"
DEFAULT_MAX_CONCURRENCY = 4
MAX_UPLOAD_MB_ENV = "SANZARU_HIGGSFIELD_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 200


def _positive_int(env: str, default: int) -> int:
    raw = os.getenv(env)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        logger.warning("%s=%r is not a positive integer - using %d", env, raw, default)
        return default
    return value


def make_limiter() -> anyio.CapacityLimiter:
    """A fresh limiter for one call's uploads/downloads/polls (default 4)."""
    return anyio.CapacityLimiter(_positive_int(MAX_CONCURRENCY_ENV, DEFAULT_MAX_CONCURRENCY))


def max_upload_bytes() -> int:
    """Largest input sanzaru will read and upload (default 200 MB).

    Checked against `storage.stat` before the read: the Databricks backend
    holds the whole object in memory, and the upload itself is one PUT.
    """
    return _positive_int(MAX_UPLOAD_MB_ENV, DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024
