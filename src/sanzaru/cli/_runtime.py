# SPDX-License-Identifier: MIT
"""Shared runtime for CLI commands.

`run_async` bridges click's sync command model to the async tool layer:
one `anyio.run` per invocation, one shared AsyncOpenAI client installed via
`config.set_client()` (so poll loops reuse a single connection pool), and a
uniform exception → envelope + exit-code mapping. Overrides are always reset
in `finally`, so the process-global seams never leak.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import ParamSpec

import anyio
import click

if sys.version_info < (3, 11):  # pragma: no cover - 3.11+ has it as a builtin
    # anyio already requires this backport below 3.11, so it is always present.
    from exceptiongroup import BaseExceptionGroup

from ._output import (
    EXIT_CONFIG,
    EXIT_INTERRUPTED,
    EXIT_RUNTIME,
    EXIT_USAGE,
    emit,
    error_envelope,
    note,
)

P = ParamSpec("P")


@dataclass
class CLIState:
    """Root-level flags threaded to commands via click's ctx.obj."""

    quiet: bool = False
    verbose: bool = False


class CLIError(Exception):
    """A structured command failure: rendered as an error envelope + exit code.

    Command bodies raise this (directly or via except-clauses that add
    context like resume hints); `run_async` renders it.
    """

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        exit_code: int = EXIT_RUNTIME,
        resume: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.exit_code = exit_code
        self.resume = resume
        self.extra = extra


def _unwrap(exc: Exception) -> Exception:
    """Peel single-error ExceptionGroups so the real failure can be classified.

    Tools that fan out with `anyio.create_task_group()` (podcast segments, TTS
    chunks, multi-file transcription) surface a failing child as an
    ExceptionGroup. Without this, an actionable message — "lower
    SANZARU_ELEVENLABS_MAX_CONCURRENCY" — is reported as
    "internal: ExceptionGroup: unhandled errors in a TaskGroup".

    Groups with several distinct failures are left intact: picking one to
    represent the rest would hide the others.
    """
    seen = 0
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1 and seen < 10:
        inner = exc.exceptions[0]
        if not isinstance(inner, Exception):
            break
        exc = inner
        seen += 1
    return exc


def _classify(exc: Exception) -> CLIError:
    """Map uncaught exceptions from the tool layer onto the error contract."""
    # Imported lazily so `sanzaru --help` never pays the openai import.
    from openai import APIConnectionError, APIStatusError

    from ..exceptions import TTSError
    from ..polling import WaitTimeoutError

    exc = _unwrap(exc)

    if isinstance(exc, CLIError):
        return exc
    if isinstance(exc, WaitTimeoutError):
        return CLIError("timeout", str(exc), exit_code=4)
    if isinstance(exc, TTSError):
        # Provider-side failures (e.g. an ElevenLabs 429 after retries).
        return CLIError("api_error", str(exc))
    if isinstance(exc, APIStatusError):
        error_type = "not_found" if exc.status_code == 404 else "api_error"
        return CLIError(error_type, exc.message, extra={"status_code": exc.status_code})
    if isinstance(exc, APIConnectionError):
        return CLIError("api_error", str(exc))
    if isinstance(exc, (RuntimeError, ImportError)):
        # get_client/get_path raise RuntimeError for missing env; command
        # modules raise ImportError for missing optional extras.
        return CLIError("config", str(exc), exit_code=EXIT_CONFIG)
    if isinstance(exc, ValueError):
        # Tool-layer validation guards (e.g. transparent + gpt-image-2).
        return CLIError("usage", str(exc), exit_code=EXIT_USAGE)
    if isinstance(exc, BaseExceptionGroup):
        # Several parallel tasks failed differently — name each one rather than
        # reporting the opaque "unhandled errors in a TaskGroup".
        causes = "; ".join(f"{type(e).__name__}: {e}" for e in exc.exceptions[:5])
        extra_count = len(exc.exceptions) - 5
        if extra_count > 0:
            causes += f"; (+{extra_count} more)"
        return CLIError("internal", f"{len(exc.exceptions)} parallel tasks failed - {causes}")
    return CLIError("internal", f"{type(exc).__name__}: {exc}")


def run_async(command: str) -> Callable[[Callable[P, Coroutine[None, None, int]]], Callable[P, None]]:
    """Wrap an async command body returning an exit code into a sync click callback.

    The body emits its own envelope(s) on success and returns the exit code
    (0, or e.g. 5/6 for job-failed/partial outcomes). Failures raise CLIError
    (or any tool-layer exception, mapped by `_classify`); the wrapper renders
    the error envelope on stdout plus a one-line summary on stderr.
    """

    def decorator(fn: Callable[P, Coroutine[None, None, int]]) -> Callable[P, None]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            async def _main() -> int:
                client = None
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    from openai import AsyncOpenAI

                    from ..config import set_client

                    client = AsyncOpenAI(api_key=api_key)
                    set_client(client)
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 — single rendering point for the error contract
                    error = _classify(exc)
                    envelope = error_envelope(
                        command,
                        error.error_type,
                        str(error),
                        resume=error.resume,
                        extra=error.extra,
                    )
                    emit(envelope)
                    note(f"error ({error.error_type}): {error}")
                    return error.exit_code
                finally:
                    if client is not None:
                        from ..config import set_client

                        set_client(None)
                        await client.close()
                    # The ElevenLabs client is built lazily on first use rather
                    # than installed here, so this is a no-op (and costs no
                    # import) for every command that never touched it.
                    from ..config import close_elevenlabs_client

                    await close_elevenlabs_client()
                    from ..storage import set_storage_backend

                    set_storage_backend(None)

            try:
                code = anyio.run(_main)
            except KeyboardInterrupt:
                # No stdout write (avoids torn JSON; flushed JSONL lines stay valid).
                note("interrupted; any submitted jobs keep running server-side — re-run the wait command to resume")
                raise SystemExit(EXIT_INTERRUPTED) from None
            if code != 0:
                raise SystemExit(code)

        return wrapper

    return decorator


def get_state(ctx: click.Context) -> CLIState:
    """Fetch the root CLIState regardless of nesting depth."""
    state = ctx.find_object(CLIState)
    return state if state is not None else CLIState()


def parse_duration(value: str | None, flag: str) -> float | None:
    """Parse a duration flag: plain seconds, or `s`/`m` suffix ("90", "90s", "5m")."""
    if value is None:
        return None
    v = value.strip().lower()
    try:
        if v.endswith("m"):
            return float(v[:-1]) * 60.0
        if v.endswith("s"):
            return float(v[:-1])
        return float(v)
    except ValueError:
        raise CLIError(
            "usage", f"{flag}: invalid duration {value!r} (use e.g. 90, 90s, or 5m)", exit_code=EXIT_USAGE
        ) from None


def make_progress_printer(label: str, *, quiet: bool, heartbeat_s: float = 30.0) -> Callable[[str, int | None], None]:
    """Build a stderr progress reporter: one line per state change plus a heartbeat.

    Output is greppable and line-based (works identically for TTYs and captured
    agent transcripts): `sanzaru: video_x in_progress 42% t=95s`.
    """
    start = time.monotonic()
    last_status: str | None = None
    last_emit = start

    def printer(status: str, progress: int | None) -> None:
        nonlocal last_status, last_emit
        if quiet:
            return
        now = time.monotonic()
        if status != last_status or (now - last_emit) >= heartbeat_s:
            pct = f" {progress}%" if progress is not None else ""
            note(f"{label} {status}{pct} t={int(now - start)}s")
            last_status, last_emit = status, now

    return printer
