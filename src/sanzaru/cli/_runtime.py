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
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import ParamSpec

import anyio
import click

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


def _classify(exc: Exception) -> CLIError:
    """Map uncaught exceptions from the tool layer onto the error contract."""
    # Imported lazily so `sanzaru --help` never pays the openai import.
    from openai import APIConnectionError, APIStatusError

    from ..polling import WaitTimeoutError

    if isinstance(exc, CLIError):
        return exc
    if isinstance(exc, WaitTimeoutError):
        return CLIError("timeout", str(exc), exit_code=4)
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
