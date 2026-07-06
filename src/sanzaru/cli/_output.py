# SPDX-License-Identifier: MIT
"""JSON output contract for the agent CLI.

stdout carries exactly one JSON envelope per input (JSONL, streamed in
completion order, for fan-out commands) and nothing else; progress, hints,
and human-readable error summaries go to stderr. A TTY changes formatting
only (pretty-printed vs compact), never structure.

Envelope shape::

    {"v": 1, "ok": true,  "command": "video.create", "result": {...}, "elapsed_s": 184.2}
    {"v": 1, "ok": false, "command": "video.wait", "error": {"type": "timeout", "message": "..."},
     "resume": "sanzaru video wait video_x --download", ...}
"""

from __future__ import annotations

import json
import pathlib
import sys

import click

ENVELOPE_VERSION = 1

# Exit codes — the CLI's contract with agents (documented in docs/cli.md).
EXIT_OK = 0
EXIT_RUNTIME = 1  # API/network/write failure, unknown ID
EXIT_USAGE = 2  # bad flags/arguments (click's own usage errors also exit 2)
EXIT_CONFIG = 3  # missing OPENAI_API_KEY, missing optional extra, bad media dir
EXIT_TIMEOUT = 4  # --timeout exceeded; job still running server-side (resumable)
EXIT_JOB_FAILED = 5  # job reached a failed terminal state server-side
EXIT_PARTIAL = 6  # fan-out with >=1 success and >=1 failure
EXIT_INTERRUPTED = 130  # SIGINT; job keeps running (resume hint on stderr)


def _json_default(obj: object) -> object:
    import pydantic  # lazy: keeps `sanzaru --help` off the pydantic import path

    if isinstance(obj, pydantic.BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, pathlib.Path):
        return str(obj)
    # No silent str() fallback — contract tests catch new unrenderable types.
    raise TypeError(f"Unrenderable type in CLI output: {type(obj)!r}")


def render(envelope: dict[str, object], *, pretty: bool | None = None) -> str:
    """Serialize an envelope; pretty-print only on a TTY (tuples become arrays)."""
    if pretty is None:
        pretty = sys.stdout.isatty()
    return json.dumps(envelope, default=_json_default, indent=2 if pretty else None)


def success_envelope(
    command: str,
    result: object,
    *,
    elapsed_s: float | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    envelope: dict[str, object] = {"v": ENVELOPE_VERSION, "ok": True, "command": command, "result": result}
    if elapsed_s is not None:
        envelope["elapsed_s"] = round(elapsed_s, 1)
    if extra:
        envelope.update(extra)
    return envelope


def error_envelope(
    command: str,
    error_type: str,
    message: str,
    *,
    resume: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    envelope: dict[str, object] = {
        "v": ENVELOPE_VERSION,
        "ok": False,
        "command": command,
        "error": {"type": error_type, "message": message},
    }
    if resume is not None:
        envelope["resume"] = resume
    if extra:
        envelope.update(extra)
    return envelope


def emit(envelope: dict[str, object]) -> None:
    """Write one envelope to stdout (the result payload — never anything else)."""
    click.echo(render(envelope))


def emit_line(envelope: dict[str, object]) -> None:
    """Write one JSONL line to stdout (always compact — one envelope per line)."""
    click.echo(render(envelope, pretty=False))


def note(message: str) -> None:
    """Write a human-readable diagnostic line to stderr."""
    click.echo(f"sanzaru: {message}", err=True)
