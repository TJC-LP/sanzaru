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


#: C0 controls, DEL, and the C1 range, rendered as visible ``\xNN`` text.
#: Tab and newline are the two controls stderr diagnostics legitimately carry.
_TERMINAL_ESCAPES = str.maketrans(
    {code: f"\\x{code:02x}" for code in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)) if code not in (0x09, 0x0A)}
)


def scrub_for_terminal(message: str) -> str:
    """Neutralize terminal control sequences in a string bound for stderr.

    Diagnostics interpolate strings this process did not author — act titles a
    planner model invented, error text an API returned — and a premise built
    from third-party material can steer those. A raw ESC on a TTY is not text:
    CSI can reposition the cursor and overwrite the cost warning or the resume
    hint printed a moment earlier, and OSC can retitle the window or push a
    command into the clipboard (OSC 52). Escaping rather than deleting keeps
    the diagnostic honest about what the string actually contained.

    Only C0/C1 and DEL are touched, by codepoint: accents, CJK and emoji are
    ordinary text and pass through untouched.
    """
    return message.translate(_TERMINAL_ESCAPES)


#: Continuation lines of a multi-line note are indented past the prefix column.
_CONTINUATION = "\n" + " " * len("sanzaru: ")


def note(message: str) -> None:
    """Write a human-readable diagnostic to stderr, prefixed ``sanzaru: ``.

    Every human-readable line in the CLI goes through here, which is why the
    scrub lives here rather than at the call sites that interpolate remote
    strings — one of those was already missed (act titles printed raw next to a
    sibling line that used ``!r``). stdout needs no equivalent: json.dumps
    escapes control characters on its own.

    Newlines are the one control the scrub lets through, because error text is
    legitimately multi-line (a missing-variables list, an API's message). They
    would also be the cheapest spoof: a remote string containing ``\\n`` followed
    by ``job failed — resume with: curl ... | sh`` used to come out as a second
    line at column 0 that read exactly like the CLI's own diagnostic. So only
    the first line carries the prefix; every continuation is indented under it.
    A line that starts at column 0 with ``sanzaru: `` was written by this
    process — nothing interpolated into a message can produce one.
    """
    click.echo(f"sanzaru: {scrub_for_terminal(message).replace(chr(10), _CONTINUATION)}", err=True)


def aggregate_exit_code(codes: list[int]) -> int:
    """Deterministic fan-out exit code.

    0 when everything succeeded; 6 (partial) on a mix of success and failure.
    When every input failed: 4 if any input timed out — resumable work remains,
    which is the actionable signal — otherwise the highest per-input code.
    """
    if not codes or all(code == 0 for code in codes):
        return EXIT_OK
    if any(code == 0 for code in codes):
        return EXIT_PARTIAL
    if EXIT_TIMEOUT in codes:
        return EXIT_TIMEOUT
    return max(codes)
