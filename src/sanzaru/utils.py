# SPDX-License-Identifier: MIT
"""Shared utility functions for the Sora MCP server."""

import re
import time
from typing import Literal

# An OpenAI resource id is a *path segment*, not an opaque token: the generated
# SDK builds `/videos/{video_id}`, `/videos/{video_id}/content` and
# `/responses/{response_id}` by plain f-string with no percent-encoding, and
# httpx then applies RFC 3986 dot-segment removal to whatever comes out. So
# video_id="../files/file-XYZ/content?" is not a lookup that fails — it is a
# GET of a *different* endpoint, authenticated with the operator's key, with the
# trailing "?" pushing the rest into the query string. Same trick reaches
# DELETE /v1/files/... and DELETE /v1/models/....
#
# Allowlist instead of escaping: percent-encoding the segment would also work,
# but every real id is already in this alphabet, so rejecting is both stricter
# and self-documenting at the call site. ":" is in because fine-tune-style ids
# carry it; "." is deliberately out, which makes ".." unrepresentable rather
# than a rule of its own. 128 is far above any id OpenAI issues.
_RESOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9_:-]{1,128}")


def validate_resource_id(value: str, param: str) -> str:
    """Check an OpenAI resource id before it is interpolated into a request path.

    Args:
        value: Caller-supplied id (video id, response id, ...)
        param: Parameter name, used in the error message

    Returns:
        The id unchanged, so call sites can validate inline

    Raises:
        ValueError: If the id is empty, over 128 characters, or contains
            anything outside letters, digits, "_", "-" and ":"
    """
    # isinstance too: these ids arrive as JSON from an MCP client, so a non-str
    # would otherwise reach re as a TypeError (exit 1) instead of usage (exit 2).
    if not isinstance(value, str) or not _RESOURCE_ID_PATTERN.fullmatch(value):
        # Repr is truncated: the rejected value is attacker-controlled and ends
        # up in logs and tool output.
        raise ValueError(
            f"{param}={value!r:.80} is not a valid OpenAI resource id; expected only letters, digits, "
            "'_', '-' or ':' (max 128 characters)"
        )
    return value


# The one name shape that is unambiguously run bookkeeping: `simrun_<id>.json`.
# It covers every run id, and nothing a person would name an episode collides
# with it.
#
# Act checkpoints (`<slug>_<runid>_<actid>.{mp3,json}`) are deliberately NOT
# matched here. The obvious rule for them — "<something>_<8 hex>_<something>" —
# cannot be written without false positives, because eight hex digits is also
# what a date looks like: it rejected `interview_20250826_part1.mp3` and
# `standup_20240101_notes.json`, which are exactly the names people give
# recordings. Refusing an operator's ordinary filename to defend a case that
# already self-heals is the wrong trade — a clobbered checkpoint fails to decode
# or fails its signature, and `_load_checkpoint` re-records it. Checkpoint
# integrity is enforced where it can be exact: the signature over run id, act id
# and an audio digest.
_RESERVED_NAME_PATTERNS = (re.compile(r"simrun_[A-Za-z0-9_-]+\.json", re.IGNORECASE),)


def reject_reserved_name(filename: str, param: str = "output filename") -> str:
    """Refuse a caller-chosen name that belongs to a run's bookkeeping.

    Every audio-producing tool writes into one flat directory under a name the
    caller picks, and the storage write is an unconditional truncate. On a
    shared deployment that made another session's run state addressable: naming
    your episode `simrun_<their id>.json` replaced their manifest with mp3
    bytes, stranding audio they had already paid for with no delete or undo in
    the storage protocol to recover it (CWE-73).

    A narrow reservation rather than a general "don't overwrite what you did not
    create": re-writing your *own* output under a stable name is normal and must
    keep working, and the local backend has no notion of who wrote what. What is
    never legitimate is a caller naming an output after another run's internals.

    Scope, stated because it is narrow on purpose: run *manifests* only. Act
    checkpoints are protected by their signature rather than by their name — see
    the comment on the pattern for why a name rule for them cannot be written
    without rejecting ordinary date-stamped recordings. For a shared deployment
    the load-bearing controls are HTTP authentication and `SANZARU_RUN_SECRET`;
    this is the cheap check that catches the obvious collision early.

    Returns the name unchanged so call sites can validate inline.
    """
    if any(pattern.fullmatch(filename) for pattern in _RESERVED_NAME_PATTERNS):
        raise ValueError(
            f"{param} {filename!r:.80} is reserved for simulated-podcast run bookkeeping "
            "(a run manifest or act checkpoint) — choose another name"
        )
    return filename


def suffix_for_variant(variant: Literal["video", "thumbnail", "spritesheet"]) -> str:
    """Get the file extension for a video asset variant.

    Args:
        variant: Asset type

    Returns:
        File extension without dot (e.g., "mp4", "webp", "jpg")
    """
    return {"video": "mp4", "thumbnail": "webp", "spritesheet": "jpg"}[variant]


def generate_filename(base_id: str, suffix: str, *, use_timestamp: bool = False) -> str:
    """Generate a filename with optional timestamp.

    Args:
        base_id: Base identifier for the file (e.g., video_id, "img")
        suffix: File extension without dot (e.g., "mp4", "png")
        use_timestamp: If True, append current Unix timestamp to base_id

    Returns:
        Generated filename (e.g., "abc123.mp4" or "img_1234567890.png")
    """
    if use_timestamp:
        timestamp = int(time.time())
        return f"{base_id}_{timestamp}.{suffix}"
    return f"{base_id}.{suffix}"
