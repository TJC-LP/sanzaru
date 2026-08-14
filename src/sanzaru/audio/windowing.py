# SPDX-License-Identifier: MIT
"""Overlapping windows for transcribing long audio.

Transcription of a long file fails two ways, and the dangerous one is silent:
a 13:14 mp3 came back cut off around 10.5 minutes with no error, and another run
silently dropped a 13-segment stretch from the *middle*. A 30-minute file fails
loudly instead, with a 400 `input_too_large`. "The transcript looks complete-ish"
is therefore not a signal, which is why the fix is to stop sending long files
rather than to detect a bad answer (#39).

The 90s window / 75s stride pattern here is not arbitrary — it is what consumers
of this tool converged on independently, reimplemented per run with ffmpeg. The
15s overlap exists so that a sentence straddling a cut is whole in at least one
window; `merge_window_texts` then removes the duplication.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from ..config import logger
from .verification import aligned_words

WINDOW_SECONDS = 90.0
"""Field-proven window length. Long enough to carry sentence context into the
transcription, short enough that no window approaches any upload or duration
limit."""

STRIDE_SECONDS = 75.0
"""Advance per window, giving a 15s overlap."""

CHUNK_THRESHOLD_SECONDS = 480.0
"""Transcribe in one call below this, window above it.

Eight minutes sits comfortably under the ~10.5 minute mark where silent
truncation was first observed, so the failure is prevented rather than
detected."""

MIN_OVERLAP_WORDS = 4
"""Shortest word run treated as a genuine overlap rather than coincidence.

Below this, common phrases ("and then the") would match across unrelated
passages and delete real speech. Losing a little duplication is recoverable;
deleting content is the bug this whole module exists to avoid."""


@dataclass(frozen=True, slots=True)
class Window:
    """One slice of a long file, in seconds from the start."""

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def plan_windows(
    duration_s: float,
    window_s: float = WINDOW_SECONDS,
    stride_s: float = STRIDE_SECONDS,
) -> list[Window]:
    """Overlapping windows covering `duration_s` end to end.

    The last window is pinned to the end of the file rather than left wherever
    the stride happened to land: a trailing sliver is exactly where a truncation
    hides, and an explicit EOF window is what the hand-rolled versions of this
    all ended up adding.
    """
    if duration_s <= 0:
        return []
    if duration_s <= window_s:
        return [Window(0, 0.0, duration_s)]

    windows: list[Window] = []
    start = 0.0
    while start < duration_s:
        end = min(start + window_s, duration_s)
        windows.append(Window(len(windows), start, end))
        if end >= duration_s:
            break
        start += stride_s

    last = windows[-1]
    if last.end_s < duration_s:  # pragma: no cover - the loop already pins it
        windows.append(Window(len(windows), max(0.0, duration_s - window_s), duration_s))
    return windows


def merge_window_texts(texts: list[str]) -> str:
    """Join window transcripts, removing the overlap between adjacent pairs.

    Matching is word-level on normalised tokens but slicing is done on the raw
    ones, so the joined text keeps its original casing and punctuation. Only the
    seam is examined — the tail of what we have against the head of what is
    arriving — because a repeated phrase elsewhere in the file is not an overlap.

    When no confident seam is found the two are simply concatenated. That can
    duplicate a few words at a boundary; the alternative, guessing, deletes
    speech, and this module exists because speech went missing.
    """
    merged: list[str] = []
    merged_norm: list[str] = []
    for position, text in enumerate(texts):
        raw, norm = aligned_words(text)
        if not raw:
            continue
        if not merged:
            merged, merged_norm = list(raw), list(norm)
            continue

        # Look no further back than one window's worth of overlap could reach.
        span = min(len(merged_norm), len(norm), 400)
        tail, head = merged_norm[-span:], norm[:span]
        match = difflib.SequenceMatcher(None, tail, head, autojunk=False).find_longest_match(0, span, 0, span)
        if match.size >= MIN_OVERLAP_WORDS:
            merged.extend(raw[match.b + match.size :])
            merged_norm.extend(norm[match.b + match.size :])
        else:
            logger.debug("no confident overlap between windows %d and %d - joining as-is", position - 1, position)
            merged.extend(raw)
            merged_norm.extend(norm)
    return " ".join(merged)
