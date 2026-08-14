# SPDX-License-Identifier: MIT
"""Did the rendered audio contain the words it was supposed to?

One home for the primitives three callers now share:

- `audio/realtime/qc.py` verifies a simulated act against the transcript its own
  models reported speaking;
- the podcast verify pass (#35) verifies a rendered segment against the script
  it was rendered from;
- windowed transcription (#39) uses the same word-level comparison to dedup the
  overlap between adjacent windows.

They agree on what a "word" is and on how much disagreement is normal, which is
the point of keeping them together — three copies of a punctuation-stripping
rule would drift, and the threshold below is calibrated against a measurement,
not chosen.
"""

from __future__ import annotations

import difflib
from io import BytesIO

DEFAULT_TRANSCRIBE_MODEL = "gpt-transcribe"
"""OpenAI's top-tier batch transcription model. Note that `gpt-live-transcribe`
is *not* a substitute — it is realtime-streaming only and /v1/audio/transcriptions
rejects it with a 404."""

TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024
"""API upload limit. Callers report audio over this as unverified rather than
failing, or split it into windows first."""

SIMILARITY_WARN_THRESHOLD = 0.80
"""Below this, intended and rendered text have diverged enough to be worth a
human listen. Normal transcription disagreement (punctuation, filler words,
numbers as digits vs words) lands around 0.85-0.95."""

_WORD_EDGES = ".,!?;:\"'()[]{}…—–-"


def words(text: str) -> list[str]:
    """Comparable words: lowercased, stripped of the punctuation glued to them."""
    return [word for word in (raw.strip(_WORD_EDGES) for raw in text.lower().split()) if word]


def aligned_words(text: str) -> tuple[list[str], list[str]]:
    """`(raw tokens, comparable tokens)` — index-aligned, same length.

    `words()` drops tokens that normalise to nothing, which is right for
    scoring and wrong for splicing: a match found on the comparable list has to
    be able to slice the raw one at the same index. Window merging needs the
    raw tokens back so the joined transcript keeps its casing and punctuation.
    """
    raw = text.split()
    return raw, [token.strip(_WORD_EDGES).lower() for token in raw]


def similarity(intended: str, rendered: str) -> float:
    """Word-level overlap between two transcripts, 0-1.

    Word-level rather than character-level so that spelling disagreements
    between the two models don't swamp the signal we care about, which is
    *missing speech*. Punctuation is stripped for the same reason and not just
    tokenised around: the models disagree about it constantly, and a word-level
    diff scores `plumbing` against `plumbing.` as a total mismatch — enough to
    put a short, clean act under the warn threshold on commas alone.
    """
    intended_words = words(intended)
    rendered_words = words(rendered)
    if not intended_words and not rendered_words:
        return 1.0
    if not intended_words or not rendered_words:
        return 0.0
    return difflib.SequenceMatcher(None, intended_words, rendered_words, autojunk=False).ratio()


async def transcribe_bytes(audio: bytes, filename: str, model: str = DEFAULT_TRANSCRIBE_MODEL) -> str:
    """Transcribe an in-memory audio buffer.

    Deliberately bypasses the storage layer and `TranscriptionService`: every
    caller already holds bytes it never wrote to disk — a rendered act, a
    segment before stitching, one window of a long file. That also means it
    inherits none of their size or duration guards, so callers check
    `TRANSCRIBE_MAX_BYTES` themselves.
    """
    from ..config import get_client

    client = get_client()
    result = await client.audio.transcriptions.create(
        file=(filename, BytesIO(audio)),
        model=model,  # type: ignore[arg-type]  # accepts any model id; AudioModel literal lags releases
        response_format="text",
    )
    return result if isinstance(result, str) else getattr(result, "text", "")
