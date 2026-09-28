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
import re
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
_INNER_DASH = re.compile(r"[-–—]")

_AUDIO_TAG = re.compile(r"\[[^\[\]]{1,80}\]")
"""An inline ElevenLabs direction: `[whispers]`, `[strong French accent]`."""


def strip_audio_tags(text: str) -> str:
    """The part of a script line that is meant to be *heard*.

    `[laughs] [defensive] Oh, we checked.` is performed, not read, so the tags
    are never in a transcript. Left in, they are words the check expects and
    cannot find: a stacked-tag short line (`[excited] [laughs] Yes!`) is two
    thirds missing before the audio is even looked at. Only for the script
    side — transcripts carry no tags to strip.
    """
    return _AUDIO_TAG.sub(" ", text)


_UNITS = {
    word: value
    for value, word in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    word: 10 * (i + 2)
    for i, word in enumerate(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])
}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_SYMBOL_WORDS = {"$": "dollars", "£": "pounds", "€": "euros"}
_DIGITS = re.compile(r"([$£€]?)(\d[\d,]*(?:\.\d+)?|\.\d+)(%?)")


def _parse_spoken_number(tokens: list[str], start: int) -> tuple[int, str]:
    """Read a spelled-out number at `start`: `(end, digits)`, or `(start, "")`.

    Deliberately small — cardinals, hundreds and scales, and `point` + digit
    words — because the question is only whether ASR's `1.2` is the script's
    `one point two`. Two units in a row end the number (`one two` is a
    sequence, not 3), so a phone-number cadence cannot be summed into nonsense.
    """
    total = current = 0
    i, seen, last = start, False, ""
    while i < len(tokens):
        token = tokens[i]
        if token in _UNITS and last not in ("unit", "teen"):
            value = _UNITS[token]
            if last == "tens" and value >= 10:
                break
            current += value
            last = "unit" if value < 10 else "teen"
        elif token in _TENS and last in ("", "hundred", "scale"):
            current += _TENS[token]
            last = "tens"
        elif token == "hundred" and seen and last != "hundred":
            current = max(current, 1) * 100
            last = "hundred"
        elif token in _SCALES and seen and last != "scale":
            total += max(current, 1) * _SCALES[token]
            current, last = 0, "scale"
        elif (
            token == "and"
            and last in ("hundred", "scale")
            and i + 1 < len(tokens)
            and (tokens[i + 1] in _UNITS or tokens[i + 1] in _TENS)
        ):
            i += 1
            continue
        else:
            break
        seen = True
        i += 1
    integer = str(total + current) if seen else ""

    # `point seven` with no integer part is how "0.7" is usually read aloud.
    if i < len(tokens) and tokens[i] == "point":
        decimals = ""
        j = i + 1
        while j < len(tokens) and tokens[j] in _UNITS and _UNITS[tokens[j]] < 10:
            decimals += str(_UNITS[tokens[j]])
            j += 1
        if decimals:
            return j, f"{integer or '0'}.{decimals}"
    return (i, integer) if seen else (start, "")


def _canonical_digits(token: str) -> list[str]:
    """`$2,000` -> `["2000", "dollars"]`, `.7` -> `["0.7"]`; anything else unchanged."""
    match = _DIGITS.fullmatch(token)
    if not match:
        return [token]
    symbol, number, percent = match.groups()
    number = number.replace(",", "")
    if number.startswith("."):
        number = "0" + number
    out = [number]
    if percent:
        out.append("percent")
    if symbol:
        out.append(_SYMBOL_WORDS[symbol])
    return out


def _normalise_numbers(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        end, digits = _parse_spoken_number(tokens, i)
        if end > i:
            out.append(digits)
            i = end
        else:
            out.extend(_canonical_digits(tokens[i]))
            i += 1
    return out


def _strip_edges(token: str) -> str:
    # `.5` keeps its point: it is a number, and the edge strip would make it 5.
    if len(token) > 1 and token[0] == "." and token[1].isdigit():
        return "." + token[1:].strip(_WORD_EDGES)
    return token.strip(_WORD_EDGES)


def words(text: str) -> list[str]:
    """Comparable words: lowercased, stripped of the punctuation glued to them.

    Numbers compare by value, not spelling: the script says `one point two`,
    ASR writes `1.2`, and on an eight-word tail that disagreement alone scored
    0.375 against a 0.75 floor — a verify failure, and a full re-render of the
    dialogue batch, for audio that said every word. Internal dashes split for
    the same reason (`back-to-back` against `back to back`, `twenty-one`).
    """
    tokens = [part for raw in text.lower().split() for part in map(_strip_edges, _INNER_DASH.split(raw)) if part]
    return _normalise_numbers(tokens)


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
    return similarity_tokens(words(intended), words(rendered))


def similarity_tokens(intended_words: list[str], rendered_words: list[str]) -> float:
    """`similarity()` over word lists that are already tokenised.

    For a caller that compares one needle against many windows of one
    transcript — the podcast verify pass slides a segment across the unit's
    text — tokenising both sides on every call was the larger cost, and it was
    paid once per window. This is the same score with the split hoisted out;
    `similarity()` is a thin wrapper over it so the two cannot disagree.
    """
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
