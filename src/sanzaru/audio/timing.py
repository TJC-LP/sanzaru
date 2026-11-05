# SPDX-License-Identifier: MIT
"""Word timing for rendered audio: ElevenLabs forced alignment and Scribe transcription.

OpenAI's current transcription models return no timestamps; only the deprecated
`whisper-1` does. Timing here comes from ElevenLabs instead:

- Forced alignment takes audio plus the text it speaks and returns a start and
  end time for every character and word. The response has one character entry
  per character of the submitted text, spaces included, which is what lets
  `align_turns` map times back onto script turns by character offset.
- Scribe (`scribe_v2`) transcribes audio whose text is not known, with word
  start and end times.

Every `elevenlabs` import is function-local, as in the TTS provider: the SDK is
an optional extra.

see: https://elevenlabs.io/docs/api-reference/forced-alignment/create
see: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
see: https://developers.openai.com/api/docs/guides/speech-to-text
"""

import io
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import anyio
from pydantic import BaseModel, Field

from ..config import get_elevenlabs_client, logger
from ..exceptions import TTSAPIError
from .verification import strip_audio_tags

SCRIBE_MODELS = ("scribe_v2",)
"""ElevenLabs transcription models `transcribe_scribe` accepts."""

_MAX_ATTEMPTS = 4
_BACKOFF_BASE_SECONDS = 1.0

T = TypeVar("T")


class TimedWord(BaseModel):
    """One spoken word and when it is heard, in seconds from the start of the file."""

    text: str
    start: float
    end: float


class TimedTurn(BaseModel):
    """One script turn's span in the rendered file."""

    index: int
    speaker: str | None = None
    text: str
    start: float
    end: float
    words: list[TimedWord]


class Timing(BaseModel):
    """Word timing for one audio file."""

    source: str = Field(description="forced_alignment or the transcription model that produced it")
    text: str = Field(description="The text the times belong to; tags removed for forced alignment")
    words: list[TimedWord]
    turns: list[TimedTurn] | None = None
    loss: float | None = Field(
        default=None,
        description="Forced-alignment loss averaged over every character; lower is a closer fit",
    )


def spoken_text(text: str) -> str:
    """`text` with inline audio tags removed and whitespace collapsed."""
    return " ".join(strip_audio_tags(text).split())


async def align(audio: bytes, filename: str, text: str) -> Timing:
    """Align `audio` to the words it speaks.

    Args:
        audio: The rendered file's bytes, any format ElevenLabs accepts.
        filename: Name sent with the upload; its extension tells the API the format.
        text: What the audio says. Inline audio tags are removed first.
    """
    timing, _ = await align_turns(audio, filename, [text])
    return timing.model_copy(update={"turns": None})


async def align_turns(
    audio: bytes,
    filename: str,
    turns: Sequence[str],
    speakers: Sequence[str | None] | None = None,
) -> tuple[Timing, list[TimedTurn]]:
    """Align `audio` to a script of turns spoken in order.

    The turns go to one forced-alignment request, joined by single spaces. Each
    turn's span is read from the character times at its offsets in that joined
    text, so the mapping does not depend on how the API splits words.

    Returns:
        The whole file's timing (with `turns` set) and the turns themselves.
    """
    cleaned = [spoken_text(turn) for turn in turns]
    if not any(cleaned):
        raise ValueError("nothing to align: the text is empty once audio tags are removed")
    if speakers is not None and len(speakers) != len(turns):
        raise ValueError(f"{len(speakers)} speakers for {len(turns)} turns")

    # A turn that is only tags has nothing to align and stays out of the
    # joined text, so it never adds a doubled, leading or trailing space.
    spoken = [(index, text) for index, text in enumerate(cleaned) if text]
    offsets: list[tuple[int, int]] = []
    position = 0
    for _, text in spoken:
        offsets.append((position, position + len(text)))
        position += len(text) + 1
    joined = " ".join(text for _, text in spoken)

    client = get_elevenlabs_client()
    response = await _with_retries(
        lambda: client.forced_alignment.create(file=(filename, io.BytesIO(audio)), text=joined)
    )
    if len(response.characters) != len(joined):
        raise TTSAPIError(f"forced alignment returned {len(response.characters)} characters for {len(joined)} of text")

    words = [TimedWord(text=word.text, start=word.start, end=word.end) for word in response.words if word.text.strip()]
    starts = [character.start for character in response.characters]
    ends = [character.end for character in response.characters]

    timed_turns: list[TimedTurn] = []
    for (index, text), (first, last) in zip(spoken, offsets, strict=True):
        start, end = starts[first], ends[last - 1]
        timed_turns.append(
            TimedTurn(
                index=index,
                speaker=speakers[index] if speakers is not None else None,
                text=text,
                start=start,
                end=end,
                words=[word for word in words if start <= word.start < end],
            )
        )

    timing = Timing(source="forced_alignment", text=joined, words=words, turns=timed_turns, loss=response.loss)
    return timing, timed_turns


async def transcribe_scribe(audio: bytes, filename: str, model: str = "scribe_v2") -> Timing:
    """Transcribe `audio` with ElevenLabs Scribe, with word start and end times."""
    if model not in SCRIBE_MODELS:
        raise ValueError(f"model={model!r} is not a Scribe model; choose one of: {', '.join(SCRIBE_MODELS)}")
    client = get_elevenlabs_client()
    response = await _with_retries(
        lambda: client.speech_to_text.convert(
            model_id=model,
            file=(filename, io.BytesIO(audio)),
            timestamps_granularity="word",
        )
    )
    text = getattr(response, "text", None)
    raw_words = getattr(response, "words", None)
    if not isinstance(text, str) or raw_words is None:
        raise TTSAPIError("ElevenLabs Scribe returned no transcript (multichannel or webhook response)")
    words = [
        TimedWord(text=word.text, start=word.start or 0.0, end=word.end or 0.0)
        for word in raw_words
        if word.type == "word"
    ]
    return Timing(source=model, text=text, words=words)


async def _with_retries(call: Callable[[], Awaitable[T]]) -> T:
    """Run `call`, retrying HTTP 429 and 5xx with exponential backoff."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await call()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = isinstance(status, int) and (status == 429 or 500 <= status < 600)
            if not retryable or attempt == _MAX_ATTEMPTS:
                if isinstance(exc, TTSAPIError | ValueError):
                    raise
                detail = f"HTTP {status}" if isinstance(status, int) else type(exc).__name__
                raise TTSAPIError(f"ElevenLabs timing request failed ({detail}): {exc}") from exc
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "ElevenLabs timing request failed with HTTP %s (attempt %d/%d) - retrying in %.1fs",
                status,
                attempt,
                _MAX_ATTEMPTS,
                delay,
            )
            await anyio.sleep(delay)
    raise TTSAPIError("ElevenLabs timing request failed")


__all__ = [
    "SCRIBE_MODELS",
    "TimedTurn",
    "TimedWord",
    "Timing",
    "align",
    "align_turns",
    "spoken_text",
    "transcribe_scribe",
]
