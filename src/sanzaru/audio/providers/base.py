# SPDX-License-Identifier: MIT
"""Provider-neutral TTS request/response contract and the chunking orchestrator.

Division of labour: a provider synthesizes **one chunk** of text and returns mp3
bytes. Everything above that — splitting long text, bounded parallel fan-out,
concatenation — lives in `synthesize_speech` here, so both TTS call sites
(`TTSService.create_speech` and `generate_podcast`) share one implementation.

mp3 is the contract, not an implementation detail: the podcast stitcher decodes
every segment with `AudioSegment.from_mp3`.
"""

import os
from dataclasses import dataclass, replace
from typing import Protocol, TypedDict

import anyio

from ...config import logger
from ..constants import TTSProviderName


class VoiceSettingsDict(TypedDict, total=False):
    """ElevenLabs voice tuning. Ignored by the OpenAI provider.

    `speed` here takes precedence over `SpeechRequest.speed` when both are given.
    """

    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool
    speed: float


VOICE_SETTINGS_FLOAT_KEYS = ("stability", "similarity_boost", "style", "speed")
VOICE_SETTINGS_BOOL_KEYS = ("use_speaker_boost",)
VOICE_SETTINGS_KEYS = VOICE_SETTINGS_FLOAT_KEYS + VOICE_SETTINGS_BOOL_KEYS


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    """One text-to-speech request, before provider-specific translation."""

    text: str
    voice: str
    model: str
    speed: float = 1.0
    instructions: str | None = None
    """Style/tone direction. OpenAI-only — ElevenLabs has no equivalent parameter
    (use inline audio tags like `[whispers]` in the text with eleven_v3)."""
    voice_settings: VoiceSettingsDict | None = None
    """ElevenLabs-only voice tuning."""
    previous_text: str | None = None
    """Preceding text, for cross-chunk prosody continuity. ElevenLabs-only."""
    next_text: str | None = None
    """Following text, for cross-chunk prosody continuity. ElevenLabs-only."""


class TTSProvider(Protocol):
    """A text-to-speech backend. Implementations must return mp3 bytes."""

    name: TTSProviderName

    def resolve_model(self, model: str | None) -> str:
        """Return the model to use, or raise ValueError if it belongs to another provider."""
        ...

    def resolve_voice(self, voice: str | None) -> str:
        """Return the voice to use, or raise ValueError when one is required and absent."""
        ...

    def max_chunk_chars(self, model: str) -> int:
        """Longest text this provider accepts in one request for `model`."""
        ...

    def max_concurrency(self, model: str) -> int:
        """Concurrent requests allowed; 0 means unbounded."""
        ...

    def validate(self, request: SpeechRequest) -> None:
        """Raise ValueError if the request is not satisfiable by this provider."""
        ...

    async def synthesize_chunk(self, request: SpeechRequest) -> bytes:
        """Synthesize one chunk of text into mp3 bytes."""
        ...


def env_concurrency(env_var: str, default: int) -> int:
    """Read a concurrency override from the environment, falling back to `default`.

    A malformed or negative value is ignored with a warning rather than failing a
    render that would otherwise have worked.
    """
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer - using default %d", env_var, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%d is negative - using default %d", env_var, value, default)
        return default
    return value


async def synthesize_speech(
    provider: TTSProvider,
    request: SpeechRequest,
    *,
    limiter: anyio.CapacityLimiter | None = None,
) -> bytes:
    """Synthesize `request` into mp3 bytes, splitting text the provider can't take at once.

    Args:
        provider: Backend to synthesize with.
        request: The (already provider-resolved) speech request.
        limiter: Optional concurrency limiter shared with the caller's own fan-out,
            so segment-level and chunk-level parallelism draw on one budget. Must be
            created inside the running event loop — CapacityLimiter is loop-bound.

    Returns:
        mp3 bytes.
    """
    provider.validate(request)

    # Local import: pydub is an optional extra, and the single-chunk path (the
    # common case) must not require it.
    from ...infrastructure import split_text_for_tts

    chunks = split_text_for_tts(request.text, provider.max_chunk_chars(request.model))

    if len(chunks) == 1:
        return await _synthesize_one(provider, replace(request, text=chunks[0]), limiter)

    logger.debug("Text split into %d %s TTS chunks", len(chunks), provider.name)

    from aioresult import ResultCapture  # type: ignore[import-untyped]

    async def _chunk(index: int) -> bytes:
        # Adjacent text gives ElevenLabs cross-chunk prosody continuity; the
        # OpenAI provider ignores these fields.
        chunk_request = replace(
            request,
            text=chunks[index],
            previous_text=chunks[index - 1] if index > 0 else None,
            next_text=chunks[index + 1] if index + 1 < len(chunks) else None,
        )
        return await _synthesize_one(provider, chunk_request, limiter)

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _chunk, i) for i in range(len(chunks))]

    # Read by index, not completion order — chunk order is the audio order.
    audio_chunks = [capture.result() for capture in captures]

    from ..processor import AudioProcessor

    return await AudioProcessor().concatenate_audio_segments(audio_chunks, format="mp3")


async def _synthesize_one(
    provider: TTSProvider,
    request: SpeechRequest,
    limiter: anyio.CapacityLimiter | None,
) -> bytes:
    if limiter is None:
        return await provider.synthesize_chunk(request)
    async with limiter:
        return await provider.synthesize_chunk(request)
