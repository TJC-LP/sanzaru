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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, TypedDict, runtime_checkable

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


def check_voice_settings_types(settings: VoiceSettingsDict, prefix: str = "") -> None:
    """Reject unknown keys and wrong value types in a voice_settings mapping.

    Shared by the podcast script validator and the ElevenLabs provider so a bad
    type is the same ValueError on every path — the alternative was a TypeError
    from the first range comparison, which the CLI reports as internal/exit 1
    instead of usage/exit 2.

    `bool` is excluded from the numeric keys explicitly: `isinstance(True, int)`
    is True, so `{"stability": true}` would otherwise pass both the type check
    and the 0.0-1.0 range check and reach the API.

    Args:
        settings: The mapping to check. Untrusted — it arrives from JSON.
        prefix: Optional location prefix for messages, e.g. "Speaker 0 ".
    """
    if not isinstance(settings, dict):
        raise ValueError(f"{prefix}voice_settings must be an object")
    for key in settings:
        if key not in VOICE_SETTINGS_KEYS:
            raise ValueError(
                f"{prefix}voice_settings has unknown key '{key}'; expected: {', '.join(VOICE_SETTINGS_KEYS)}"
            )
    for float_key in VOICE_SETTINGS_FLOAT_KEYS:
        if float_key in settings:
            value = settings[float_key]  # type: ignore[literal-required]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{prefix}voice_settings['{float_key}'] must be a number")
    for bool_key in VOICE_SETTINGS_BOOL_KEYS:
        if bool_key in settings and not isinstance(settings[bool_key], bool):  # type: ignore[literal-required]
            raise ValueError(f"{prefix}voice_settings['{bool_key}'] must be a boolean")


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


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """One speaker's turn within a multi-speaker dialogue request."""

    text: str
    voice: str


@dataclass(frozen=True, slots=True)
class SpeechUsage:
    """What a synthesis actually submitted (#52).

    Counted here rather than reported by the provider: the text is already in
    hand at this layer, and ElevenLabs bills *characters submitted* — which is
    exactly what we send, inline audio tags included. Nothing has to come back
    from the API for this to be right, so the number is available even on a
    request that later fails.

    OpenAI is not quota-metered this way, so its count is informational. On
    ElevenLabs, where a firm-wide free tier is 10,000 characters a month, it is
    the number a caller most needs — and had to count by hand before.
    """

    provider: TTSProviderName
    model: str
    characters: int = 0
    requests: int = 0

    def __add__(self, other: "SpeechUsage") -> "SpeechUsage":
        if (self.provider, self.model) != (other.provider, other.model):
            raise ValueError(f"cannot add usage across {self.provider}/{self.model} and {other.provider}/{other.model}")
        return SpeechUsage(
            provider=self.provider,
            model=self.model,
            characters=self.characters + other.characters,
            requests=self.requests + other.requests,
        )


@dataclass(frozen=True, slots=True)
class SpeechResult:
    """Audio plus what producing it cost."""

    audio: bytes
    usage: SpeechUsage


class TTSProvider(Protocol):
    """A text-to-speech backend. Implementations must return mp3 bytes."""

    name: TTSProviderName
    supports_dialogue: bool
    """True when the provider can render several turns in one request, letting
    the model pace the conversation instead of us inserting silence gaps."""

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


@runtime_checkable
class DialogueProvider(Protocol):
    """A provider that can render a multi-speaker exchange in one request.

    Kept separate from TTSProvider so backends without the capability (OpenAI)
    are not forced to stub it out. Narrow with `as_dialogue_provider`.
    """

    name: TTSProviderName
    supports_dialogue: bool

    def supports_dialogue_model(self, model: str) -> bool:
        """True when `model` can render dialogue (not every model can)."""
        ...

    def max_dialogue_chars(self, model: str) -> int:
        """Total characters accepted across all turns in one dialogue request."""
        ...

    async def synthesize_dialogue(
        self,
        turns: Sequence[DialogueTurn],
        model: str,
        stability: float | None = None,
    ) -> bytes:
        """Render `turns` as one continuous conversation, returning mp3 bytes."""
        ...


def as_dialogue_provider(provider: TTSProvider) -> "DialogueProvider | None":
    """Return `provider` narrowed to DialogueProvider, or None if unsupported."""
    if getattr(provider, "supports_dialogue", False) and isinstance(provider, DialogueProvider):
        return provider
    return None


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
) -> SpeechResult:
    """Synthesize `request` into mp3 bytes, splitting text the provider can't take at once.

    Args:
        provider: Backend to synthesize with.
        request: The (already provider-resolved) speech request.
        limiter: Optional concurrency limiter shared with the caller's own fan-out,
            so segment-level and chunk-level parallelism draw on one budget. Must be
            created inside the running event loop — CapacityLimiter is loop-bound.

    Returns:
        mp3 bytes and the characters they cost. Characters are summed over the
        *chunks actually sent*, not over `request.text`: splitting drops the
        whitespace at each boundary, so the two differ by a few characters on a
        long segment and the submitted count is the billable one.
    """
    provider.validate(request)

    # Local import: pydub is an optional extra, and the single-chunk path (the
    # common case) must not require it.
    from ...infrastructure import split_text_for_tts

    chunks = split_text_for_tts(request.text, provider.max_chunk_chars(request.model))
    usage = SpeechUsage(
        provider=provider.name,
        model=request.model,
        characters=sum(len(chunk) for chunk in chunks),
        requests=len(chunks),
    )

    if len(chunks) == 1:
        audio = await _synthesize_one(provider, replace(request, text=chunks[0]), limiter)
        return SpeechResult(audio=audio, usage=usage)

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

    joined = await AudioProcessor().concatenate_audio_segments(audio_chunks, format="mp3")
    return SpeechResult(audio=joined, usage=usage)


async def _synthesize_one(
    provider: TTSProvider,
    request: SpeechRequest,
    limiter: anyio.CapacityLimiter | None,
) -> bytes:
    if limiter is None:
        return await provider.synthesize_chunk(request)
    async with limiter:
        return await provider.synthesize_chunk(request)
