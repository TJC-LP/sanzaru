# SPDX-License-Identifier: MIT
"""OpenAI text-to-speech provider (client.audio.speech)."""

from typing import cast, get_args

from openai._types import Omit, omit
from openai.types.audio.speech_model import SpeechModel

from ...config import get_client
from ..constants import (
    DEFAULT_OPENAI_TTS_MODEL,
    DEFAULT_OPENAI_VOICE,
    DEFAULT_TTS_MAX_LENGTH,
    OPENAI_SPEED_RANGE,
    OPENAI_TTS_MODELS,
    TTSProviderName,
    TTSVoice,
)
from .base import SpeechRequest, env_concurrency

_VOICES: tuple[str, ...] = get_args(TTSVoice)


class OpenAITTSProvider:
    """Speech via OpenAI's `client.audio.speech.create`, always as mp3."""

    name: TTSProviderName = "openai"
    # No multi-speaker endpoint; dialogue runs fall back to per-segment rendering.
    supports_dialogue = False

    def resolve_model(self, model: str | None) -> str:
        if model is None:
            return DEFAULT_OPENAI_TTS_MODEL
        # Not a hard allowlist: OpenAI ships new speech models faster than we
        # release, and an unknown name surfaces as a clean API error anyway.
        return model

    def resolve_voice(self, voice: str | None) -> str:
        return voice or DEFAULT_OPENAI_VOICE

    def max_chunk_chars(self, model: str) -> int:
        return DEFAULT_TTS_MAX_LENGTH

    def max_concurrency(self, model: str) -> int:
        # 0 == unbounded, which is the historical behavior of the podcast fan-out.
        return env_concurrency("SANZARU_OPENAI_MAX_CONCURRENCY", 0)

    def validate(self, request: SpeechRequest) -> None:
        low, high = OPENAI_SPEED_RANGE
        if not low <= request.speed <= high:
            raise ValueError(f"speed must be between {low} and {high} for provider='openai', got {request.speed}")
        if request.voice_settings:
            raise ValueError("voice_settings is an ElevenLabs-only option; provider='openai' does not accept it")
        if request.voice not in _VOICES:
            # A warning, not an error: OpenAI adds voices between our releases.
            from ...config import logger

            logger.warning(
                "voice=%r is not a known OpenAI voice (expected one of: %s)", request.voice, ", ".join(_VOICES)
            )

    async def synthesize_chunk(self, request: SpeechRequest) -> bytes:
        client = get_client()
        instructions: str | Omit = omit if request.instructions is None else request.instructions
        response = await client.audio.speech.create(
            input=request.text,
            model=cast(SpeechModel, request.model),
            voice=cast(TTSVoice, request.voice),
            speed=request.speed,
            instructions=instructions,
            response_format="mp3",
        )
        return response.content


__all__ = ["OPENAI_TTS_MODELS", "OpenAITTSProvider"]
