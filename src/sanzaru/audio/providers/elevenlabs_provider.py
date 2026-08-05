# SPDX-License-Identifier: MIT
"""ElevenLabs text-to-speech provider (client.text_to_speech.convert).

Every `elevenlabs` import is function-local: the SDK is an optional extra, and
this module must stay importable without it so feature detection and error
messages work on an OpenAI-only install.
"""

from typing import TYPE_CHECKING, cast, get_args

import anyio

from ...config import get_elevenlabs_client, logger
from ...exceptions import TTSAPIError
from ..constants import (
    DEFAULT_ELEVENLABS_MODEL,
    ELEVENLABS_DEFAULT_CONCURRENCY,
    ELEVENLABS_MAX_CHARS,
    ELEVENLABS_MODELS,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SPEED_RANGE,
    ElevenLabsModel,
    TTSProviderName,
)
from .base import SpeechRequest, check_voice_settings_types, env_concurrency

if TYPE_CHECKING:
    from elevenlabs.client import AsyncElevenLabs
    from elevenlabs.types.voice_settings import VoiceSettings

_MODELS: tuple[str, ...] = get_args(ElevenLabsModel)

# eleven_v3 renders expressively but ignores the speed setting entirely.
_NO_SPEED_MODELS = frozenset({"eleven_v3"})

_MAX_ATTEMPTS = 4
_BACKOFF_BASE_SECONDS = 1.0

# The SDK marks optional body params with a private `OMIT = cast(Any, ...)`
# sentinel; passing None instead would serialize an explicit null. We mirror the
# trick with a typed cast so call sites stay checkable.
_OMIT_STR = cast(str, ...)
_OMIT_SETTINGS = cast("VoiceSettings", ...)


class ElevenLabsTTSProvider:
    """Speech via ElevenLabs, requested as mp3 so the stitch path is unchanged."""

    name: TTSProviderName = "elevenlabs"

    def resolve_model(self, model: str | None) -> str:
        if model is None:
            return DEFAULT_ELEVENLABS_MODEL
        if model not in _MODELS:
            raise ValueError(
                f"model={model!r} is not an ElevenLabs model; choose one of: {', '.join(ELEVENLABS_MODELS)}"
            )
        return model

    def resolve_voice(self, voice: str | None) -> str:
        if not voice or not voice.strip():
            raise ValueError(
                "provider='elevenlabs' requires an explicit voice id "
                "(from your ElevenLabs voice library), not a named OpenAI voice"
            )
        return voice.strip()

    def max_chunk_chars(self, model: str) -> int:
        return ELEVENLABS_MAX_CHARS[cast(ElevenLabsModel, model)]

    def max_concurrency(self, model: str) -> int:
        default = ELEVENLABS_DEFAULT_CONCURRENCY[cast(ElevenLabsModel, model)]
        return env_concurrency("SANZARU_ELEVENLABS_MAX_CONCURRENCY", default)

    def validate(self, request: SpeechRequest) -> None:
        if request.model not in _MODELS:
            raise ValueError(
                f"model={request.model!r} is not an ElevenLabs model; choose one of: {', '.join(ELEVENLABS_MODELS)}"
            )
        if not request.voice.strip():
            raise ValueError("provider='elevenlabs' requires an explicit voice id")

        settings = request.voice_settings or {}
        # Types first: the range comparisons below would raise TypeError on a
        # string, which the CLI reports as internal/exit 1 rather than usage.
        check_voice_settings_types(settings)
        for key, value in (
            ("stability", settings.get("stability")),
            ("similarity_boost", settings.get("similarity_boost")),
            ("style", settings.get("style")),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"voice_settings[{key!r}] must be between 0.0 and 1.0, got {value}")

        # voice_settings["speed"] wins over the neutral request.speed, matching
        # how the ElevenLabs API layers them.
        speed = float(settings["speed"]) if "speed" in settings else request.speed
        if request.model in _NO_SPEED_MODELS:
            if speed != 1.0:
                raise ValueError(
                    f"{request.model} does not support speed adjustment (got {speed}); "
                    "use eleven_multilingual_v2 for speed control"
                )
        else:
            low, high = ELEVENLABS_SPEED_RANGE
            if not low <= speed <= high:
                # Deliberately not rescaled from OpenAI's 0.25-4.0 range: speed=2.0
                # must not silently mean two different things per provider.
                raise ValueError(
                    f"speed must be between {low} and {high} for provider='elevenlabs', got {speed} "
                    f"(OpenAI's 0.25-4.0 range does not carry over)"
                )

        if request.instructions and request.instructions.strip():
            logger.warning(
                "instructions is an OpenAI-only parameter and is ignored by provider='elevenlabs'; "
                "use inline audio tags such as [whispers] in the text with eleven_v3"
            )

    async def synthesize_chunk(self, request: SpeechRequest) -> bytes:
        client = get_elevenlabs_client()
        settings = _build_voice_settings(request)

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await self._convert(client, request, settings)
            except Exception as exc:
                status = _status_code(exc)
                if status is None or not _is_retryable(status) or attempt == _MAX_ATTEMPTS:
                    raise _as_tts_error(exc, status) from exc
                delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "ElevenLabs request failed with HTTP %s (attempt %d/%d) - retrying in %.1fs",
                    status,
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                await anyio.sleep(delay)

        # Unreachable: the final attempt either returns or raises above.
        raise TTSAPIError("ElevenLabs request failed")

    async def _convert(
        self,
        client: "AsyncElevenLabs",
        request: SpeechRequest,
        settings: "VoiceSettings | None",
    ) -> bytes:
        # `convert` is an async-generator function: call it, then iterate. It is
        # not awaitable, so there is nothing to await on the call itself.
        stream = client.text_to_speech.convert(
            voice_id=request.voice,
            text=request.text,
            model_id=request.model,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
            voice_settings=settings if settings is not None else _OMIT_SETTINGS,
            previous_text=request.previous_text if request.previous_text is not None else _OMIT_STR,
            next_text=request.next_text if request.next_text is not None else _OMIT_STR,
        )
        buffer = bytearray()
        async for part in stream:
            buffer.extend(part)
        if not buffer:
            raise TTSAPIError(f"ElevenLabs returned no audio for voice {request.voice!r}")
        return bytes(buffer)


def _build_voice_settings(request: SpeechRequest) -> "VoiceSettings | None":
    """Translate our provider-neutral request into the SDK's VoiceSettings.

    Returns None when nothing needs overriding, so the voice's own defaults apply.
    Fields are set explicitly rather than by dict-unpacking, which would widen to
    Any under mypy.
    """
    settings = request.voice_settings or {}
    speed: float | None = float(settings["speed"]) if "speed" in settings else None
    if speed is None and request.speed != 1.0 and request.model not in _NO_SPEED_MODELS:
        speed = request.speed

    stability = settings.get("stability")
    similarity_boost = settings.get("similarity_boost")
    style = settings.get("style")
    use_speaker_boost = settings.get("use_speaker_boost")

    if all(value is None for value in (stability, similarity_boost, style, use_speaker_boost, speed)):
        return None

    from elevenlabs.types.voice_settings import VoiceSettings

    return VoiceSettings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
        speed=speed,
    )


def _status_code(exc: Exception | None) -> int | None:
    """HTTP status from an ElevenLabs ApiError, if this is one."""
    if exc is None:
        return None
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(status: int) -> bool:
    # 429 is the concurrency cap; 5xx is transient upstream failure.
    return status == 429 or 500 <= status < 600


def _as_tts_error(exc: Exception, status: int | None) -> Exception:
    """Wrap SDK errors in TTSAPIError; leave our own exceptions alone."""
    if isinstance(exc, TTSAPIError | ValueError):
        return exc
    if status == 429:
        return TTSAPIError(
            "ElevenLabs rate limit / concurrency cap hit (HTTP 429). Lower "
            "SANZARU_ELEVENLABS_MAX_CONCURRENCY or upgrade your subscription tier."
        )
    if status == 401:
        return TTSAPIError("ElevenLabs rejected the API key (HTTP 401) - check ELEVENLABS_API_KEY")
    if status is not None:
        return TTSAPIError(f"ElevenLabs request failed (HTTP {status}): {exc}")
    return TTSAPIError(f"ElevenLabs request failed: {exc}")


__all__ = ["ElevenLabsTTSProvider"]
