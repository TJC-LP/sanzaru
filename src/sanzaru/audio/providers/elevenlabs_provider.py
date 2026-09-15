# SPDX-License-Identifier: MIT
"""ElevenLabs text-to-speech provider (client.text_to_speech.convert).

Every `elevenlabs` import is function-local: the SDK is an optional extra, and
this module must stay importable without it so feature detection and error
messages work on an OpenAI-only install.
"""

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, get_args

import anyio

from ...config import get_elevenlabs_client, logger
from ...exceptions import TTSAPIError
from ..constants import (
    DEFAULT_ELEVENLABS_MODEL,
    ELEVENLABS_DEFAULT_CONCURRENCY,
    ELEVENLABS_DIALOGUE_MAX_CHARS,
    ELEVENLABS_DIALOGUE_MODELS,
    ELEVENLABS_MAX_CHARS,
    ELEVENLABS_MODELS,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SPEED_RANGE,
    ElevenLabsModel,
    TTSProviderName,
)
from .base import DialogueTurn, SpeechRequest, check_voice_settings_types, env_concurrency

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from elevenlabs.client import AsyncElevenLabs
    from elevenlabs.types.model_settings_response_model import ModelSettingsResponseModel
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
_OMIT_MODEL_SETTINGS = cast("ModelSettingsResponseModel", ...)

# The voice id is a URL *path segment*: the Fern-generated SDK interpolates it
# into f"v1/text-to-speech/{voice_id}" and string-concatenates that onto the
# base URL unencoded (its `_build_url` explicitly avoids urljoin), and httpx
# then collapses the dot segments when it builds the request. A voice of
# "../../v1/voices/VICTIM/settings/edit?" therefore retargets an authenticated
# POST carrying the operator's xi-api-key, and the non-2xx body comes back to
# the caller through _as_tts_error. Real ids are 20-character alphanumeric
# tokens (21m00Tcm4TlvDq8ikWAM); "_" and "-" are allowed because callers name
# voices that way in scripts and fixtures, and "." is out so ".." cannot be
# spelled at all. 64 leaves room for a longer id without leaving room for a URL.
_VOICE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _validate_voice_id(voice: str) -> str:
    """Return `voice` if it is a plain ElevenLabs voice id, else raise ValueError."""
    if not isinstance(voice, str) or not _VOICE_ID_PATTERN.fullmatch(voice):
        # Truncated repr: the value is caller-controlled and lands in logs.
        raise ValueError(
            f"voice={voice!r:.80} is not a valid ElevenLabs voice id; expected only letters, digits, "
            "'_' or '-' (max 64 characters)"
        )
    return voice


class ElevenLabsTTSProvider:
    """Speech via ElevenLabs, requested as mp3 so the stitch path is unchanged."""

    name: TTSProviderName = "elevenlabs"
    supports_dialogue = True

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
        # Every caller path — TTSService.create_speech, the podcast segment and
        # dialogue planners — resolves the voice here, so this is the one place
        # the path-segment check has to live.
        return _validate_voice_id(voice.strip())

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
        # Backstop for a SpeechRequest assembled without resolve_voice: validate()
        # is the last thing synthesize_speech runs before the URL is built.
        # resolve_voice strips, so whitespace reaching here means the request
        # skipped it; say so rather than reporting a valid id as a bad one.
        if request.voice != request.voice.strip():
            raise ValueError(
                f"voice={request.voice!r:.80} has surrounding whitespace; pass the bare voice id "
                "(resolve_voice strips it)"
            )
        _validate_voice_id(request.voice)

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
        return await self._with_retries(lambda: self._convert(client, request, settings))

    # ---------- dialogue ----------

    def supports_dialogue_model(self, model: str) -> bool:
        return model in ELEVENLABS_DIALOGUE_MODELS

    def max_dialogue_chars(self, model: str) -> int:
        return ELEVENLABS_DIALOGUE_MAX_CHARS

    async def synthesize_dialogue(
        self,
        turns: Sequence[DialogueTurn],
        model: str,
        stability: float | None = None,
    ) -> bytes:
        """Render several turns as one conversation via /v1/text-to-dialogue.

        The model sees every turn at once, so it paces the exchange itself —
        turn-taking gaps sized to the content, reactions that land on the
        previous line. That is the whole point, and the reason this cannot be
        expressed as N independent synthesize_chunk calls.
        """
        if not turns:
            raise ValueError("synthesize_dialogue requires at least one turn")
        if not self.supports_dialogue_model(model):
            raise ValueError(
                f"model={model!r} does not support dialogue rendering; "
                f"use one of: {', '.join(sorted(ELEVENLABS_DIALOGUE_MODELS))}"
            )
        if stability is not None and not 0.0 <= stability <= 1.0:
            raise ValueError(f"dialogue stability must be between 0.0 and 1.0, got {stability}")

        # Over-budget requests fail by *terminating the stream early*, which
        # _convert_dialogue cannot distinguish from a complete take — it would
        # return a truncated conversation as a success. The planner already
        # splits runs at this budget; this is the backstop for direct callers.
        total_chars = sum(len(turn.text) for turn in turns)
        if total_chars > ELEVENLABS_DIALOGUE_MAX_CHARS:
            raise ValueError(
                f"dialogue request is {total_chars} characters across {len(turns)} turns, over the "
                f"{ELEVENLABS_DIALOGUE_MAX_CHARS}-character limit; split it at a turn boundary"
            )

        # Here the voice rides in the request body rather than the path, so this
        # is consistency rather than a fix: one malformed id fails the same way
        # whichever endpoint would have received it.
        for turn in turns:
            _validate_voice_id(turn.voice)

        client = get_elevenlabs_client()
        return await self._with_retries(lambda: self._convert_dialogue(client, turns, model, stability))

    async def _convert_dialogue(
        self,
        client: "AsyncElevenLabs",
        turns: Sequence[DialogueTurn],
        model: str,
        stability: float | None,
    ) -> bytes:
        from elevenlabs.types.dialogue_input import DialogueInput

        settings = _OMIT_MODEL_SETTINGS
        if stability is not None:
            from elevenlabs.types.model_settings_response_model import ModelSettingsResponseModel

            settings = ModelSettingsResponseModel(stability=stability)

        stream = client.text_to_dialogue.convert(
            inputs=[DialogueInput(text=turn.text, voice_id=turn.voice) for turn in turns],
            model_id=model,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
            settings=settings,
        )
        buffer = bytearray()
        async for part in stream:
            buffer.extend(part)
        if not buffer:
            raise TTSAPIError(f"ElevenLabs returned no audio for a {len(turns)}-turn dialogue")
        return bytes(buffer)

    # ---------- shared ----------

    async def _with_retries(self, call: "Callable[[], Awaitable[bytes]]") -> bytes:
        """Run `call`, retrying 429s and 5xx with exponential backoff."""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await call()
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
