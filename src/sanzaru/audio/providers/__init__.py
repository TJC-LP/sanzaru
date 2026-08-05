# SPDX-License-Identifier: MIT
"""TTS provider registry.

Providers are cheap, stateless objects — build one per call rather than caching,
so nothing outlives an event loop. Imports are lazy so an OpenAI-only install
never imports the optional `elevenlabs` SDK.
"""

from typing import get_args

from ..constants import DEFAULT_TTS_PROVIDER, TTSProviderName
from .base import (
    VOICE_SETTINGS_BOOL_KEYS,
    VOICE_SETTINGS_FLOAT_KEYS,
    VOICE_SETTINGS_KEYS,
    DialogueProvider,
    DialogueTurn,
    SpeechRequest,
    TTSProvider,
    VoiceSettingsDict,
    as_dialogue_provider,
    check_voice_settings_types,
    synthesize_speech,
)

PROVIDER_NAMES: tuple[str, ...] = get_args(TTSProviderName)


def get_provider(name: str | None) -> TTSProvider:
    """Build the provider for `name` (None → the default provider).

    Raises:
        ValueError: If `name` is not a known provider.
        ImportError: If the provider's optional extra is not installed.
    """
    resolved = name or DEFAULT_TTS_PROVIDER
    if resolved == "openai":
        from .openai_provider import OpenAITTSProvider

        return OpenAITTSProvider()
    if resolved == "elevenlabs":
        from .elevenlabs_provider import ElevenLabsTTSProvider

        return ElevenLabsTTSProvider()
    raise ValueError(f"unknown TTS provider {resolved!r}; expected one of: {', '.join(PROVIDER_NAMES)}")


def validate_provider_name(name: str, context: str) -> TTSProviderName:
    """Narrow a free-form string to a provider name, or raise ValueError naming `context`."""
    if name not in PROVIDER_NAMES:
        raise ValueError(f"{context}: unknown provider {name!r}; expected one of: {', '.join(PROVIDER_NAMES)}")
    return name  # type: ignore[return-value]


__all__ = [
    "PROVIDER_NAMES",
    "VOICE_SETTINGS_BOOL_KEYS",
    "VOICE_SETTINGS_FLOAT_KEYS",
    "VOICE_SETTINGS_KEYS",
    "DialogueProvider",
    "DialogueTurn",
    "SpeechRequest",
    "TTSProvider",
    "VoiceSettingsDict",
    "as_dialogue_provider",
    "check_voice_settings_types",
    "get_provider",
    "synthesize_speech",
    "validate_provider_name",
]
