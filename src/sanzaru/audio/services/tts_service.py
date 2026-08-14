"""Text-to-speech service - orchestrates TTS operations.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import time

import anyio

from ...infrastructure import FileSystemRepository
from ..models import TTSResult
from ..providers import SpeechRequest, VoiceSettingsDict, get_provider, synthesize_speech


class TTSService:
    """Service for text-to-speech operations."""

    def __init__(self):
        """Initialize the TTS service."""
        self.file_repo = FileSystemRepository()

    async def create_speech(
        self,
        text_prompt: str,
        output_filename: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        speed: float = 1.0,
        provider: str | None = None,
        voice_settings: VoiceSettingsDict | None = None,
    ) -> TTSResult:
        """Generate text-to-speech audio from text.

        Args:
        ----
            text_prompt: Text to convert to speech.
            output_filename: Optional name for output file.
            model: TTS model. Defaults to the selected provider's default.
            voice: Voice to use. OpenAI takes a named voice; ElevenLabs takes a voice id.
            instructions: Optional style direction. OpenAI-only — ElevenLabs ignores it.
            speed: Speech speed (0.25-4.0 on OpenAI, 0.7-1.2 on ElevenLabs).
            provider: "openai" (default) or "elevenlabs".
            voice_settings: ElevenLabs voice tuning. Rejected by the OpenAI provider.

        Returns:
        -------
            TTSResult: Result with name of the generated audio file.

        """
        tts = get_provider(provider)
        request = SpeechRequest(
            text=text_prompt,
            voice=tts.resolve_voice(voice),
            model=tts.resolve_model(model),
            speed=speed,
            instructions=instructions,
            voice_settings=voice_settings,
        )

        limit = tts.max_concurrency(request.model)
        limiter = anyio.CapacityLimiter(limit) if limit else None
        rendered = await synthesize_speech(tts, request, limiter=limiter)

        filename = output_filename or f"speech_{int(time.time() * 1000)}.mp3"
        await self.file_repo.write_audio_file(filename, rendered.audio)

        return TTSResult(
            output_file=filename,
            provider=rendered.usage.provider,
            model=rendered.usage.model,
            characters=rendered.usage.characters,
            requests=rendered.usage.requests,
        )
