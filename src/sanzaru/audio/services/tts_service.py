"""Text-to-speech service - orchestrates TTS operations.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import time

from openai.types.audio.speech_model import SpeechModel

from ...config import get_client, logger
from ...infrastructure import FileSystemRepository, split_text_for_tts
from .. import AudioProcessor
from ..constants import TTSVoice
from ..models import TTSResult


class TTSService:
    """Service for text-to-speech operations."""

    def __init__(self):
        """Initialize the TTS service."""
        self.file_repo = FileSystemRepository()
        self.audio_processor = AudioProcessor()

    async def create_speech(
        self,
        text_prompt: str,
        output_filename: str | None = None,
        model: SpeechModel = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate text-to-speech audio from text.

        Args:
        ----
            text_prompt: Text to convert to speech.
            output_filename: Optional name for output file.
            model: TTS model to use.
            voice: Voice to use for TTS.
            instructions: Optional instructions for speech generation.
            speed: Speech speed (0.25 to 4.0).

        Returns:
        -------
            TTSResult: Result with name of the generated audio file.

        """
        # Determine output filename
        filename = output_filename or f"speech_{int(time.time() * 1000)}.mp3"

        # Split text if it exceeds the API limit
        text_chunks = split_text_for_tts(text_prompt)

        client = get_client()

        if len(text_chunks) == 1:
            # Single chunk - process directly
            response = await client.audio.speech.create(
                input=text_chunks[0],
                model=model,
                voice=voice,
                speed=speed,
            )

            # Get audio bytes from response
            audio_bytes = response.content

            # Write audio file via storage backend
            await self.file_repo.write_audio_file(filename, audio_bytes)

        else:
            # Multiple chunks - process in parallel and concatenate
            logger.info(f"Text exceeds TTS API limit, splitting into {len(text_chunks)} chunks")

            # Generate TTS for all chunks in parallel using anyio and aioresult
            import anyio
            from aioresult import ResultCapture

            async def generate_chunk(text: str) -> bytes:
                response = await client.audio.speech.create(
                    input=text,
                    model=model,
                    voice=voice,
                    speed=speed,
                )
                return response.content

            async with anyio.create_task_group() as tg:
                captures = [ResultCapture.start_soon(tg, generate_chunk, chunk) for chunk in text_chunks]

            # Collect results
            audio_chunks = [c.result() for c in captures]

            # Concatenate audio chunks using domain logic
            combined_audio = await self.audio_processor.concatenate_audio_segments(
                audio_chunks=audio_chunks,
                format="mp3",
            )

            # Write combined audio file via storage backend
            await self.file_repo.write_audio_file(filename, combined_audio)

        return TTSResult(output_file=filename)
