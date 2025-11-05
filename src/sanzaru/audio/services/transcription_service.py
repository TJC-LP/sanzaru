"""Transcription service - orchestrates transcription operations.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import base64
from io import BytesIO
from typing import Any, Literal

from openai._types import omit
from openai.types import AudioModel, AudioResponseFormat

from ...config import get_client, get_path
from ...infrastructure import FileSystemRepository, SecurePathResolver
from ..constants import ENHANCEMENT_PROMPTS, AudioChatModel, EnhancementType
from ..models import ChatResult, TranscriptionResult


class TranscriptionService:
    """Service for audio transcription operations."""

    def __init__(self):
        """Initialize the transcription service."""
        audio_path = get_path("audio")
        self.file_repo = FileSystemRepository(audio_path)
        self.path_resolver = SecurePathResolver(audio_path)

    async def transcribe_audio(
        self,
        filename: str,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using OpenAI's transcription API.

        Args:
        ----
            filename: Name of the audio file.
            model: Transcription model to use.
            response_format: Format of the response.
            prompt: Optional prompt to guide transcription.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            TranscriptionResult: Transcription result with typed fields.

        """
        client = get_client()

        # Resolve filename to path
        file_path = self.path_resolver.resolve_input(filename)

        # Read audio file
        audio_bytes = await self.file_repo.read_audio_file(file_path)

        # Transcribe using OpenAI
        transcription = await client.audio.transcriptions.create(
            file=(filename, BytesIO(audio_bytes)),
            model=model,
            response_format=response_format,  # type: ignore[arg-type]  # SDK stubs incomplete for AudioResponseFormat
            prompt=prompt if prompt is not None else omit,
            timestamp_granularities=timestamp_granularities if timestamp_granularities is not None else omit,
        )

        # Convert to TranscriptionResult
        if isinstance(transcription, str):
            return TranscriptionResult(text=transcription)
        else:
            return TranscriptionResult(**transcription.model_dump())

    async def chat_with_audio(
        self,
        filename: str,
        model: AudioChatModel = "gpt-4o-audio-preview-2025-06-03",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> ChatResult:
        """Chat with audio using GPT-4o audio models.

        Args:
        ----
            filename: Name of the audio file.
            model: Audio chat model to use.
            system_prompt: Optional system prompt.
            user_prompt: Optional user text prompt.

        Returns:
        -------
            ChatResult: Chat response with typed text field.

        """
        client = get_client()

        # Resolve filename to path
        file_path = self.path_resolver.resolve_input(filename)

        # Validate format
        ext = file_path.suffix.lower().replace(".", "")
        if ext not in ["mp3", "wav"]:
            raise ValueError(f"Expected mp3 or wav extension, but got {ext}")

        # Read audio file
        audio_bytes = await self.file_repo.read_audio_file(file_path)

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Build messages
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add audio input with optional text prompt
        user_content: list[dict[str, Any]] = [
            {"type": "input_audio", "input_audio": {"data": audio_base64, "format": ext}}
        ]
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})

        messages.append({"role": "user", "content": user_content})

        # Chat with audio using OpenAI
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
        )

        # Extract text from response
        text = response.choices[0].message.content or ""

        return ChatResult(text=text)

    async def transcribe_enhanced(
        self,
        filename: str,
        enhancement_type: EnhancementType,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio with enhancement prompts for different styles.

        Args:
        ----
            filename: Name of the audio file.
            enhancement_type: Type of enhancement (detailed, storytelling, professional, analytical).
            model: Transcription model to use.
            response_format: Format of the response.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            TranscriptionResult: Enhanced transcription result.

        """
        # Use enhancement prompts from constants
        prompt = ENHANCEMENT_PROMPTS.get(enhancement_type, ENHANCEMENT_PROMPTS["detailed"])

        return await self.transcribe_audio(
            filename=filename,
            model=model,
            response_format=response_format,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )
