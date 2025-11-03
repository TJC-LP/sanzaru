# SPDX-License-Identifier: MIT
"""Audio tools for sanzaru - Whisper transcription, GPT-4o audio chat, and TTS.

These tools provide MCP interfaces to audio processing capabilities including:
- File management (listing, filtering, sorting)
- Audio processing (conversion, compression)
- Transcription (Whisper, GPT-4o)
- Text-to-speech generation

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from typing import Literal

from openai.types import AudioModel, AudioResponseFormat
from openai.types.audio.speech_model import SpeechModel

from ..audio.constants import AudioChatModel, EnhancementType, SortBy, SupportedChatWithAudioFormat, TTSVoice
from ..audio.models import AudioProcessingResult, ChatResult, FilePathSupportParams, TranscriptionResult, TTSResult
from ..audio.services import AudioService, FileService, TranscriptionService, TTSService

# ==================== FILE MANAGEMENT TOOLS ====================


async def list_audio_files(
    pattern: str | None = None,
    min_size_bytes: int | None = None,
    max_size_bytes: int | None = None,
    min_duration_seconds: float | None = None,
    max_duration_seconds: float | None = None,
    min_modified_time: float | None = None,
    max_modified_time: float | None = None,
    format: str | None = None,
    sort_by: SortBy = SortBy.NAME,
    reverse: bool = False,
) -> list[FilePathSupportParams]:
    """List, filter, and sort audio files with comprehensive options.

    Args:
        pattern: Optional regex pattern to filter files by name
        min_size_bytes: Minimum file size in bytes
        max_size_bytes: Maximum file size in bytes
        min_duration_seconds: Minimum audio duration in seconds
        max_duration_seconds: Maximum audio duration in seconds
        min_modified_time: Minimum file modification time (Unix timestamp)
        max_modified_time: Maximum file modification time (Unix timestamp)
        format: Specific audio format to filter by (e.g., 'mp3', 'wav')
        sort_by: Field to sort by (name, size, duration, modified_time, format)
        reverse: Set to true for descending order

    Returns:
        List of FilePathSupportParams with full metadata and model support info
    """
    service = FileService()
    return await service.list_audio_files(
        pattern=pattern,
        min_size_bytes=min_size_bytes,
        max_size_bytes=max_size_bytes,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        min_modified_time=min_modified_time,
        max_modified_time=max_modified_time,
        format_filter=format,
        sort_by=sort_by,
        reverse=reverse,
    )


async def get_latest_audio() -> FilePathSupportParams:
    """Get the most recently modified audio file with model support info.

    Returns:
        FilePathSupportParams with metadata for the latest audio file
    """
    service = FileService()
    return await service.get_latest_audio_file()


# ==================== AUDIO PROCESSING TOOLS ====================


async def convert_audio(
    input_file_name: str,
    target_format: SupportedChatWithAudioFormat = "mp3",
    output_file_name: str | None = None,
) -> AudioProcessingResult:
    """Convert audio file to supported format (mp3 or wav for GPT-4o compatibility).

    Args:
        input_file_name: Name of the input audio file to process
        target_format: Target audio format (mp3 or wav)
        output_file_name: Optional custom name for output file

    Returns:
        AudioProcessingResult with name of the converted audio file
    """
    service = AudioService()
    return await service.convert_audio(
        input_filename=input_file_name,
        output_filename=output_file_name,
        target_format=target_format,
    )


async def compress_audio(
    input_file_name: str,
    max_mb: int = 25,
    output_file_name: str | None = None,
) -> AudioProcessingResult:
    """Compress audio file if it exceeds size limit.

    Automatically adjusts bitrate to meet target size. Useful for preparing
    files for API upload (25MB limit).

    Args:
        input_file_name: Name of the input audio file to process
        max_mb: Maximum file size in MB (default 25)
        output_file_name: Optional custom name for output file

    Returns:
        AudioProcessingResult with name of compressed file (or original if no compression needed)

    Raises:
        ValueError: If max_mb is out of valid range (1-1000)
    """
    # Validate max_mb parameter
    if not 1 <= max_mb <= 1000:
        raise ValueError(f"max_mb must be between 1 and 1000, got {max_mb}")

    service = AudioService()
    return await service.compress_audio(
        input_filename=input_file_name,
        output_filename=output_file_name,
        max_mb=max_mb,
    )


# ==================== TRANSCRIPTION TOOLS ====================


async def transcribe_audio(
    input_file_name: str,
    model: AudioModel = "gpt-4o-mini-transcribe",
    response_format: AudioResponseFormat = "text",
    prompt: str | None = None,
    timestamp_granularities: list[Literal["word", "segment"]] | None = None,
) -> TranscriptionResult:
    """Transcribe audio using OpenAI Whisper or GPT-4o models.

    Args:
        input_file_name: Name of the input audio file to process
        model: Transcription model (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)
        response_format: Response format (text, json, verbose_json, srt, vtt)
        prompt: Optional prompt to guide the transcription
        timestamp_granularities: Optional timestamp granularities (word, segment)

    Returns:
        TranscriptionResult with transcribed text and metadata
    """
    service = TranscriptionService()
    return await service.transcribe_audio(
        filename=input_file_name,
        model=model,
        response_format=response_format,
        prompt=prompt,
        timestamp_granularities=timestamp_granularities,
    )


async def chat_with_audio(
    input_file_name: str,
    model: AudioChatModel = "gpt-4o-audio-preview",
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> ChatResult:
    """Interactive audio analysis using GPT-4o audio models.

    Have a conversation about audio content, ask questions, analyze tone,
    or extract insights from audio.

    Args:
        input_file_name: Name of the input audio file to process
        model: Audio chat model to use (gpt-4o-audio-preview recommended)
        system_prompt: Optional system prompt for conversation context
        user_prompt: Optional user prompt with questions or instructions

    Returns:
        ChatResult with conversational response text
    """
    service = TranscriptionService()
    return await service.chat_with_audio(
        filename=input_file_name,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


async def transcribe_with_enhancement(
    input_file_name: str,
    enhancement_type: EnhancementType = "detailed",
    model: AudioModel = "gpt-4o-mini-transcribe",
    response_format: AudioResponseFormat = "text",
    timestamp_granularities: list[Literal["word", "segment"]] | None = None,
) -> TranscriptionResult:
    """Enhanced transcription with specialized templates.

    Enhancement types provide different styles of transcription:
    - detailed: Includes tone, emotion, and background details
    - storytelling: Transforms transcript into narrative form
    - professional: Creates formal, business-appropriate output
    - analytical: Adds analysis of speech patterns and key points

    Args:
        input_file_name: Name of the input audio file to process
        enhancement_type: Type of enhancement to apply
        model: Transcription model to use
        response_format: Response format
        timestamp_granularities: Optional timestamp granularities

    Returns:
        TranscriptionResult with enhanced transcription
    """
    service = TranscriptionService()
    return await service.transcribe_enhanced(
        filename=input_file_name,
        enhancement_type=enhancement_type,
        model=model,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
    )


# ==================== TEXT-TO-SPEECH TOOLS ====================


async def create_audio(
    text_prompt: str,
    model: SpeechModel = "gpt-4o-mini-tts",
    voice: TTSVoice = "alloy",
    instructions: str | None = None,
    speed: float = 1.0,
    output_file_name: str | None = None,
) -> TTSResult:
    """Generate text-to-speech audio using OpenAI TTS API.

    Handles texts of any length by splitting into chunks at natural boundaries
    and concatenating the audio. OpenAI TTS has a 4096 character limit per request.

    Args:
        text_prompt: Text to convert to speech
        model: TTS model to use (gpt-4o-mini-tts recommended)
        voice: Voice for TTS (alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer)
        instructions: Optional instructions for speech style (tonality, accent, etc.)
        speed: Speed of speech conversion (0.25 to 4.0)
        output_file_name: Optional custom name for output file

    Returns:
        TTSResult with name of the generated audio file

    Raises:
        ValueError: If speed is out of valid range (0.25-4.0)
    """
    # Validate speed parameter
    if not 0.25 <= speed <= 4.0:
        raise ValueError(f"speed must be between 0.25 and 4.0, got {speed}")

    service = TTSService()
    return await service.create_speech(
        text_prompt=text_prompt,
        output_filename=output_file_name,
        model=model,
        voice=voice,
        instructions=instructions,
        speed=speed,
    )
