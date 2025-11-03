# SPDX-License-Identifier: MIT
"""Audio domain models for sanzaru.

This module contains Pydantic models for audio processing, transcription,
text-to-speech, and related functionality.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from pathlib import Path
from typing import Any, Literal

from openai.types import AudioModel, AudioResponseFormat
from openai.types.audio.speech_model import SpeechModel
from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_MAX_FILE_SIZE_MB,
    ENHANCEMENT_PROMPTS,
    AudioChatModel,
    EnhancementType,
    SortBy,
    SupportedChatWithAudioFormat,
    TTSVoice,
)

# ========== Base Models ==========


class BaseInputPath(BaseModel):
    """Base file path input."""

    input_file_path: Path = Field(description="Path to the input audio file to process")

    model_config = {"arbitrary_types_allowed": True}


class BaseAudioInputParams(BaseInputPath):
    """Base params for audio operations."""

    output_file_path: Path | None = Field(
        default=None,
        description="Optional custom path for the output file. "
        "If not provided, defaults to input_file_path with appropriate extension",
    )


# ========== Audio Processing Models ==========


class ConvertAudioInputParams(BaseAudioInputParams):
    """Params for converting audio to mp3 or wav."""

    target_format: SupportedChatWithAudioFormat = Field(
        default="mp3", description="Target audio format to convert to (mp3 or wav)"
    )


class CompressAudioInputParams(BaseAudioInputParams):
    """Params for compressing audio."""

    max_mb: int = Field(
        default=DEFAULT_MAX_FILE_SIZE_MB,
        gt=0,
        description="Maximum file size in MB. Files larger than this will be compressed",
    )


class FilePathSupportParams(BaseModel):
    """Params for checking if a file supports transcription."""

    file_name: str = Field(description="Name of the audio file")
    transcription_support: list[AudioModel] | None = Field(
        default=None, description="List of transcription models that support this file format"
    )
    chat_support: list[AudioChatModel] | None = Field(
        default=None, description="List of audio LLM models that support this file format"
    )
    modified_time: float = Field(description="Last modified timestamp of the file (Unix time)")
    size_bytes: int = Field(description="Size of the file in bytes")
    format: str = Field(description="Audio format of the file (e.g., 'mp3', 'wav')")
    duration_seconds: float | None = Field(
        default=None, description="Duration of the audio file in seconds, if available"
    )


class ListAudioFilesInputParams(BaseModel):
    """Input parameters for the list_audio_files tool."""

    pattern: str | None = Field(default=None, description="Optional regex pattern to filter audio files by name")
    min_size_bytes: int | None = Field(default=None, description="Minimum file size in bytes")
    max_size_bytes: int | None = Field(default=None, description="Maximum file size in bytes")
    min_duration_seconds: float | None = Field(default=None, description="Minimum audio duration in seconds")
    max_duration_seconds: float | None = Field(default=None, description="Maximum audio duration in seconds")
    min_modified_time: float | None = Field(default=None, description="Minimum file modification time (Unix timestamp)")
    max_modified_time: float | None = Field(default=None, description="Maximum file modification time (Unix timestamp)")
    format: str | None = Field(default=None, description="Specific audio format to filter by (e.g., 'mp3', 'wav')")
    sort_by: SortBy = Field(
        default=SortBy.NAME, description="Field to sort results by (name, size, duration, modified_time, format)"
    )
    reverse: bool = Field(default=False, description="Sort in reverse order if True")

    model_config = {"arbitrary_types_allowed": True}


# ========== Transcription Models ==========


class TranscribeAudioInputParamsBase(BaseInputPath):
    """Base params for transcribing audio with audio-to-text model."""

    model: AudioModel = Field(
        default="gpt-4o-mini-transcribe",
        description="The transcription model to use (e.g., 'whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe')",
    )
    response_format: AudioResponseFormat = Field(
        "text",
        description="The response format of the transcription model. "
        'Use `verbose_json` with `model="whisper-1"` for timestamps. '
        "`gpt-4o-transcribe` and `gpt-4o-mini-transcribe` only support `text` and `json`.",
    )
    timestamp_granularities: list[Literal["word", "segment"]] | None = Field(
        None,
        description="""The timestamp granularities to populate for this transcription.
`response_format` must be set `verbose_json` to use timestamp granularities.
Either or both of these options are supported: `word`, or `segment`.
Note: There is no additional latency for segment timestamps, but generating word timestamp incurs additional latency.
""",
    )


class TranscribeAudioInputParams(TranscribeAudioInputParamsBase):
    """Params for transcribing audio with audio-to-text model."""

    prompt: str | None = Field(
        None,
        description="""An optional prompt to guide the transcription model's output. Effective prompts can:

        1. Correct specific words/acronyms: Include technical terms or names that might be misrecognized
           Example: "The transcript discusses OpenAI's DALL·E and GPT-4 technology"

        2. Maintain context from previous segments: Include the last part of previous transcript
           Note: Model only considers final 224 tokens of the prompt

        3. Enforce punctuation: Include properly punctuated example text
           Example: "Hello, welcome to my lecture. Today, we'll discuss..."

        4. Preserve filler words: Include example with verbal hesitations
           Example: "Umm, let me think like, hmm... Okay, here's what I'm thinking"

        5. Set writing style: Use examples in desired format (simplified/traditional, formal/casual)

        The model will try to match the style and formatting of your prompt.""",
    )


class ChatWithAudioInputParams(BaseInputPath):
    """Params for transcribing audio with LLM using custom prompt."""

    system_prompt: str | None = Field(default=None, description="Custom system prompt to use.")
    user_prompt: str | None = Field(default=None, description="Custom user prompt to use.")
    model: AudioChatModel = Field(
        default="gpt-4o-audio-preview-2024-12-17", description="The audio LLM model to use for transcription"
    )


class TranscribeWithEnhancementInputParams(TranscribeAudioInputParamsBase):
    """Params for transcribing audio with LLM using template prompt."""

    enhancement_type: EnhancementType = Field(
        default="detailed",
        description="Type of enhancement to apply to the transcription: "
        "detailed, storytelling, professional, or analytical.",
    )

    def to_transcribe_audio_input_params(self) -> TranscribeAudioInputParams:
        """Transfer audio with LLM using custom prompt."""
        return TranscribeAudioInputParams(
            input_file_path=self.input_file_path,
            prompt=ENHANCEMENT_PROMPTS[self.enhancement_type],
            model=self.model,
            timestamp_granularities=self.timestamp_granularities,
            response_format=self.response_format,
        )


# ========== Text-to-Speech Models ==========


class CreateClaudecastInputParams(BaseModel):
    """Params for text-to-speech using OpenAI's API."""

    text_prompt: str = Field(description="Text to convert to speech")
    output_file_path: Path | None = Field(
        default=None, description="Output file path (defaults to speech.mp3 in current directory)"
    )
    model: SpeechModel = Field(
        default="gpt-4o-mini-tts", description="TTS model to use. gpt-4o-mini-tts is always preferred."
    )
    voice: TTSVoice = Field(
        default="alloy",
        description="Voice for the TTS",
    )
    instructions: str | None = Field(
        default=None,
        description="Optional instructions for the speech conversion, such as tonality, accent, style, etc.",
    )
    speed: float = Field(
        default=1.0,
        gt=0.25,
        lt=4.0,
        description="Speed of the speech conversion. Use if the user prompts slow or fast speech.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ========== Response Models ==========


class AudioProcessingResult(BaseModel):
    """Result from audio processing operations (convert, compress)."""

    output_file: str = Field(description="Name of the processed audio file")


class TranscriptionResult(BaseModel):
    """Result from transcription operations.

    Includes all fields that OpenAI's transcription API might return.
    """

    text: str = Field(description="The transcribed text")
    duration: float | None = Field(default=None, description="Duration of the audio in seconds")
    language: str | None = Field(default=None, description="Detected language of the audio")
    segments: list[dict[str, Any]] | None = Field(default=None, description="Timestamped segments")
    words: list[dict[str, Any]] | None = Field(default=None, description="Word-level timestamps")
    usage: dict[str, Any] | None = Field(default=None, description="Token usage information")
    logprobs: Any | None = Field(default=None, description="Log probabilities if requested")

    model_config = {"arbitrary_types_allowed": True}


class ChatResult(BaseModel):
    """Result from chat_with_audio operation."""

    text: str = Field(description="The response text from the audio chat")


class TTSResult(BaseModel):
    """Result from text-to-speech operations."""

    output_file: str = Field(description="Name of the generated audio file")


# ========== Exports ==========

__all__ = [
    # Base models
    "BaseInputPath",
    "BaseAudioInputParams",
    # Audio processing models
    "ConvertAudioInputParams",
    "CompressAudioInputParams",
    "FilePathSupportParams",
    "ListAudioFilesInputParams",
    # Transcription models
    "TranscribeAudioInputParamsBase",
    "TranscribeAudioInputParams",
    "ChatWithAudioInputParams",
    "TranscribeWithEnhancementInputParams",
    # TTS models
    "CreateClaudecastInputParams",
    # Response models
    "AudioProcessingResult",
    "TranscriptionResult",
    "ChatResult",
    "TTSResult",
]
