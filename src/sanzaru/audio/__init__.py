# SPDX-License-Identifier: MIT
"""Audio feature for sanzaru - Whisper transcription, GPT-4o audio chat, and TTS.

This module provides comprehensive audio processing capabilities via OpenAI APIs.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from .config import AudioConfig, get_audio_config, get_audio_path
from .constants import (
    CHAT_WITH_AUDIO_FORMATS,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_TTS_MAX_LENGTH,
    DEFAULT_TTS_SAMPLE_RATE,
    ENHANCEMENT_PROMPTS,
    TRANSCRIBE_AUDIO_FORMATS,
    AudioChatModel,
    EnhancementType,
    SortBy,
    SupportedChatWithAudioFormat,
    TTSVoice,
)
from .file_filter import FileFilterSorter
from .models import (
    AudioProcessingResult,
    BaseAudioInputParams,
    BaseInputPath,
    ChatResult,
    ChatWithAudioInputParams,
    CompressAudioInputParams,
    ConvertAudioInputParams,
    CreateClaudecastInputParams,
    FilePathSupportParams,
    ListAudioFilesInputParams,
    TranscribeAudioInputParams,
    TranscribeAudioInputParamsBase,
    TranscribeWithEnhancementInputParams,
    TranscriptionResult,
    TTSResult,
)
from .processor import AudioProcessor

__all__ = [
    # Config
    "AudioConfig",
    "get_audio_config",
    "get_audio_path",
    # Constants
    "TRANSCRIBE_AUDIO_FORMATS",
    "CHAT_WITH_AUDIO_FORMATS",
    "DEFAULT_MAX_FILE_SIZE_MB",
    "DEFAULT_TTS_MAX_LENGTH",
    "DEFAULT_TTS_SAMPLE_RATE",
    "ENHANCEMENT_PROMPTS",
    "AudioChatModel",
    "EnhancementType",
    "SortBy",
    "SupportedChatWithAudioFormat",
    "TTSVoice",
    # Core classes
    "AudioProcessor",
    "FileFilterSorter",
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

__version__ = "0.2.0"
