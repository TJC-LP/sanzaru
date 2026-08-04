# SPDX-License-Identifier: MIT
"""Constants and enumerations for sanzaru audio features.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from enum import Enum
from typing import Literal

from openai.types import AudioModel
from openai.types.audio.speech_model import SpeechModel

# Type Aliases
SupportedChatWithAudioFormat = Literal["mp3", "wav"]
AudioChatModel = Literal[
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
]
EnhancementType = Literal["detailed", "storytelling", "professional", "analytical"]
TTSVoice = Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]

# ---------- TTS providers ----------
TTSProviderName = Literal["openai", "elevenlabs"]
ElevenLabsModel = Literal["eleven_v3", "eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5"]

DEFAULT_TTS_PROVIDER: TTSProviderName = "openai"
DEFAULT_OPENAI_TTS_MODEL: SpeechModel = "gpt-4o-mini-tts"
DEFAULT_OPENAI_VOICE: TTSVoice = "alloy"
DEFAULT_ELEVENLABS_MODEL: ElevenLabsModel = "eleven_v3"

OPENAI_TTS_MODELS: list[SpeechModel] = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
ELEVENLABS_MODELS: list[ElevenLabsModel] = [
    "eleven_v3",
    "eleven_multilingual_v2",
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
]

# Per-request text budget. Ours are deliberately conservative: the split is only
# an upper bound, and over-long requests fail the whole segment.
ELEVENLABS_MAX_CHARS: dict[ElevenLabsModel, int] = {
    "eleven_v3": 3000,
    "eleven_multilingual_v2": 10000,
    "eleven_flash_v2_5": 40000,
    "eleven_turbo_v2_5": 40000,
}

# Concurrent requests allowed in parallel. ElevenLabs caps this per subscription
# tier (Flash/Turbo 4→30, other models 2→15); these are Free-tier-safe floors,
# overridable with SANZARU_ELEVENLABS_MAX_CONCURRENCY.
ELEVENLABS_DEFAULT_CONCURRENCY: dict[ElevenLabsModel, int] = {
    "eleven_v3": 3,
    "eleven_multilingual_v2": 3,
    "eleven_flash_v2_5": 4,
    "eleven_turbo_v2_5": 4,
}

# 44.1kHz/128kbps mp3 is available on every tier (192k requires Creator+), and
# mp3 keeps the podcast stitch path's AudioSegment.from_mp3 contract intact.
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"

# ElevenLabs speed lives in voice_settings and has a much narrower range than
# OpenAI's 0.25-4.0. eleven_v3 does not support it at all.
ELEVENLABS_SPEED_RANGE = (0.7, 1.2)
OPENAI_SPEED_RANGE = (0.25, 4.0)

# Segments are decoded to this rate before stitching so a mixed-provider episode
# (OpenAI mp3 is 24kHz, ElevenLabs mp3_44100_128 is 44.1kHz) is deterministic
# regardless of segment order.
PODCAST_TARGET_FRAME_RATE = 44100

# Model Lists (for file support detection)
TRANSCRIPTION_MODELS: list[AudioModel] = [
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe-diarize",
]

AUDIO_CHAT_MODELS: list[AudioChatModel] = [
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
]

# Supported Audio Formats
TRANSCRIBE_AUDIO_FORMATS = {
    ".flac",  # Added FLAC support
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",  # Added OGG support
    ".wav",
    ".webm",
}

CHAT_WITH_AUDIO_FORMATS = {".mp3", ".wav"}

# Enhancement Prompts
ENHANCEMENT_PROMPTS: dict[EnhancementType, str] = {
    "detailed": "The following is a detailed transcript that includes all verbal and non-verbal elements. "
    "Background noises are noted in [brackets]. Speech characteristics like [pause], [laughs], and [sighs] "
    "are preserved. Filler words like 'um', 'uh', 'like', and 'you know' are included. "
    "Hello... [deep breath] Let me explain what I mean by that. [background noise] You know, it's like...",
    "storytelling": "The following is a natural conversation with proper punctuation and flow. "
    "Each speaker's words are captured in a new paragraph with emotional context preserved. "
    "Hello! I'm excited to share this story with you. It began on a warm summer morning...",
    "professional": "The following is a clear, professional transcript with proper capitalization and punctuation. "
    "Each sentence is complete and properly structured. Technical terms and acronyms are preserved exactly. "
    "Welcome to today's presentation on the Q4 financial results. Our KPIs show significant growth.",
    "analytical": "The following is a precise technical transcript that preserves speech patterns and terminology. "
    "Note changes in speaking pace, emphasis, and technical terms exactly as spoken. "
    "Preserve specialized vocabulary, acronyms, and technical jargon with high fidelity. "
    "Example: The API endpoint /v1/completions [spoken slowly] accepts JSON payloads "
    "with a maximum token count of 4096 [emphasis on numbers].",
}


class SortBy(str, Enum):
    """Sorting options for audio files."""

    NAME = "name"
    SIZE = "size"
    DURATION = "duration"
    MODIFIED_TIME = "modified_time"
    FORMAT = "format"


# Default Values
DEFAULT_MAX_FILE_SIZE_MB = 25
DEFAULT_TTS_MAX_LENGTH = 4000
DEFAULT_TTS_SAMPLE_RATE = 11025
