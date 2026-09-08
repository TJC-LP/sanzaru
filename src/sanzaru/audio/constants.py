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
# tier (Flash/Turbo 4→30, other models 2→15). These are the Free-tier limits:
# verified against a live Free account, 3 concurrent v3 requests returns HTTP 429.
# Paid tiers should raise this with SANZARU_ELEVENLABS_MAX_CONCURRENCY.
ELEVENLABS_DEFAULT_CONCURRENCY: dict[ElevenLabsModel, int] = {
    "eleven_v3": 2,
    "eleven_multilingual_v2": 2,
    "eleven_flash_v2_5": 4,
    "eleven_turbo_v2_5": 4,
}

# 44.1kHz/128kbps mp3 is available on every tier (192k requires Creator+), and
# mp3 keeps the podcast stitch path's AudioSegment.from_mp3 contract intact.
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"

# ---------- Dialogue rendering ----------
# "segments": one TTS request per segment, joined with explicit silence gaps.
# "dialogue": consecutive turns by one provider go out in a single request so
# the model paces the conversation itself (natural turn-taking, reactions that
# land on the previous line).
PodcastRenderMode = Literal["segments", "dialogue"]
RENDER_MODES: tuple[str, ...] = ("segments", "dialogue")
DEFAULT_RENDER_MODE: PodcastRenderMode = "segments"

# Only eleven_v3 is trained for multi-speaker dialogue.
ELEVENLABS_DIALOGUE_MODELS: frozenset[str] = frozenset({"eleven_v3"})

# Per-request budget for the whole conversation, summed over inputs[].text.
# /v1/text-to-dialogue documents 2,000 as the ceiling for reliable generation:
# past it a request "can terminate early in streaming responses", which our
# stream reader cannot tell apart from a complete take — it would ship a
# truncated episode as a success. Not the text-to-speech per-request budget
# (ELEVENLABS_MAX_CHARS), which is far larger. Longer runs are split at turn
# boundaries into several dialogue calls.
ELEVENLABS_DIALOGUE_MAX_CHARS = 2000

# Distinct voices a run needs before batching is worth it. A stretch of turns in
# one voice has no turn-taking for the model to pace, so a dialogue request buys
# nothing and silently drops the pause_after between those paragraph beats. At 2
# this also excludes a lone turn, which would additionally lose its per-speaker
# voice_settings.
MIN_DIALOGUE_SPEAKERS = 2

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

# Extensions we are willing to hand to ffmpeg/pydub as the *input demuxer*.
#
# pydub selects the demuxer from `format=<ext>`, so taking that from a file's own
# (untrusted) extension is a local-file-inclusion vector: playlist demuxers
# (hls, concat, dash, m3u8) treat the file's *content* as references to other
# local files and decode those into the result (CWE-610).
#
# The rule is "a real, self-contained container", not "a format we transcribe".
# Deriving this from TRANSCRIBE_AUDIO_FORMATS was too narrow and broke the tool
# it was protecting: `convert_audio` exists precisely to turn formats the API
# cannot take into ones it can, and .aac/.opus/.aiff are the reason anyone calls
# it. What matters for the vulnerability is only that the demuxer cannot
# dereference an embedded path.
DECODABLE_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        "aac",
        "aif",
        "aifc",
        "aiff",
        "amr",
        "au",
        "caf",
        "flac",
        "m4a",
        "m4b",
        "mka",
        "mov",
        "mp3",
        "mp4",
        "mpeg",
        "mpga",
        "oga",
        "ogg",
        "opus",
        "wav",
        "webm",
        "wma",
    }
)

#: What a caller may name as a convert/compress *output*. Deliberately the
#: narrow set: an output name is a file this server creates, and there is no
#: reason it should be able to mint a `.json` manifest or an `.html` page that
#: the /media route would then serve (CWE-73/CWE-79).
SAFE_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    ext.lstrip(".") for ext in (TRANSCRIBE_AUDIO_FORMATS | CHAT_WITH_AUDIO_FORMATS)
)


def safe_audio_format(name_or_suffix: str, *, allowed: frozenset[str] | None = None) -> str:
    """Return the demuxer/format name for an allowlisted audio extension.

    Accepts either a filename (``track.mp3``) or a bare/dotted suffix
    (``mp3`` / ``.mp3``) and returns the lowercased extension. Raises
    ``ValueError`` for anything outside `allowed`, so a planted playlist file
    (``evil.hls``, ``evil.concat``) is refused before ``AudioSegment.from_file``
    can invoke a demuxer that opens other local files.

    `allowed` defaults to the decodable set, which is the one that matters for
    the demuxer; pass :data:`SAFE_AUDIO_EXTENSIONS` when validating a name this
    server is about to *create*.
    """
    permitted = DECODABLE_AUDIO_EXTENSIONS if allowed is None else allowed
    ext = name_or_suffix.rsplit(".", 1)[-1].lower() if "." in name_or_suffix else name_or_suffix.lower()
    if ext not in permitted:
        raise ValueError(f"unsupported audio format {ext!r}; allowed: {', '.join(sorted(permitted))}")
    return ext


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
