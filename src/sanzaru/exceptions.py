# SPDX-License-Identifier: MIT
"""Custom exceptions for sanzaru MCP server.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""


class SanzaruError(Exception):
    """Base exception for all sanzaru errors."""

    pass


class ConfigurationError(SanzaruError):
    """Raised when there is a configuration issue."""

    pass


class AudioFileError(SanzaruError):
    """Base exception for audio file-related errors."""

    pass


class AudioFileNotFoundError(AudioFileError):
    """Raised when an audio file is not found."""

    pass


class UnsupportedAudioFormatError(AudioFileError):
    """Raised when an audio format is not supported."""

    pass


class AudioProcessingError(AudioFileError):
    """Raised when audio processing fails."""

    pass


class AudioConversionError(AudioProcessingError):
    """Raised when audio format conversion fails."""

    pass


class AudioCompressionError(AudioProcessingError):
    """Raised when audio compression fails."""

    pass


class TranscriptionError(SanzaruError):
    """Base exception for transcription-related errors."""

    pass


class TranscriptionAPIError(TranscriptionError):
    """Raised when the transcription API call fails."""

    pass


class TTSError(SanzaruError):
    """Base exception for text-to-speech errors."""

    pass


class TTSAPIError(TTSError):
    """Raised when the TTS API call fails."""

    pass
