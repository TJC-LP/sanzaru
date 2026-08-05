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


class RealtimeError(SanzaruError):
    """Base exception for realtime conversation simulation."""

    pass


class RealtimeAPIError(RealtimeError):
    """Raised when a Realtime API session or response fails."""

    pass


class CostCeilingError(RealtimeError):
    """Raised when a simulation is stopped by its cost ceiling.

    Carries what was already recorded so the caller can report which acts are
    safe on disk — an abort that silently discards paid-for audio is worse than
    no ceiling at all.

    `suggested_limit_usd` is what makes the abort recoverable rather than a
    loop: a resumed run replays the spend already on disk against the *same*
    restored ceiling, so "resume to finish" only finishes if the caller also
    raises it. The number is the ceiling that would have covered this run.
    """

    def __init__(
        self,
        message: str,
        *,
        spent_usd: float,
        limit_usd: float,
        completed_acts: list[str],
        suggested_limit_usd: float | None = None,
    ) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.completed_acts = completed_acts
        self.suggested_limit_usd = suggested_limit_usd
