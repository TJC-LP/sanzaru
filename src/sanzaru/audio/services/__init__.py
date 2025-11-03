"""Audio services for sanzaru - transcription, file management, TTS, processing.

This module provides high-level service classes that orchestrate audio operations.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

# TODO(Track B): Uncomment once Track A and Track C provide dependencies
# These imports will work once the migration is complete
try:
    from .audio_service import AudioService
    from .file_service import FileService
    from .transcription_service import TranscriptionService
    from .tts_service import TTSService

    __all__ = [
        "AudioService",
        "FileService",
        "TranscriptionService",
        "TTSService",
    ]
except ImportError:
    # Services depend on Track A (domain) and Track C (infrastructure)
    # This is expected during parallel migration
    __all__ = []
