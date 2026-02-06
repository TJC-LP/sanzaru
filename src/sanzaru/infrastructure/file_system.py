# SPDX-License-Identifier: MIT
"""File system operations for audio file management.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import re

import anyio
from openai.types import AudioModel
from pydub import AudioSegment  # type: ignore

from ..audio.constants import (
    AUDIO_CHAT_MODELS,
    CHAT_WITH_AUDIO_FORMATS,
    TRANSCRIBE_AUDIO_FORMATS,
    TRANSCRIPTION_MODELS,
    AudioChatModel,
)
from ..audio.models import FilePathSupportParams
from ..exceptions import AudioFileError, AudioFileNotFoundError
from ..storage import get_storage
from ..storage.protocol import FileInfo, StorageBackend


class FileSystemRepository:
    """Repository for file system operations related to audio files.

    Delegates I/O to the configured :class:`StorageBackend` (local or
    Databricks), keeping the existing API for backward compatibility
    with the audio service layer.
    """

    def __init__(self, storage: StorageBackend | None = None):
        """Initialize the file system repository.

        Args:
            storage: Storage backend to use.  Defaults to ``get_storage()``.
        """
        self._storage = storage or get_storage()

    async def get_audio_file_support(self, filename: str) -> FilePathSupportParams:
        """Determine audio transcription file format support and metadata.

        Includes file size, format, and duration information where available.

        Args:
            filename: Name of the audio file.

        Returns:
            FilePathSupportParams: File metadata and model support information.
        """
        file_ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        audio_format = file_ext[1:] if file_ext.startswith(".") else file_ext

        transcription_support: list[AudioModel] | None = (
            TRANSCRIPTION_MODELS if file_ext in TRANSCRIBE_AUDIO_FORMATS else None
        )
        chat_support: list[AudioChatModel] | None = AUDIO_CHAT_MODELS if file_ext in CHAT_WITH_AUDIO_FORMATS else None

        # Get file stats from storage backend
        info = await self._storage.stat("audio", filename)

        # Get duration if possible (downloads file for remote backends)
        duration_seconds = None
        try:
            async with self._storage.local_path("audio", filename) as local:
                audio = await anyio.to_thread.run_sync(lambda: AudioSegment.from_file(str(local), format=audio_format))
                duration_seconds = len(audio) / 1000.0
        except Exception:
            pass

        return FilePathSupportParams(
            file_name=filename,
            transcription_support=transcription_support,
            chat_support=chat_support,
            modified_time=info.modified_timestamp,
            size_bytes=info.size_bytes,
            format=audio_format,
            duration_seconds=duration_seconds,
        )

    async def get_latest_audio_file(self) -> FilePathSupportParams:
        """Get the most recently modified audio file with model support info.

        Supported formats:
        - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
        - GPT-4o: mp3, wav

        Returns:
            FilePathSupportParams: File metadata and model support information.

        Raises:
            AudioFileNotFoundError: If no supported audio files are found.
            AudioFileError: If there's an error accessing audio files.
        """
        audio_extensions = TRANSCRIBE_AUDIO_FORMATS | CHAT_WITH_AUDIO_FORMATS
        try:
            file_infos = await self._storage.list_files("audio", extensions=audio_extensions)

            if not file_infos:
                raise AudioFileNotFoundError("No supported audio files found")

            latest = max(file_infos, key=lambda x: x.modified_timestamp)
            return await self.get_audio_file_support(latest.name)

        except AudioFileNotFoundError:
            raise
        except Exception as e:
            raise AudioFileError(f"Failed to get latest audio file: {e}") from e

    async def list_audio_files(
        self,
        pattern: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        format_filter: str | None = None,
    ) -> list[FileInfo]:
        """List audio files matching the given criteria.

        Args:
            pattern: Optional regex pattern to filter files by name.
            min_size_bytes: Minimum file size in bytes.
            max_size_bytes: Maximum file size in bytes.
            format_filter: Specific audio format to filter by (e.g., 'mp3', 'wav').

        Returns:
            list[FileInfo]: List of file info objects matching the criteria.
        """
        audio_extensions = TRANSCRIBE_AUDIO_FORMATS | CHAT_WITH_AUDIO_FORMATS
        file_infos = await self._storage.list_files("audio", extensions=audio_extensions)

        results: list[FileInfo] = []
        for info in file_infos:
            file_ext = ("." + info.name.rsplit(".", 1)[-1].lower()) if "." in info.name else ""

            # Apply regex pattern filtering if provided
            if pattern and not re.search(pattern, info.name):
                continue

            # Apply format filtering if provided
            if format_filter and file_ext[1:].lower() != format_filter.lower():
                continue

            # Apply size filtering if provided
            if min_size_bytes is not None and info.size_bytes < min_size_bytes:
                continue
            if max_size_bytes is not None and info.size_bytes > max_size_bytes:
                continue

            results.append(info)

        return results

    async def read_audio_file(self, filename: str) -> bytes:
        """Read an audio file asynchronously.

        Args:
            filename: Name of the audio file.

        Returns:
            bytes: The file content as bytes.

        Raises:
            AudioFileNotFoundError: If the file doesn't exist.
            AudioFileError: If there's an error reading the file.
        """
        try:
            return await self._storage.read("audio", filename)
        except (FileNotFoundError, ValueError) as e:
            raise AudioFileNotFoundError(f"File not found: {filename}") from e
        except Exception as e:
            raise AudioFileError(f"Failed to read audio file '{filename}': {e}") from e

    async def write_audio_file(self, filename: str, content: bytes) -> str:
        """Write audio content to a file asynchronously.

        Args:
            filename: Name of the file to write.
            content: Audio content as bytes.

        Returns:
            str: Display path of the written file.

        Raises:
            AudioFileError: If there's an error writing the file.
        """
        try:
            return await self._storage.write("audio", filename, content)
        except Exception as e:
            raise AudioFileError(f"Failed to write audio file '{filename}': {e}") from e

    async def get_file_size(self, filename: str) -> int:
        """Get the size of a file in bytes.

        Args:
            filename: Name of the file.

        Returns:
            int: File size in bytes.

        Raises:
            AudioFileNotFoundError: If the file doesn't exist.
        """
        try:
            info = await self._storage.stat("audio", filename)
            return info.size_bytes
        except (FileNotFoundError, ValueError) as e:
            raise AudioFileNotFoundError(f"File not found: {filename}") from e
