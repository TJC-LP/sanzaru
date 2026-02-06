"""Audio processing service - orchestrates domain and infrastructure.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from pathlib import Path

from ...config import logger
from ...storage import get_storage
from .. import AudioProcessor
from ..constants import DEFAULT_MAX_FILE_SIZE_MB, SupportedChatWithAudioFormat
from ..models import AudioProcessingResult


class AudioService:
    """Service for audio conversion and compression operations."""

    def __init__(self):
        """Initialize the audio service."""
        self.processor = AudioProcessor()

    async def convert_audio(
        self,
        input_filename: str,
        output_filename: str | None = None,
        target_format: SupportedChatWithAudioFormat = "mp3",
    ) -> AudioProcessingResult:
        """Convert audio file to supported format (mp3 or wav).

        Args:
        ----
            input_filename: Name of input audio file.
            output_filename: Optional name for output file.
            target_format: Target format ('mp3' or 'wav').

        Returns:
        -------
            AudioProcessingResult: Result with name of the converted audio file.

        """
        output_name = output_filename or f"{Path(input_filename).stem}.{target_format}"
        storage = get_storage()

        async with (
            storage.local_path("audio", input_filename) as input_path,
            storage.local_tempfile("audio", output_name) as output_path,
        ):
            # Load audio from local path (pydub needs filesystem access)
            audio_data = await self.processor.load_audio_from_path(input_path)

            # Convert format — writes to output_path via pydub
            await self.processor.convert_audio_format(
                audio_data=audio_data,
                target_format=target_format,
                output_path=output_path,
            )
            # local_tempfile uploads to storage on context exit

        return AudioProcessingResult(output_file=output_name)

    async def compress_audio(
        self,
        input_filename: str,
        output_filename: str | None = None,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> AudioProcessingResult:
        """Compress audio file if it exceeds size limit.

        Args:
        ----
            input_filename: Name of input audio file.
            output_filename: Optional name for output file.
            max_mb: Maximum file size in MB.

        Returns:
        -------
            AudioProcessingResult: Result with name of the compressed audio file (or original if no compression needed).

        """
        storage = get_storage()

        # Check if compression is needed
        info = await storage.stat("audio", input_filename)
        needs_compression = self.processor.calculate_compression_needed(info.size_bytes, max_mb)

        if not needs_compression:
            return AudioProcessingResult(output_file=input_filename)  # No compression needed

        logger.info(f"File '{input_filename}' size > {max_mb}MB. Attempting compression...")

        # Convert to MP3 if not already
        if not input_filename.lower().endswith(".mp3"):
            logger.info("Converting to MP3 first...")
            conversion_result = await self.convert_audio(input_filename, None, "mp3")
            input_filename = conversion_result.output_file

        # Determine output filename
        stem = Path(input_filename).stem
        output_name = output_filename or f"compressed_{stem}.mp3"

        logger.debug(f"Original file: {input_filename}")
        logger.debug(f"Output file: {output_name}")

        async with (
            storage.local_path("audio", input_filename) as input_path,
            storage.local_tempfile("audio", output_name) as output_path,
        ):
            # Load and compress via pydub (needs local filesystem)
            audio_data = await self.processor.load_audio_from_path(input_path)
            await self.processor.compress_mp3(audio_data, output_path)
            # local_tempfile uploads to storage on context exit

        # Get compressed size for logging
        compressed_info = await storage.stat("audio", output_name)
        logger.info(f"Compressed file size: {compressed_info.size_bytes} bytes")

        return AudioProcessingResult(output_file=output_name)

    async def maybe_compress_file(
        self,
        input_filename: str,
        output_filename: str | None = None,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> AudioProcessingResult:
        """Compress file if needed, maintaining backward compatibility.

        This method provides the same interface as the original server.py function.

        Args:
        ----
            input_filename: Name of input audio file.
            output_filename: Optional name for output file.
            max_mb: Maximum file size in MB.

        Returns:
        -------
            AudioProcessingResult: Result with name of the (possibly compressed) audio file.

        """
        return await self.compress_audio(input_filename, output_filename, max_mb)
