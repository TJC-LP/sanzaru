"""Audio processing service - orchestrates domain and infrastructure.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import shutil
from pathlib import Path

import anyio.to_thread

from ...config import logger
from ...storage import get_storage
from ...utils import reject_reserved_name
from .. import AudioProcessor
from ..constants import (
    DEFAULT_MAX_FILE_SIZE_MB,
    SAFE_AUDIO_EXTENSIONS,
    SupportedChatWithAudioFormat,
    safe_audio_format,
)
from ..models import AudioProcessingResult


def _require_audio_name(filename: str, role: str, *, is_output: bool = False) -> None:
    """Refuse a convert/compress path whose extension is not a real audio format.

    The no-compression branch (and format conversion) ultimately does a plain
    byte copy for same-size inputs; without this a caller could duplicate any
    audio-directory file (e.g. a run manifest ``simrun_*.json``) under an
    arbitrary name/extension — minting a ``.json``/``.html`` artifact served by
    the ``/media`` route (CWE-79). Both the input and any caller-chosen output
    are constrained to audio extensions.

    Outputs are additionally refused the run-bookkeeping namespace: an act
    checkpoint *is* an mp3, so the extension rule alone would still let one be
    overwritten (CWE-73). Inputs are not — reading your own checkpoint back is
    legitimate, and only the write destroys anything.
    """
    try:
        # Outputs are held to the narrower set — this server is creating that
        # file. Inputs only need to be a container ffmpeg can decode without
        # dereferencing a path out of its contents, which is what keeps
        # `convert_audio` able to do its actual job (.aac, .opus, ...).
        safe_audio_format(filename, allowed=SAFE_AUDIO_EXTENSIONS if is_output else None)
    except ValueError as exc:
        raise ValueError(f"{role} {filename!r}: {exc}") from exc
    if is_output:
        reject_reserved_name(filename, role)


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
        _require_audio_name(input_filename, "input file")
        output_name = output_filename or f"{Path(input_filename).stem}.{target_format}"
        # Check the name that is actually written, not the argument. Guarding
        # only an explicit `output_filename` left the derived one unchecked, and
        # since inputs are deliberately *not* reserved-checked, converting
        # `<slug>_<runid>_act1.wav` produced `<slug>_<runid>_act1.mp3` — the
        # victim's checkpoint audio, overwritten, through the guard.
        _require_audio_name(output_name, "output file", is_output=True)
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

    async def _copy(self, input_filename: str, output_filename: str) -> None:
        """Copy one audio file to another name within the audio path type.

        Goes through `local_path`/`local_tempfile` rather than `read`+`write`
        so the local backend does a plain filesystem copy instead of holding
        the whole file in memory; the Databricks backend still round-trips.
        """
        storage = get_storage()
        async with (
            storage.local_path("audio", input_filename) as source,
            storage.local_tempfile("audio", output_filename) as destination,
        ):
            await anyio.to_thread.run_sync(shutil.copyfile, source, destination)

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
            AudioProcessingResult: Result with name of the compressed audio file. When no
            compression was needed, that is `output_filename` if one was asked for, and the
            input's own name otherwise.

        """
        _require_audio_name(input_filename, "input file")
        if output_filename is not None:
            _require_audio_name(output_filename, "output file", is_output=True)
        storage = get_storage()

        # Check if compression is needed
        info = await storage.stat("audio", input_filename)
        needs_compression = self.processor.calculate_compression_needed(info.size_bytes, max_mb)

        if not needs_compression:
            # The contract is "a file exists at output_filename", not "a file was
            # compressed" (#59). Returning the *input's* name here let the CLI's
            # cross-directory branch shutil.move the source away, destroying it —
            # and the caller who ran compress defensively before transcribing lost
            # exactly the originals that were already small enough.
            if output_filename is None or output_filename == input_filename:
                return AudioProcessingResult(output_file=input_filename)
            await self._copy(input_filename, output_filename)
            return AudioProcessingResult(output_file=output_filename)

        logger.info(f"File '{input_filename}' size > {max_mb}MB. Attempting compression...")

        # Convert to MP3 if not already
        if not input_filename.lower().endswith(".mp3"):
            logger.info("Converting to MP3 first...")
            conversion_result = await self.convert_audio(input_filename, None, "mp3")
            input_filename = conversion_result.output_file

        # Determine output filename, then check the name actually written —
        # `compressed_<stem>.mp3` is derived, so guarding only the argument
        # would leave it unchecked (the same gap convert_audio had).
        stem = Path(input_filename).stem
        output_name = output_filename or f"compressed_{stem}.mp3"
        _require_audio_name(output_name, "output file", is_output=True)

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
