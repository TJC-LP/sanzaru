"""Audio processing service - orchestrates domain and infrastructure.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread

from ...config import logger
from ...infrastructure import FileSystemRepository
from ...storage import get_storage
from ...utils import reject_reserved_name
from .. import AudioProcessor
from ..constants import (
    DECODABLE_AUDIO_EXTENSIONS,
    DEFAULT_MAX_FILE_SIZE_MB,
    SAFE_AUDIO_EXTENSIONS,
    SupportedChatWithAudioFormat,
    safe_audio_format,
)
from ..models import AudioProcessingResult


def require_audio_name(filename: str, role: str, *, is_output: bool = False) -> None:
    """Refuse an audio filename whose extension is not a real audio format.

    Shared by convert/compress (both roles) and `TTSService.create_speech`
    (outputs). The no-compression branch of `compress_audio` is a plain byte
    copy that never decodes its input; without this a caller could duplicate
    any audio-directory file — a run manifest, say — under an arbitrary name
    and extension, minting a ``.json``/``.html`` artifact for the ``/media``
    route to serve (CWE-73, CWE-79).

    Inputs need only be a container ffmpeg can decode without dereferencing a
    path out of its contents (`DECODABLE_AUDIO_EXTENSIONS`), which is what
    keeps `convert_audio` able to do its actual job (.aac, .opus, ...).
    Outputs are held to the narrower `SAFE_AUDIO_EXTENSIONS`: this server is
    creating that file. A real dotted extension is required in both roles —
    `safe_audio_format` accepts ``"mp3"`` as a *format*, but as a *filename*
    that would write a file literally named ``mp3``.

    What this does not do, stated because an earlier docstring claimed it: it
    does not protect simulated-podcast act checkpoints. Those are ordinary
    ``.mp3`` names, and `reject_reserved_name` deliberately matches only run
    manifests (``simrun_<id>.json``) — see its comment in `utils.py` for why a
    name rule for checkpoints cannot be written without rejecting date-stamped
    recordings. Checkpoint protection belongs to the write path
    (`FileSystemRepository.write_audio_file`), which is why every output this
    module produces goes through that one method rather than straight to
    storage. The manifest check on outputs is forward-looking defense: today
    the extension rule already subsumes it (``.json`` is never an audio
    extension), so the branch is unreachable from here.

    Raises:
    ------
        ValueError: If the name has no extension, or its extension is outside
            the set for its role, or (outputs) it names run bookkeeping.

    """
    allowed = SAFE_AUDIO_EXTENSIONS if is_output else DECODABLE_AUDIO_EXTENSIONS
    stem, dot, _ext = filename.rpartition(".")
    if not dot or not stem:
        raise ValueError(
            f"{role} {filename!r:.80} has no audio extension; name it with one of: {', '.join(sorted(allowed))}"
        )
    try:
        safe_audio_format(filename, allowed=allowed)
    except ValueError as exc:
        raise ValueError(f"{role} {filename!r:.80}: {exc}") from exc
    if is_output:
        reject_reserved_name(filename, role)


@asynccontextmanager
async def _scratch_export_path(filename: str) -> AsyncIterator[Path]:
    """A private temp path for pydub to export into, removed on exit.

    pydub opens the export path itself, so exporting straight into the audio
    directory (via `storage.local_tempfile`) bypassed
    `FileSystemRepository.write_audio_file` — the one write path every
    audio-producing tool shares, and the place any write-time policy lives.
    The processor already reads its export back into memory to return bytes,
    so routing those bytes through the repository costs nothing extra.
    """
    tmp_dir = await anyio.to_thread.run_sync(lambda: tempfile.mkdtemp(prefix="sanzaru_audio_"))
    try:
        yield Path(tmp_dir) / Path(filename).name
    finally:
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))


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

        Raises:
        ------
            ValueError: If the input is not a decodable audio container, or the
                output name (given or derived) is not a safe audio filename.
                Raised before any I/O.
            AudioConversionError: If decoding or encoding fails.

        """
        require_audio_name(input_filename, "input file")
        output_name = output_filename or f"{Path(input_filename).stem}.{target_format}"
        # Check the name that is actually written, not the argument: guarding
        # only an explicit `output_filename` left the derived one unchecked.
        require_audio_name(output_name, "output file", is_output=True)
        storage = get_storage()

        async with (
            storage.local_path("audio", input_filename) as input_path,
            _scratch_export_path(output_name) as scratch,
        ):
            # Load audio from local path (pydub needs filesystem access)
            audio_data = await self.processor.load_audio_from_path(input_path)
            converted = await self.processor.convert_audio_format(
                audio_data=audio_data,
                target_format=target_format,
                output_path=scratch,
            )

        await FileSystemRepository(storage).write_audio_file(output_name, converted)
        return AudioProcessingResult(output_file=output_name)

    async def _copy(self, input_filename: str, output_filename: str) -> None:
        """Copy one audio file to another name within the audio path type.

        Reads the file and writes it back through
        `FileSystemRepository.write_audio_file`, so the copy is subject to the
        same write path as every other audio output. This used to be a
        `shutil.copyfile` between `local_path` and `local_tempfile`, which the
        local backend did without holding the file in memory; the branch only
        runs for files already under `max_mb`, so the footprint is bounded by
        the caller's own size budget (the Databricks backend round-tripped the
        bytes either way).
        """
        storage = get_storage()
        data = await storage.read("audio", input_filename)
        await FileSystemRepository(storage).write_audio_file(output_filename, data)

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

        Raises:
        ------
            ValueError: If the input is not a decodable audio container, or the
                output name (given or derived) is not a safe audio filename.
                Raised before any I/O.
            AudioConversionError: If a non-mp3 input fails to convert first.
            AudioCompressionError: If the mp3 re-encode fails.

        """
        require_audio_name(input_filename, "input file")
        if output_filename is not None:
            require_audio_name(output_filename, "output file", is_output=True)
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
        require_audio_name(output_name, "output file", is_output=True)

        logger.debug(f"Original file: {input_filename}")
        logger.debug(f"Output file: {output_name}")

        async with (
            storage.local_path("audio", input_filename) as input_path,
            _scratch_export_path(output_name) as scratch,
        ):
            # Load and compress via pydub (needs local filesystem)
            audio_data = await self.processor.load_audio_from_path(input_path)
            compressed = await self.processor.compress_mp3(audio_data, scratch)

        await FileSystemRepository(storage).write_audio_file(output_name, compressed)

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
