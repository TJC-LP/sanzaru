# SPDX-License-Identifier: MIT
"""File system operations for audio file management.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import fnmatch
from collections.abc import Callable

import anyio
from openai.types import AudioModel
from pydub import AudioSegment  # type: ignore

from ..audio.constants import (
    AUDIO_CHAT_MODELS,
    CHAT_WITH_AUDIO_FORMATS,
    DECODABLE_AUDIO_EXTENSIONS,
    TRANSCRIBE_AUDIO_FORMATS,
    TRANSCRIPTION_MODELS,
    AudioChatModel,
)
from ..audio.models import FilePathSupportParams
from ..audio.processor import demuxer_for
from ..exceptions import AudioFileError, AudioFileNotFoundError
from ..storage import get_storage
from ..storage.protocol import FileInfo, StorageBackend

# Names are filtered by substring or fnmatch glob, never by a caller-supplied
# regex: a pattern like ``(a+)+$`` against a long filename made Python's ``re``
# engine backtrack indefinitely on the event loop, freezing the whole server
# (CWE-1333). fnmatch *is* regex-backed — ``fnmatch.translate`` compiles to
# ``re`` — but since Python 3.9 (bpo-40480) it emits ``*`` as an atomic group
# (``(?>.*?x)``), so a glob cannot backtrack exponentially. That translation,
# not "no regex", is the property this leans on; a hand-rolled ``re.compile`` of
# a translated glob would not have it.
_MAX_PATTERN_LEN = 256
_GLOB_METACHARACTERS = frozenset("*?[")
# Regex syntax with no glob meaning. This filter used to *be* a regex, so these
# still arrive from saved scripts and old tool descriptions; a pattern carrying
# one is refused rather than quietly matching nothing — an agent recovers from
# an error, not from a plausible empty list.
_REGEX_ONLY_METACHARACTERS = frozenset("\\^$()|+")


def compile_name_filter(pattern: str) -> Callable[[str], bool]:
    """Turn a caller-supplied name filter into a linear-time predicate.

    The pattern is a case-insensitive substring, or an fnmatch glob when it
    contains ``*``, ``?`` or ``[`` (a literal ``[`` is spelled ``[[]``).
    Raises ValueError for an over-long pattern — truncating one would turn a
    substring into a prefix and match *more* files than asked — and for
    regex-only syntax, so a caller working from the old regex contract learns
    the new one instead of receiving an empty result.
    """
    if len(pattern) > _MAX_PATTERN_LEN:
        raise ValueError(f"pattern is {len(pattern)} characters long; the limit is {_MAX_PATTERN_LEN}")
    stray = sorted(set(pattern) & _REGEX_ONLY_METACHARACTERS)
    if stray:
        raise ValueError(
            f"pattern {pattern!r:.80} uses regex syntax ({' '.join(stray)}), which is not supported: a pattern is a "
            "case-insensitive substring, or a glob using * ? [ ] (e.g. '*.mp3'). Stand in for such a character "
            "with ?, and match a literal [ as [[]"
        )
    lowered_pattern = pattern.lower()
    if _GLOB_METACHARACTERS & set(pattern):
        # fnmatchcase: both sides are lowercased here already, and fnmatch's
        # extra os.path.normcase would rewrite slashes on Windows.
        return lambda name: fnmatch.fnmatchcase(name.lower(), lowered_pattern)
    return lambda name: lowered_pattern in name.lower()


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

        # Get duration if possible (downloads file for remote backends). Only
        # decode allowlisted audio containers — the demuxer is chosen from the
        # untrusted extension, and a playlist demuxer (hls/concat/dash) would
        # read other local files (CWE-610). A non-audio extension simply yields
        # no duration rather than invoking ffmpeg. `demuxer_for` also maps the
        # extensions ffmpeg does not know by name (.opus → ogg) onto the demuxer
        # that reads them, so those files get a duration instead of a silent None.
        duration_seconds = None
        if audio_format in DECODABLE_AUDIO_EXTENSIONS:
            demuxer = demuxer_for(audio_format)
            try:
                async with self._storage.local_path("audio", filename) as local:
                    audio = await anyio.to_thread.run_sync(lambda: AudioSegment.from_file(str(local), format=demuxer))
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
            pattern: Optional case-insensitive filter — a substring, or a glob
                (e.g. ``*.mp3``) when it contains ``*``, ``?`` or ``[``. Not a
                regex: see :func:`compile_name_filter`.
            min_size_bytes: Minimum file size in bytes.
            max_size_bytes: Maximum file size in bytes.
            format_filter: Specific audio format to filter by (e.g., 'mp3', 'wav').

        Returns:
            list[FileInfo]: List of file info objects matching the criteria.

        Raises:
            ValueError: If `pattern` is over-long or uses regex syntax.
        """
        # Compiled once, before any I/O: a bad pattern is a usage error and
        # should not cost a directory listing to discover.
        matches = compile_name_filter(pattern) if pattern else None

        audio_extensions = TRANSCRIBE_AUDIO_FORMATS | CHAT_WITH_AUDIO_FORMATS
        file_infos = await self._storage.list_files("audio", extensions=audio_extensions)

        results: list[FileInfo] = []
        for info in file_infos:
            file_ext = ("." + info.name.rsplit(".", 1)[-1].lower()) if "." in info.name else ""

            # Apply pattern filtering if provided (linear-time; see compile_name_filter)
            if matches is not None and not matches(info.name):
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

    async def refuse_clobbering_a_checkpoint(self, filename: str) -> None:
        """Refuse a write that would land on a simulated-podcast act checkpoint.

        Public because the podcast tools also call it *pre-flight*, before any
        synthesis is billed: the copy inside `write_audio_file` is what actually
        protects the checkpoint, but it fires only at the final write — after
        the whole episode has been rendered and paid for, and the scripted path
        has no checkpoints of its own to recover that spend from.

        Checked by *looking*, not by matching the name. The name shape
        (`<slug>_<runid>_<actid>.mp3`) cannot be recognised reliably: eight hex
        characters is also what a date looks like, so a pattern strict enough to
        catch checkpoints also rejected `interview_20250826_part1.mp3`. Asking
        whether an act sidecar actually sits beside the target has no such
        ambiguity.

        Worth the extra stat because the alternative is silent: every
        audio-producing tool writes into one flat directory under a caller-chosen
        name and the storage write is an unconditional truncate, so on a shared
        deployment this was how one session replaced another's paid-for act
        audio — and, when the replacement decoded, how attacker-chosen audio got
        into someone else's finished episode (CWE-73).
        """
        stem, _, suffix = filename.rpartition(".")
        if suffix.lower() not in ("mp3", "wav") or not stem:
            return
        sidecar = f"{stem}.json"
        try:
            if not await self._storage.exists("audio", sidecar):
                return
            head = (await self._storage.read("audio", sidecar))[:2048]
        except Exception:  # noqa: BLE001 - a failed probe must not block a legitimate write
            return
        if b'"act_id"' in head and b'"turns"' in head:
            raise AudioFileError(
                f"'{filename}' is the audio of a recorded act belonging to another run "
                f"(its checkpoint sidecar '{sidecar}' is present) — refusing to overwrite it; "
                "choose another output name"
            )

    async def write_audio_file(self, filename: str, content: bytes, *, is_bookkeeping: bool = False) -> str:
        """Write audio content to a file asynchronously.

        Args:
            filename: Name of the file to write.
            content: Audio content as bytes.
            is_bookkeeping: True when the caller *is* the run-checkpoint writer,
                which is the one path allowed to write over its own act audio
                (``--qc-retry`` re-records an act that is already on disk).

        Returns:
            str: Display path of the written file.

        Raises:
            AudioFileError: If there's an error writing the file.
        """
        if not is_bookkeeping:
            await self.refuse_clobbering_a_checkpoint(filename)
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
