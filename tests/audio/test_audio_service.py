"""Test audio service orchestration layer."""

import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from sanzaru.audio.models import AudioProcessingResult
from sanzaru.audio.processor import AudioProcessor, demuxer_for
from sanzaru.audio.services.audio_service import AudioService, require_audio_name
from sanzaru.infrastructure import FileSystemRepository
from sanzaru.storage.protocol import FileInfo

pytestmark = pytest.mark.audio


@asynccontextmanager
async def _fake_local_path(tmp_dir: Path, filename: str, content: bytes = b""):
    """Fake local_path that yields a real temp file with given content."""
    path = tmp_dir / filename
    if content:
        path.write_bytes(content)
    yield path


def _storage_over(audio_dir: Path) -> MagicMock:
    """A mock StorageBackend whose read/write really hit `audio_dir`.

    `local_tempfile` is deliberately an assertion: convert/compress must write
    through `FileSystemRepository.write_audio_file` → `storage.write`, never by
    handing pydub or shutil a destination path of their own.
    """
    storage = MagicMock()

    async def _read(path_type: str, filename: str) -> bytes:
        return (audio_dir / filename).read_bytes()

    async def _write(path_type: str, filename: str, data: bytes) -> str:
        (audio_dir / filename).write_bytes(data)
        return str(audio_dir / filename)

    storage.read = AsyncMock(side_effect=_read)
    storage.write = AsyncMock(side_effect=_write)
    storage.local_tempfile = MagicMock(side_effect=AssertionError("outputs must go through write_audio_file"))
    return storage


class TestAudioService:
    """Test suite for AudioService."""

    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        """Create temporary audio directory."""
        audio_path = tmp_path / "audio"
        audio_path.mkdir()
        return audio_path

    @pytest.fixture
    def mock_storage(self, audio_dir: Path) -> MagicMock:
        """Create mock StorageBackend backed by `audio_dir`."""
        return _storage_over(audio_dir)

    @pytest.fixture
    def service(self) -> AudioService:
        """Create AudioService instance."""
        return AudioService()

    @pytest.mark.anyio
    async def test_convert_audio_success(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test successful audio conversion."""
        # Create input file
        input_file = audio_dir / "input.wav"
        input_file.write_bytes(b"wav data")

        # Mock get_storage
        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"wav data")
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        # Mock AudioProcessor methods
        mock_audio_data = MagicMock()
        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_convert = mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock)
        mock_load.return_value = mock_audio_data
        mock_convert.return_value = b"mp3 data"

        result = await service.convert_audio(
            input_filename="input.wav",
            target_format="mp3",
        )

        assert isinstance(result, AudioProcessingResult)
        assert result.output_file == "input.mp3"
        mock_load.assert_called_once()
        mock_convert.assert_called_once()
        # The processor's bytes land through storage.write, under the derived name.
        mock_storage.write.assert_awaited_once_with("audio", "input.mp3", b"mp3 data")
        assert (audio_dir / "input.mp3").read_bytes() == b"mp3 data"

    @pytest.mark.anyio
    async def test_convert_audio_with_custom_output_filename(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test conversion with custom output filename."""
        input_file = audio_dir / "source.wav"
        input_file.write_bytes(b"wav")

        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"wav")
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        mock_audio_data = MagicMock()
        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_convert = mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock)
        mock_load.return_value = mock_audio_data
        mock_convert.return_value = b"mp3"

        result = await service.convert_audio(
            input_filename="source.wav",
            output_filename="custom_output.mp3",
            target_format="mp3",
        )

        assert result.output_file == "custom_output.mp3"

    @pytest.mark.anyio
    async def test_compress_audio_below_threshold(
        self, service: AudioService, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test that files below threshold are not compressed."""
        file_size = 10 * 1024 * 1024  # 10 MB (below 25MB threshold)
        mock_storage.stat = AsyncMock(
            return_value=FileInfo(name="small.mp3", size_bytes=file_size, modified_timestamp=0)
        )
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        result = await service.compress_audio(
            input_filename="small.mp3",
            max_mb=25,
        )

        # Should return original filename without compression
        assert result.output_file == "small.mp3"

    @pytest.mark.anyio
    async def test_below_threshold_with_an_output_name_copies_rather_than_no_ops(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """A small file plus `output_filename` must still produce that file (#59).

        Returning the *input's* name here is what let the CLI's cross-directory
        branch `shutil.move` the source away — the caller ran compress
        defensively and lost the original.
        """
        (audio_dir / "small.mp3").write_bytes(b"already small")
        mock_storage.stat = AsyncMock(
            return_value=FileInfo(name="small.mp3", size_bytes=10 * 1024 * 1024, modified_timestamp=0)
        )
        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"already small")
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)
        compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)

        result = await service.compress_audio(
            input_filename="small.mp3",
            output_filename="wanted.mp3",
            max_mb=25,
        )

        assert result.output_file == "wanted.mp3"
        assert (audio_dir / "wanted.mp3").read_bytes() == b"already small"
        assert (audio_dir / "small.mp3").exists(), "the input must survive"
        compress.assert_not_called()  # copied, not re-encoded

    @pytest.mark.anyio
    async def test_below_threshold_with_the_same_output_name_stays_a_no_op(
        self, service: AudioService, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Asking for the name it already has must not copy a file onto itself."""
        mock_storage.stat = AsyncMock(
            return_value=FileInfo(name="small.mp3", size_bytes=10 * 1024 * 1024, modified_timestamp=0)
        )
        mock_storage.local_path = MagicMock(side_effect=AssertionError("should not open the file"))
        mock_storage.read = AsyncMock(side_effect=AssertionError("should not read the file"))
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        result = await service.compress_audio(
            input_filename="small.mp3",
            output_filename="small.mp3",
            max_mb=25,
        )

        assert result.output_file == "small.mp3"
        mock_storage.write.assert_not_awaited()

    @pytest.mark.anyio
    async def test_compress_audio_above_threshold(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test that files above threshold are compressed."""
        input_file = audio_dir / "large.mp3"
        input_file.write_bytes(b"large file")

        file_size = 30 * 1024 * 1024  # 30 MB (above 25MB threshold)
        compressed_size = 20 * 1024 * 1024

        mock_storage.stat = AsyncMock(
            side_effect=[
                FileInfo(name="large.mp3", size_bytes=file_size, modified_timestamp=0),
                FileInfo(name="compressed_large.mp3", size_bytes=compressed_size, modified_timestamp=0),
            ]
        )
        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"large file")
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        mock_audio_data = MagicMock()
        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)
        mock_load.return_value = mock_audio_data
        mock_compress.return_value = b"compressed data"

        result = await service.compress_audio(
            input_filename="large.mp3",
            max_mb=25,
        )

        assert result.output_file == "compressed_large.mp3"
        mock_compress.assert_called_once()
        mock_storage.write.assert_awaited_once_with("audio", "compressed_large.mp3", b"compressed data")

    @pytest.mark.anyio
    async def test_compress_audio_non_mp3_converts_first(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test that non-MP3 files are converted to MP3 before compression."""
        input_file = audio_dir / "large.wav"
        input_file.write_bytes(b"large wav")

        file_size = 30 * 1024 * 1024  # Large file
        compressed_size = 20 * 1024 * 1024

        mock_storage.stat = AsyncMock(
            side_effect=[
                FileInfo(name="large.wav", size_bytes=file_size, modified_timestamp=0),
                FileInfo(name="compressed_large.mp3", size_bytes=compressed_size, modified_timestamp=0),
            ]
        )
        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"audio data")
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        mock_audio_data = MagicMock()
        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_convert = mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock)
        mock_compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)
        mock_load.return_value = mock_audio_data
        mock_convert.return_value = b"converted mp3"
        mock_compress.return_value = b"compressed"

        await service.compress_audio(
            input_filename="large.wav",
            max_mb=25,
        )

        # Should have converted to MP3 first
        assert mock_convert.call_count >= 1
        # Then compressed
        assert mock_compress.call_count >= 1

    @pytest.mark.anyio
    async def test_compress_audio_with_custom_output_filename(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test compression with custom output filename."""
        input_file = audio_dir / "large.mp3"
        input_file.write_bytes(b"large")

        file_size = 30 * 1024 * 1024
        compressed_size = 20 * 1024 * 1024

        mock_storage.stat = AsyncMock(
            side_effect=[
                FileInfo(name="large.mp3", size_bytes=file_size, modified_timestamp=0),
                FileInfo(name="custom_compressed.mp3", size_bytes=compressed_size, modified_timestamp=0),
            ]
        )
        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"large")
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        mock_audio_data = MagicMock()
        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)
        mock_load.return_value = mock_audio_data
        mock_compress.return_value = b"compressed"

        result = await service.compress_audio(
            input_filename="large.mp3",
            output_filename="custom_compressed.mp3",
            max_mb=25,
        )

        assert result.output_file == "custom_compressed.mp3"

    @pytest.mark.anyio
    async def test_maybe_compress_file_delegates_to_compress_audio(
        self, service: AudioService, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test that maybe_compress_file delegates to compress_audio."""
        file_size = 10 * 1024 * 1024  # Small file
        mock_storage.stat = AsyncMock(
            return_value=FileInfo(name="test.mp3", size_bytes=file_size, modified_timestamp=0)
        )
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        result = await service.maybe_compress_file("test.mp3", max_mb=25)

        # Should not compress (below threshold)
        assert result.output_file == "test.mp3"


@pytest.mark.unit
class TestAudioPathsAreNotAGenericFileGadget:
    """convert/compress must not become a way to move arbitrary bytes around.

    The no-compression branch is a plain `shutil.copyfile` — it never decodes
    the input as audio — so without an extension rule it copied ANY file in the
    audio directory to ANY caller-chosen name. That is two separate primitives:
    replacing another run's `simrun_*.json` bookkeeping (CWE-73) and republishing
    stored text under a browser-executable `.html` name that `/media` would then
    serve from the server's own origin (CWE-79).
    """

    @pytest.fixture
    def service(self) -> AudioService:
        return AudioService()

    async def test_compress_refuses_to_write_a_run_manifest(self, service, mocker: MockerFixture):
        """A valid input, so it is the *output* rule that fires — the role prefix proves which."""
        storage = MagicMock()
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=storage)

        with pytest.raises(ValueError, match=r"output file 'simrun_victim.json'"):
            await service.compress_audio("episode.mp3", "simrun_victim.json", max_mb=1000)

        storage.stat.assert_not_called()

    def test_a_checkpoint_shaped_name_is_not_refused_by_name(self):
        """Pins what the docstring now says instead of what it used to claim.

        Act checkpoints are ordinary .mp3 names; `reject_reserved_name` matches
        only run manifests, on purpose (eight hex characters is also a date).
        Their protection belongs to the write path, not to this check — so if
        this test starts failing, either the name rule grew a false positive
        for `interview_20250826_part1.mp3`-style names, or the docstring is
        wrong again.
        """
        require_audio_name("show_deadbeef_act1.mp3", "output file", is_output=True)
        require_audio_name("interview_20250826_part1.mp3", "output file", is_output=True)

    @pytest.mark.parametrize("name", ["mp3", ".mp3", "wav", ""])
    def test_a_bare_suffix_is_not_a_filename(self, name):
        """`safe_audio_format` accepts "mp3" as a *format*; as a filename it would write a file called mp3."""
        with pytest.raises(ValueError, match="has no audio extension"):
            require_audio_name(name, "output file", is_output=True)

    async def test_compress_refuses_a_browser_executable_output_name(self, service, mocker: MockerFixture):
        storage = MagicMock()
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=storage)

        with pytest.raises(ValueError, match="unsupported audio format"):
            await service.compress_audio("episode.mp3", "x.html")

        storage.stat.assert_not_called()

    async def test_convert_refuses_a_non_audio_output_name(self, service, mocker: MockerFixture):
        storage = MagicMock()
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=storage)

        with pytest.raises(ValueError, match="unsupported audio format"):
            await service.convert_audio("episode.mp3", "notes.json")


@pytest.mark.unit
class TestFfmpegDemuxerIsNotChosenByTheFile:
    """pydub picks the ffmpeg demuxer from `format=<ext>`.

    Deriving that from an untrusted filename made playlist demuxers reachable:
    a file named `evil.hls` is parsed by ffmpeg's HLS demuxer, whose *content*
    then names other local files to decode and concatenate into the result —
    reading straight past the storage layer's directory confinement (CWE-610).
    """

    @pytest.mark.parametrize("name", ["evil.hls", "evil.concat", "evil.dash", "evil.m3u8", "evil.txt"])
    async def test_a_playlist_extension_never_reaches_ffmpeg(self, tmp_path: Path, mocker: MockerFixture, name):
        planted = tmp_path / name
        planted.write_text("#EXTM3U\n#EXTINF:600,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n")
        from_file = mocker.patch("sanzaru.audio.processor.AudioSegment.from_file")

        with pytest.raises(ValueError, match="unsupported audio format"):
            await AudioProcessor.load_audio_from_path(planted)

        from_file.assert_not_called()

    @pytest.mark.parametrize("name", ["clip.mp3", "clip.wav", "clip.m4a", "clip.flac", "clip.MP3"])
    async def test_real_audio_still_decodes(self, tmp_path: Path, mocker: MockerFixture, name):
        path = tmp_path / name
        path.write_bytes(b"data")
        from_file = mocker.patch("sanzaru.audio.processor.AudioSegment.from_file")

        await AudioProcessor.load_audio_from_path(path)

        assert from_file.call_args.kwargs["format"] == name.rsplit(".", 1)[1].lower()

    @pytest.mark.parametrize(
        ("name", "demuxer"),
        [
            ("clip.opus", "ogg"),
            ("clip.oga", "ogg"),
            ("clip.m4b", "mov"),
            ("clip.mka", "matroska"),
            ("clip.wma", "asf"),
            ("clip.aif", "aiff"),
            ("clip.aifc", "aiff"),
            ("clip.mpga", "mp3"),
            ("clip.OPUS", "ogg"),
        ],
    )
    async def test_extensions_ffmpeg_does_not_know_are_mapped_to_their_demuxer(
        self, tmp_path: Path, mocker: MockerFixture, name, demuxer
    ):
        """pydub passes `format` through as `ffmpeg -f <format>`; `-f opus` is "Unknown format".

        Without this map the allowlist *admitted* these files and then every
        one of them failed to decode — the ".opus still converts" claim was
        only ever true under a mock.
        """
        path = tmp_path / name
        path.write_bytes(b"data")
        from_file = mocker.patch("sanzaru.audio.processor.AudioSegment.from_file")

        await AudioProcessor.load_audio_from_path(path)

        assert from_file.call_args.kwargs["format"] == demuxer
        assert demuxer_for(name) == demuxer

    @pytest.mark.parametrize("name", ["evil.hls", "evil.m3u8", "evil.concat"])
    def test_the_alias_table_cannot_admit_a_playlist(self, name):
        with pytest.raises(ValueError, match="unsupported audio format"):
            demuxer_for(name)


@pytest.mark.unit
class TestCheckpointAudioIsNotClobberable:
    """One flat directory, caller-chosen names, and a truncating write.

    Recognising a checkpoint by *name* cannot be done without false positives
    (eight hex characters is also a date), so the guard looks for the act
    sidecar sitting beside the target instead.
    """

    @pytest.fixture
    def repo(self, tmp_path: Path, mocker: MockerFixture):
        from sanzaru.infrastructure import FileSystemRepository
        from sanzaru.storage.local import LocalStorageBackend

        media = (tmp_path / "audio").resolve()
        media.mkdir()
        mocker.patch(
            "sanzaru.infrastructure.file_system.get_storage",
            return_value=LocalStorageBackend(path_overrides={"audio": media}),
        )
        return FileSystemRepository(), media

    @staticmethod
    def _plant_checkpoint(media: Path, stem: str) -> None:
        import json

        (media / f"{stem}.mp3").write_bytes(b"VICTIM-PAID-AUDIO")
        (media / f"{stem}.json").write_text(
            json.dumps(
                {
                    "act_id": "act1",
                    "title": "Act 1",
                    "stop_reason": "complete",
                    "usage": {"output_audio_tokens": 10},
                    "turns": [],
                }
            )
        )

    async def test_another_run_cannot_overwrite_act_audio(self, repo):
        from sanzaru.exceptions import AudioFileError

        file_repo, media = repo
        self._plant_checkpoint(media, "Show_a1b2c3d4_act1")

        with pytest.raises(AudioFileError, match="refusing to overwrite"):
            await file_repo.write_audio_file("Show_a1b2c3d4_act1.mp3", b"ATTACKER")

        assert (media / "Show_a1b2c3d4_act1.mp3").read_bytes() == b"VICTIM-PAID-AUDIO"

    async def test_the_run_may_still_re_record_its_own_act(self, repo):
        """--qc-retry re-records an act that is already on disk."""
        file_repo, media = repo
        self._plant_checkpoint(media, "Show_a1b2c3d4_act1")

        await file_repo.write_audio_file("Show_a1b2c3d4_act1.mp3", b"RETAKE", is_bookkeeping=True)

        assert (media / "Show_a1b2c3d4_act1.mp3").read_bytes() == b"RETAKE"

    async def test_a_date_stamped_recording_is_not_protected(self, repo):
        """The false positive a name-shaped rule produced: YYYYMMDD is 8 hex digits."""
        file_repo, media = repo
        (media / "interview_20250826_part1.mp3").write_bytes(b"old")
        (media / "interview_20250826_part1.json").write_text('{"notes": "my own metadata"}')

        await file_repo.write_audio_file("interview_20250826_part1.mp3", b"new take")

        assert (media / "interview_20250826_part1.mp3").read_bytes() == b"new take"

    async def test_an_ordinary_new_write_is_untouched(self, repo):
        file_repo, media = repo
        await file_repo.write_audio_file("brand_new.mp3", b"fresh")
        assert (media / "brand_new.mp3").read_bytes() == b"fresh"

    async def test_generate_podcast_refuses_the_name_before_any_synthesis(self, repo, mocker):
        """The copy of this check inside `write_audio_file` fires only after
        the whole episode has been rendered and billed — and the scripted path
        has no checkpoints to recover that spend from. The pre-flight copy in
        `generate_podcast` must refuse before a single TTS request goes out."""
        from sanzaru.exceptions import AudioFileError
        from sanzaru.tools.podcast import generate_podcast

        _, media = repo
        self._plant_checkpoint(media, "Show_a1b2c3d4_act1")
        synth = mocker.patch("sanzaru.tools.podcast.synthesize_speech", new_callable=mocker.AsyncMock)

        script = {
            "title": "My Episode",
            "speakers": [{"id": "a", "name": "Alex", "voice": "ash"}],
            "segments": [{"speaker": "a", "text": "Hello."}],
        }
        with pytest.raises(AudioFileError, match="refusing to overwrite"):
            await generate_podcast(script, filename="Show_a1b2c3d4_act1.mp3")

        synth.assert_not_called()
        assert (media / "Show_a1b2c3d4_act1.mp3").read_bytes() == b"VICTIM-PAID-AUDIO"


@pytest.mark.unit
class TestConversionStillAcceptsUnsupportedFormats:
    """`convert_audio` exists to turn formats the API cannot take into ones it
    can, so gating its *input* on the transcription allowlist broke the tool's
    entire purpose. Inputs need only be a container ffmpeg decodes without
    dereferencing paths out of its contents."""

    @pytest.mark.parametrize("name", ["clip.aac", "clip.opus", "clip.aiff", "clip.wma", "clip.m4b"])
    def test_formats_worth_converting_are_accepted_as_input(self, name):
        require_audio_name(name, "input file")

    @pytest.mark.parametrize("name", ["evil.hls", "evil.concat", "evil.m3u8", "evil.dash"])
    def test_playlist_extensions_are_still_refused_as_input(self, name):
        with pytest.raises(ValueError, match="unsupported audio format"):
            require_audio_name(name, "input file")

    @pytest.mark.parametrize("name", ["out.aac", "out.opus", "out.json", "out.html"])
    def test_outputs_stay_on_the_narrow_set(self, name):
        """An output is a file this server creates; no reason it can mint these."""
        with pytest.raises(ValueError, match="unsupported audio format"):
            require_audio_name(name, "output file", is_output=True)


@pytest.mark.integration
class TestAudioOutputsShareOneWritePath:
    """Every convert/compress output goes through `FileSystemRepository.write_audio_file`.

    pydub used to export straight into `storage.local_tempfile`, and the no-op
    copy was a `shutil.copyfile` — two writes into the audio directory that no
    repository-level policy could see. The checkpoint-overwrite refusal lives
    on `write_audio_file`, so a write that bypasses it is unprotected however
    careful the name check is. These pin the *route*, not the guard.
    """

    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        audio_path = tmp_path / "audio"
        audio_path.mkdir()
        return audio_path

    @pytest.fixture
    def storage(self, audio_dir: Path, mocker: MockerFixture) -> MagicMock:
        storage = _storage_over(audio_dir)
        storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn)
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=storage)
        return storage

    async def test_convert_writes_through_the_repository(self, audio_dir: Path, storage, mocker: MockerFixture):
        (audio_dir / "raw.wav").write_bytes(b"wav")
        mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock, return_value=b"mp3!")
        write = mocker.patch.object(FileSystemRepository, "write_audio_file", new_callable=AsyncMock)

        await AudioService().convert_audio("raw.wav", "final.mp3")

        write.assert_awaited_once_with("final.mp3", b"mp3!")
        storage.local_tempfile.assert_not_called()

    async def test_compress_reencode_writes_through_the_repository(
        self, audio_dir: Path, storage, mocker: MockerFixture
    ):
        (audio_dir / "large.mp3").write_bytes(b"big")
        storage.stat = AsyncMock(
            side_effect=[
                FileInfo(name="large.mp3", size_bytes=30 * 1024 * 1024, modified_timestamp=0),
                FileInfo(name="small.mp3", size_bytes=1, modified_timestamp=0),
            ]
        )
        mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock, return_value=b"smaller")
        write = mocker.patch.object(FileSystemRepository, "write_audio_file", new_callable=AsyncMock)

        await AudioService().compress_audio("large.mp3", "small.mp3", max_mb=25)

        write.assert_awaited_once_with("small.mp3", b"smaller")
        storage.local_tempfile.assert_not_called()

    async def test_compress_copy_writes_through_the_repository(self, audio_dir: Path, storage, mocker: MockerFixture):
        """The branch that never decodes — the one that made this a generic file gadget."""
        (audio_dir / "small.mp3").write_bytes(b"already small")
        storage.stat = AsyncMock(return_value=FileInfo(name="small.mp3", size_bytes=10, modified_timestamp=0))
        write = mocker.patch.object(FileSystemRepository, "write_audio_file", new_callable=AsyncMock)

        await AudioService().compress_audio("small.mp3", "copy.mp3", max_mb=25)

        write.assert_awaited_once_with("copy.mp3", b"already small")
        storage.local_tempfile.assert_not_called()

    async def test_the_scratch_export_is_cleaned_up(self, audio_dir: Path, storage, mocker: MockerFixture):
        """pydub exports into a private temp dir, not the audio dir; nothing of it survives."""
        (audio_dir / "raw.wav").write_bytes(b"wav")
        seen: list[Path] = []

        async def fake_convert(audio_data, target_format, output_path):
            seen.append(output_path)
            output_path.write_bytes(b"mp3!")
            return b"mp3!"

        mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mocker.patch.object(AudioProcessor, "convert_audio_format", side_effect=fake_convert)

        await AudioService().convert_audio("raw.wav", "final.mp3")

        assert len(seen) == 1
        assert audio_dir not in seen[0].parents
        assert not seen[0].parent.exists()
        assert (audio_dir / "final.mp3").read_bytes() == b"mp3!"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="exercises the real ffmpeg demuxer table")
class TestAliasedContainersReallyDecode:
    """Unmocked: the demuxer names in `AUDIO_DEMUXER_BY_EXTENSION` must be ones ffmpeg accepts.

    `.mpga` is absent because ffmpeg has a demuxer alias for it but no muxer,
    so a fixture cannot be generated by extension.
    """

    @pytest.mark.parametrize("ext", ["opus", "oga", "m4b", "mka", "wma", "aif", "aifc"])
    async def test_a_generated_clip_decodes(self, tmp_path: Path, ext):
        path = tmp_path / f"clip.{ext}"
        started = time.monotonic()
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.3",
            ]
            + [str(path)],
            check=True,
            timeout=60,
        )

        audio = await AudioProcessor.load_audio_from_path(path)

        assert 250 <= len(audio) <= 400, f"{ext}: decoded {len(audio)}ms of a 300ms clip"
        assert time.monotonic() - started < 30
