"""Test audio service orchestration layer."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from sanzaru.audio.models import AudioProcessingResult
from sanzaru.audio.processor import AudioProcessor
from sanzaru.audio.services.audio_service import AudioService
from sanzaru.storage.protocol import FileInfo

pytestmark = pytest.mark.audio


@asynccontextmanager
async def _fake_local_path(tmp_dir: Path, filename: str, content: bytes = b""):
    """Fake local_path that yields a real temp file with given content."""
    path = tmp_dir / filename
    if content:
        path.write_bytes(content)
    yield path


@asynccontextmanager
async def _fake_local_tempfile(tmp_dir: Path, filename: str):
    """Fake local_tempfile that yields a writable temp path."""
    path = tmp_dir / filename
    yield path


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
        """Create mock StorageBackend."""
        storage = MagicMock()
        return storage

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
        mock_storage.local_tempfile = lambda pt, fn: _fake_local_tempfile(audio_dir, fn)
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

    @pytest.mark.anyio
    async def test_convert_audio_with_custom_output_filename(
        self, service: AudioService, audio_dir: Path, mock_storage: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test conversion with custom output filename."""
        input_file = audio_dir / "source.wav"
        input_file.write_bytes(b"wav")

        mock_storage.local_path = lambda pt, fn: _fake_local_path(audio_dir, fn, b"wav")
        mock_storage.local_tempfile = lambda pt, fn: _fake_local_tempfile(audio_dir, fn)
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
        mock_storage.local_tempfile = lambda pt, fn: _fake_local_tempfile(audio_dir, fn)
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
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=mock_storage)

        result = await service.compress_audio(
            input_filename="small.mp3",
            output_filename="small.mp3",
            max_mb=25,
        )

        assert result.output_file == "small.mp3"

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
        mock_storage.local_tempfile = lambda pt, fn: _fake_local_tempfile(audio_dir, fn)
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
        mock_storage.local_tempfile = lambda pt, fn: _fake_local_tempfile(audio_dir, fn)
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
        mock_storage.local_tempfile = lambda pt, fn: _fake_local_tempfile(audio_dir, fn)
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
        storage = MagicMock()
        mocker.patch("sanzaru.audio.services.audio_service.get_storage", return_value=storage)

        with pytest.raises(ValueError, match="unsupported audio format"):
            await service.compress_audio("simrun_attacker.json", "simrun_victim.json", max_mb=1000)

        storage.stat.assert_not_called()

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


@pytest.mark.unit
class TestConversionStillAcceptsUnsupportedFormats:
    """`convert_audio` exists to turn formats the API cannot take into ones it
    can, so gating its *input* on the transcription allowlist broke the tool's
    entire purpose. Inputs need only be a container ffmpeg decodes without
    dereferencing paths out of its contents."""

    @pytest.mark.parametrize("name", ["clip.aac", "clip.opus", "clip.aiff", "clip.wma", "clip.m4b"])
    def test_formats_worth_converting_are_accepted_as_input(self, name):
        from sanzaru.audio.services.audio_service import _require_audio_name

        _require_audio_name(name, "input file")

    @pytest.mark.parametrize("name", ["evil.hls", "evil.concat", "evil.m3u8", "evil.dash"])
    def test_playlist_extensions_are_still_refused_as_input(self, name):
        from sanzaru.audio.services.audio_service import _require_audio_name

        with pytest.raises(ValueError, match="unsupported audio format"):
            _require_audio_name(name, "input file")

    @pytest.mark.parametrize("name", ["out.aac", "out.opus", "out.json", "out.html"])
    def test_outputs_stay_on_the_narrow_set(self, name):
        """An output is a file this server creates; no reason it can mint these."""
        from sanzaru.audio.services.audio_service import _require_audio_name

        with pytest.raises(ValueError, match="unsupported audio format"):
            _require_audio_name(name, "output file", is_output=True)
