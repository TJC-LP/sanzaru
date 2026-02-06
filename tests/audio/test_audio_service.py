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
