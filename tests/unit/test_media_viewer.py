# SPDX-License-Identifier: MIT
"""Unit tests for media viewer tool functions."""

import base64

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.media_viewer import (
    DEFAULT_CHUNK_SIZE,
    MEDIA_TYPE_TO_PATH_TYPE,
    _guess_mime_type,
    _resolve_path_type,
    get_media_data,
    view_media,
)


@pytest.mark.unit
class TestMediaTypeMapping:
    """Test MEDIA_TYPE_TO_PATH_TYPE mapping."""

    def test_video_maps_to_video(self):
        assert MEDIA_TYPE_TO_PATH_TYPE["video"] == "video"

    def test_image_maps_to_reference(self):
        assert MEDIA_TYPE_TO_PATH_TYPE["image"] == "reference"

    def test_audio_maps_to_audio(self):
        assert MEDIA_TYPE_TO_PATH_TYPE["audio"] == "audio"


@pytest.mark.unit
class TestResolvePathType:
    """Test _resolve_path_type helper."""

    def test_valid_types(self):
        assert _resolve_path_type("video") == "video"
        assert _resolve_path_type("image") == "reference"
        assert _resolve_path_type("audio") == "audio"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown media_type"):
            _resolve_path_type("unknown")


@pytest.mark.unit
class TestGuessMimeType:
    """Test MIME type guessing with fallbacks."""

    def test_mp4(self):
        assert _guess_mime_type("video.mp4", "video") == "video/mp4"

    def test_png(self):
        assert _guess_mime_type("image.png", "image") == "image/png"

    def test_jpg(self):
        assert _guess_mime_type("photo.jpg", "image") == "image/jpeg"

    def test_mp3(self):
        assert _guess_mime_type("song.mp3", "audio") == "audio/mpeg"

    def test_wav(self):
        assert _guess_mime_type("sound.wav", "audio") == "audio/x-wav"

    def test_webp(self):
        assert _guess_mime_type("img.webp", "image") == "image/webp"

    def test_unknown_extension_falls_back(self):
        result = _guess_mime_type("file.zzzzz", "video")
        assert result == "video/mp4"

    def test_no_extension_falls_back(self):
        result = _guess_mime_type("noext", "audio")
        assert result == "audio/mpeg"


@pytest.mark.unit
class TestViewMedia:
    """Test view_media function."""

    async def test_valid_file(self, mocker, tmp_video_path):
        """view_media returns correct metadata for an existing file."""
        # Create a test file
        test_file = tmp_video_path / "test.mp4"
        test_file.write_bytes(b"x" * 1024)

        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        result = await view_media("video", "test.mp4")

        assert result["filename"] == "test.mp4"
        assert result["media_type"] == "video"
        assert result["size_bytes"] == 1024
        assert result["mime_type"] == "video/mp4"

    async def test_missing_file_raises(self, mocker, tmp_video_path):
        """view_media raises ValueError for non-existent file."""
        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        with pytest.raises(ValueError, match="File not found"):
            await view_media("video", "nonexistent.mp4")

    async def test_image_file(self, mocker, tmp_reference_path):
        """view_media works for images (maps to 'reference' path type)."""
        test_file = tmp_reference_path / "photo.png"
        test_file.write_bytes(b"PNG" + b"\x00" * 100)

        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        result = await view_media("image", "photo.png")

        assert result["filename"] == "photo.png"
        assert result["media_type"] == "image"
        assert result["mime_type"] == "image/png"

    async def test_audio_file(self, mocker, tmp_audio_path):
        """view_media works for audio files."""
        test_file = tmp_audio_path / "track.mp3"
        test_file.write_bytes(b"\xff\xfb\x90" + b"\x00" * 200)

        storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        result = await view_media("audio", "track.mp3")

        assert result["filename"] == "track.mp3"
        assert result["media_type"] == "audio"
        assert result["mime_type"] == "audio/mpeg"


@pytest.mark.unit
class TestGetMediaData:
    """Test get_media_data function."""

    async def test_small_file_single_chunk(self, mocker, tmp_video_path):
        """Small file fits in one chunk with is_last=True."""
        content = b"hello world"
        test_file = tmp_video_path / "small.mp4"
        test_file.write_bytes(content)

        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        result = await get_media_data("video", "small.mp4")

        assert result["is_last"] is True
        assert result["offset"] == 0
        assert result["chunk_size"] == len(content)
        assert result["total_size"] == len(content)
        assert result["mime_type"] == "video/mp4"
        assert base64.b64decode(result["data"]) == content

    async def test_chunked_reading(self, mocker, tmp_video_path):
        """Large file is read in chunks with correct offset math."""
        content = b"A" * 100
        test_file = tmp_video_path / "chunked.mp4"
        test_file.write_bytes(content)

        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        # First chunk: 40 bytes
        r1 = await get_media_data("video", "chunked.mp4", offset=0, chunk_size=40)
        assert r1["chunk_size"] == 40
        assert r1["is_last"] is False
        assert r1["total_size"] == 100
        assert len(base64.b64decode(r1["data"])) == 40

        # Second chunk: 40 bytes
        r2 = await get_media_data("video", "chunked.mp4", offset=40, chunk_size=40)
        assert r2["chunk_size"] == 40
        assert r2["is_last"] is False
        assert r2["offset"] == 40

        # Third chunk: remaining 20 bytes
        r3 = await get_media_data("video", "chunked.mp4", offset=80, chunk_size=40)
        assert r3["chunk_size"] == 20
        assert r3["is_last"] is True

    async def test_offset_past_end(self, mocker, tmp_video_path):
        """Offset past file end returns empty chunk with is_last=True."""
        test_file = tmp_video_path / "short.mp4"
        test_file.write_bytes(b"abc")

        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        result = await get_media_data("video", "short.mp4", offset=999)

        assert result["chunk_size"] == 0
        assert result["is_last"] is True
        assert result["data"] == ""  # empty base64

    async def test_negative_offset_raises(self, mocker, tmp_video_path):
        """Negative offset raises ValueError."""
        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

        with pytest.raises(ValueError, match="offset must be non-negative"):
            await get_media_data("video", "any.mp4", offset=-1)

    async def test_default_chunk_size(self):
        """Default chunk size is 2 MB."""
        assert DEFAULT_CHUNK_SIZE == 2 * 1024 * 1024
