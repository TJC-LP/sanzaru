# SPDX-License-Identifier: MIT
"""Unit tests for configuration management."""

import os

import pytest

from sanzaru.config import get_path


@pytest.fixture(autouse=True)
def clear_path_cache():
    """Clear get_path() cache before each test to ensure isolation."""
    get_path.cache_clear()
    yield
    get_path.cache_clear()


@pytest.mark.unit
class TestGetPathHappyPath:
    """Test get_path() with valid configurations."""

    def test_get_path_video_valid(self, mocker, tmp_video_path):
        """Test that valid VIDEO_PATH returns correct path."""
        mocker.patch.dict(os.environ, {"VIDEO_PATH": str(tmp_video_path)})

        result = get_path("video")

        assert result == tmp_video_path.resolve()
        assert result.is_dir()

    def test_get_path_reference_valid(self, mocker, tmp_reference_path):
        """Test that valid IMAGE_PATH returns correct path."""
        mocker.patch.dict(os.environ, {"IMAGE_PATH": str(tmp_reference_path)})

        result = get_path("reference")

        assert result == tmp_reference_path.resolve()
        assert result.is_dir()

    def test_get_path_audio_valid(self, mocker, tmp_audio_path):
        """Test that valid AUDIO_PATH returns correct path."""
        mocker.patch.dict(os.environ, {"AUDIO_PATH": str(tmp_audio_path)})

        result = get_path("audio")

        assert result == tmp_audio_path.resolve()
        assert result.is_dir()


@pytest.mark.unit
class TestGetPathCaching:
    """Test get_path() caching behavior."""

    def test_caching_prevents_revalidation(self, mocker, tmp_video_path):
        """Test that second call uses cached result without re-validation."""
        mocker.patch.dict(os.environ, {"VIDEO_PATH": str(tmp_video_path)})

        # First call - validates and caches
        result1 = get_path("video")
        assert result1.exists()

        # Delete directory to prove second call doesn't re-validate
        tmp_video_path.rmdir()

        # Second call - returns cached result (doesn't fail even though dir deleted)
        result2 = get_path("video")
        assert result1 == result2

    def test_cache_separate_for_video_and_reference(self, mocker, tmp_video_path, tmp_reference_path):
        """Test that video and reference paths are cached separately."""
        mocker.patch.dict(
            os.environ,
            {
                "VIDEO_PATH": str(tmp_video_path),
                "IMAGE_PATH": str(tmp_reference_path),
            },
        )

        result_video = get_path("video")
        result_reference = get_path("reference")

        assert result_video != result_reference
        assert result_video == tmp_video_path.resolve()
        assert result_reference == tmp_reference_path.resolve()

    def test_cache_separate_for_all_path_types(self, mocker, tmp_video_path, tmp_reference_path, tmp_audio_path):
        """Test that video, reference, and audio paths are cached separately."""
        mocker.patch.dict(
            os.environ,
            {
                "VIDEO_PATH": str(tmp_video_path),
                "IMAGE_PATH": str(tmp_reference_path),
                "AUDIO_PATH": str(tmp_audio_path),
            },
        )

        result_video = get_path("video")
        result_reference = get_path("reference")
        result_audio = get_path("audio")

        # All paths should be different
        assert result_video != result_reference
        assert result_video != result_audio
        assert result_reference != result_audio

        # Each should match its expected path
        assert result_video == tmp_video_path.resolve()
        assert result_reference == tmp_reference_path.resolve()
        assert result_audio == tmp_audio_path.resolve()


@pytest.mark.unit
class TestGetPathErrorCases:
    """Test get_path() error handling."""

    def test_missing_video_env_var(self, mocker):
        """Test that missing VIDEO_PATH raises RuntimeError."""
        mocker.patch.dict(os.environ, {}, clear=True)

        with pytest.raises(RuntimeError, match="not configured.*Set VIDEO_PATH or SANZARU_MEDIA_PATH"):
            get_path("video")

    def test_missing_reference_env_var(self, mocker):
        """Test that missing IMAGE_PATH raises RuntimeError."""
        mocker.patch.dict(os.environ, {}, clear=True)

        with pytest.raises(RuntimeError, match="not configured.*Set IMAGE_PATH or SANZARU_MEDIA_PATH"):
            get_path("reference")

    def test_missing_audio_env_var(self, mocker):
        """Test that missing AUDIO_PATH raises RuntimeError."""
        mocker.patch.dict(os.environ, {}, clear=True)

        with pytest.raises(RuntimeError, match="not configured.*Set AUDIO_PATH or SANZARU_MEDIA_PATH"):
            get_path("audio")

    def test_empty_string_env_var(self, mocker):
        """Test that empty string env var raises RuntimeError."""
        mocker.patch.dict(os.environ, {"VIDEO_PATH": ""}, clear=True)

        with pytest.raises(RuntimeError, match="not configured"):
            get_path("video")

    def test_whitespace_only_env_var(self, mocker):
        """Test that whitespace-only env var raises RuntimeError."""
        mocker.patch.dict(os.environ, {"VIDEO_PATH": "   \t\n  "}, clear=True)

        with pytest.raises(RuntimeError, match="not configured"):
            get_path("video")

    def test_nonexistent_directory(self, mocker, tmp_path):
        """Test that non-existent directory raises RuntimeError."""
        nonexistent = tmp_path / "does_not_exist"
        mocker.patch.dict(os.environ, {"VIDEO_PATH": str(nonexistent)})

        with pytest.raises(RuntimeError, match="does not exist"):
            get_path("video")

    def test_file_not_directory(self, mocker, tmp_path):
        """Test that a file (not directory) raises RuntimeError."""
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("content")

        mocker.patch.dict(os.environ, {"VIDEO_PATH": str(file_path)})

        with pytest.raises(RuntimeError, match="is not a directory"):
            get_path("video")

    def test_symlink_path_rejected(self, mocker, tmp_path):
        """Test that symlink path raises RuntimeError with security message."""
        real_dir = tmp_path / "real_directory"
        real_dir.mkdir()

        link = tmp_path / "symlink_dir"
        link.symlink_to(real_dir)

        mocker.patch.dict(os.environ, {"IMAGE_PATH": str(link)})

        with pytest.raises(RuntimeError, match="cannot be a symbolic link"):
            get_path("reference")


@pytest.mark.unit
class TestGetPathEdgeCases:
    """Test get_path() edge cases and special scenarios."""

    def test_strips_leading_trailing_whitespace(self, mocker, tmp_video_path):
        """Test that whitespace in env var is stripped and path still works."""
        mocker.patch.dict(os.environ, {"VIDEO_PATH": f"  {tmp_video_path}  \t\n"})

        result = get_path("video")

        assert result == tmp_video_path.resolve()

    def test_relative_path_resolved_to_absolute(self, mocker, tmp_path):
        """Test that relative paths are resolved to absolute paths."""
        # Create a subdirectory
        subdir = tmp_path / "videos"
        subdir.mkdir()

        # Use relative path syntax
        relative_path = str(subdir)
        mocker.patch.dict(os.environ, {"VIDEO_PATH": relative_path})

        result = get_path("video")

        # Result should be absolute
        assert result.is_absolute()
        assert result == subdir.resolve()

    def test_path_with_spaces_in_name(self, mocker, tmp_path):
        """Test that paths with spaces work correctly."""
        dir_with_spaces = tmp_path / "my videos folder"
        dir_with_spaces.mkdir()

        mocker.patch.dict(os.environ, {"VIDEO_PATH": str(dir_with_spaces)})

        result = get_path("video")

        assert result == dir_with_spaces.resolve()
        assert " " in result.name


@pytest.mark.unit
class TestGetPathUnifiedMediaPath:
    """Test get_path() with SANZARU_MEDIA_PATH."""

    def test_unified_path_video(self, mocker, tmp_path):
        """Test that SANZARU_MEDIA_PATH creates videos/ subdir."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        result = get_path("video")

        assert result == (media_root / "videos").resolve()
        assert result.is_dir()

    def test_unified_path_reference(self, mocker, tmp_path):
        """Test that SANZARU_MEDIA_PATH creates images/ subdir."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        result = get_path("reference")

        assert result == (media_root / "images").resolve()
        assert result.is_dir()

    def test_unified_path_audio(self, mocker, tmp_path):
        """Test that SANZARU_MEDIA_PATH creates audio/ subdir."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        result = get_path("audio")

        assert result == (media_root / "audio").resolve()
        assert result.is_dir()

    def test_individual_path_takes_precedence(self, mocker, tmp_path):
        """Test that individual env var takes precedence over SANZARU_MEDIA_PATH."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        custom_video = tmp_path / "custom_videos"
        custom_video.mkdir()

        mocker.patch.dict(
            os.environ,
            {"SANZARU_MEDIA_PATH": str(media_root), "VIDEO_PATH": str(custom_video)},
            clear=True,
        )

        result = get_path("video")

        assert result == custom_video.resolve()
        # Unified subdir should NOT have been created
        assert not (media_root / "videos").exists()

    def test_auto_creates_subdir(self, mocker, tmp_path):
        """Test that unified path auto-creates subdirs."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        videos_dir = media_root / "videos"
        assert not videos_dir.exists()

        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        result = get_path("video")

        assert videos_dir.exists()
        assert result == videos_dir.resolve()

    def test_auto_creates_nested_dirs(self, mocker, tmp_path):
        """Test that unified path creates both root and subdir with parents=True."""
        media_root = tmp_path / "deep" / "nested" / "media"
        assert not media_root.exists()

        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        result = get_path("video")

        assert (media_root / "videos").exists()
        assert result == (media_root / "videos").resolve()

    def test_mixed_precedence(self, mocker, tmp_path):
        """Test mixed config: some individual, some unified."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        custom_video = tmp_path / "custom_videos"
        custom_video.mkdir()

        mocker.patch.dict(
            os.environ,
            {"SANZARU_MEDIA_PATH": str(media_root), "VIDEO_PATH": str(custom_video)},
            clear=True,
        )

        video_result = get_path("video")
        image_result = get_path("reference")
        audio_result = get_path("audio")

        # VIDEO_PATH takes precedence
        assert video_result == custom_video.resolve()
        # Others fall back to unified path
        assert image_result == (media_root / "images").resolve()
        assert audio_result == (media_root / "audio").resolve()

    def test_empty_individual_falls_through(self, mocker, tmp_path):
        """Test that empty individual var falls through to unified path."""
        media_root = tmp_path / "media"
        media_root.mkdir()

        mocker.patch.dict(
            os.environ,
            {"SANZARU_MEDIA_PATH": str(media_root), "VIDEO_PATH": ""},
            clear=True,
        )

        result = get_path("video")

        assert result == (media_root / "videos").resolve()

    def test_unified_path_symlink_rejected(self, mocker, tmp_path):
        """Test that symlinked unified subdirectory is rejected."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        real_dir = tmp_path / "real_videos"
        real_dir.mkdir()

        # Pre-create subdirectory as a symlink
        videos_link = media_root / "videos"
        videos_link.symlink_to(real_dir)

        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        with pytest.raises(RuntimeError, match="cannot be a symbolic link"):
            get_path("video")

    def test_existing_subdir_reused(self, mocker, tmp_path):
        """Test that existing subdirectory is reused without error."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        videos_dir = media_root / "videos"
        videos_dir.mkdir()
        # Place a marker file to prove we're using the existing dir
        marker = videos_dir / "existing.txt"
        marker.write_text("marker")

        mocker.patch.dict(os.environ, {"SANZARU_MEDIA_PATH": str(media_root)}, clear=True)

        result = get_path("video")

        assert result == videos_dir.resolve()
        assert (result / "existing.txt").exists()
