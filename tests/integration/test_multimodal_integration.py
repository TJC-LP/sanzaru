"""Integration tests for multimodal feature support."""

import pytest

from sanzaru.config import get_path
from sanzaru.features import check_audio_available, check_image_available, check_video_available


@pytest.mark.integration
def test_video_feature_detection(monkeypatch, tmp_video_path):
    """Test that video feature detection works with VIDEO_PATH set."""
    monkeypatch.setenv("VIDEO_PATH", str(tmp_video_path))
    assert check_video_available() is True


@pytest.mark.integration
def test_video_feature_disabled_without_path(monkeypatch):
    """Test that video feature is disabled when VIDEO_PATH not set."""
    monkeypatch.delenv("VIDEO_PATH", raising=False)
    assert check_video_available() is False


@pytest.mark.integration
def test_image_feature_detection(monkeypatch, tmp_reference_path):
    """Test that image feature detection works with IMAGE_PATH set."""
    monkeypatch.setenv("IMAGE_PATH", str(tmp_reference_path))
    try:
        import PIL  # noqa: F401

        assert check_image_available() is True
    except ImportError:
        # Dependencies not installed
        assert check_image_available() is False


@pytest.mark.integration
def test_image_feature_disabled_without_path(monkeypatch):
    """Test that image feature is disabled when IMAGE_PATH not set."""
    monkeypatch.delenv("IMAGE_PATH", raising=False)
    assert check_image_available() is False


@pytest.mark.integration
@pytest.mark.audio
def test_audio_feature_detection(monkeypatch, tmp_audio_path):
    """Test that audio feature detection works with AUDIO_PATH set."""
    monkeypatch.setenv("AUDIO_PATH", str(tmp_audio_path))
    try:
        import ffmpeg  # noqa: F401
        import pydub  # noqa: F401

        assert check_audio_available() is True
    except ImportError:
        # Dependencies not installed
        assert check_audio_available() is False


@pytest.mark.integration
def test_audio_feature_disabled_without_path(monkeypatch):
    """Test that audio feature is disabled when AUDIO_PATH not set."""
    monkeypatch.delenv("AUDIO_PATH", raising=False)
    assert check_audio_available() is False


@pytest.mark.integration
def test_audio_only_mode(monkeypatch, tmp_audio_path):
    """Test audio-only mode - only AUDIO_PATH set."""
    # Clear all paths except audio
    monkeypatch.delenv("VIDEO_PATH", raising=False)
    monkeypatch.delenv("IMAGE_PATH", raising=False)
    monkeypatch.setenv("AUDIO_PATH", str(tmp_audio_path))

    # Only audio should be available
    assert check_video_available() is False
    assert check_image_available() is False

    # Audio availability depends on dependencies
    try:
        import ffmpeg  # noqa: F401
        import pydub  # noqa: F401

        assert check_audio_available() is True
    except ImportError:
        assert check_audio_available() is False


@pytest.mark.integration
def test_selective_features(monkeypatch, tmp_video_path, tmp_audio_path):
    """Test selective feature enablement - video + audio, no image."""
    monkeypatch.setenv("VIDEO_PATH", str(tmp_video_path))
    monkeypatch.setenv("AUDIO_PATH", str(tmp_audio_path))
    monkeypatch.delenv("IMAGE_PATH", raising=False)

    # Video and audio should be available, image should not
    assert check_video_available() is True
    assert check_image_available() is False

    # Audio availability depends on dependencies
    try:
        import ffmpeg  # noqa: F401
        import pydub  # noqa: F401

        assert check_audio_available() is True
    except ImportError:
        assert check_audio_available() is False


@pytest.mark.integration
def test_config_supports_audio_path():
    """Test that config.get_path supports 'audio' path type."""
    # Should accept 'audio' as a path type (will fail if env var not set, expected)
    try:
        path = get_path("audio")
        assert path is not None
    except RuntimeError as e:
        # Expected if AUDIO_PATH not set
        assert "AUDIO_PATH" in str(e)


@pytest.mark.integration
def test_config_supports_video_path():
    """Test that config.get_path supports 'video' path type."""
    try:
        path = get_path("video")
        assert path is not None
    except RuntimeError as e:
        # Expected if VIDEO_PATH not set
        assert "VIDEO_PATH" in str(e)


@pytest.mark.integration
def test_config_supports_reference_path():
    """Test that config.get_path supports 'reference' path type."""
    try:
        path = get_path("reference")
        assert path is not None
    except RuntimeError as e:
        # Expected if IMAGE_PATH not set
        assert "IMAGE_PATH" in str(e)
