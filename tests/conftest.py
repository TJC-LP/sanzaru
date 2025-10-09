# SPDX-License-Identifier: MIT
"""Shared pytest fixtures for Sora MCP server tests."""

import pathlib

import pytest
from PIL import Image


@pytest.fixture
def tmp_reference_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary directory for reference images."""
    ref_path = tmp_path / "references"
    ref_path.mkdir()
    return ref_path


@pytest.fixture
def tmp_video_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary directory for video downloads."""
    video_path = tmp_path / "videos"
    video_path.mkdir()
    return video_path


@pytest.fixture
def sample_image(tmp_reference_path: pathlib.Path) -> pathlib.Path:
    """Create a sample RGB image for testing.

    Returns path to a 200x100 RGB test image.
    """
    img_path = tmp_reference_path / "test.png"
    img = Image.new("RGB", (200, 100), color=(255, 0, 0))  # Red 200x100 image
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def sample_square_image(tmp_reference_path: pathlib.Path) -> pathlib.Path:
    """Create a sample square RGB image for testing.

    Returns path to a 100x100 RGB test image.
    """
    img_path = tmp_reference_path / "square.png"
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))  # Green 100x100 image
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def sample_rgba_image(tmp_reference_path: pathlib.Path) -> pathlib.Path:
    """Create a sample RGBA image for testing color conversion.

    Returns path to a 100x100 RGBA test image.
    """
    img_path = tmp_reference_path / "rgba.png"
    img = Image.new("RGBA", (100, 100), color=(0, 0, 255, 128))  # Semi-transparent blue
    img.save(img_path, "PNG")
    return img_path


# ==================== Integration Test Fixtures ====================


@pytest.fixture
def mock_video_response():
    """Sample Video object for mocking OpenAI video API responses."""
    from openai.types import Video

    return Video(
        id="vid_test123",
        object="video",  # Required by Pydantic
        status="completed",
        progress=100,
        model="sora-2",
        seconds="8",
        size="1280x720",
        created_at=1234567890,
    )


@pytest.fixture
def mock_video_queued():
    """Sample Video object in queued state."""
    from openai.types import Video

    return Video(
        id="vid_queued",
        object="video",  # Required by Pydantic
        status="queued",
        progress=0,
        model="sora-2",
        seconds="8",
        size="1280x720",
        created_at=1234567890,
    )
