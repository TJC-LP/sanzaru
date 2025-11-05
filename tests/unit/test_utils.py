# SPDX-License-Identifier: MIT
"""Unit tests for utility functions."""

from unittest.mock import patch

import pytest

from sanzaru.utils import generate_filename, suffix_for_variant


@pytest.mark.unit
class TestSuffixForVariant:
    """Test file extension mapping for video variants."""

    def test_video_variant(self):
        assert suffix_for_variant("video") == "mp4"

    def test_thumbnail_variant(self):
        assert suffix_for_variant("thumbnail") == "webp"

    def test_spritesheet_variant(self):
        assert suffix_for_variant("spritesheet") == "jpg"


@pytest.mark.unit
class TestGenerateFilename:
    """Test filename generation logic."""

    def test_basic_filename(self):
        result = generate_filename("abc123", "mp4")
        assert result == "abc123.mp4"

    def test_different_extension(self):
        result = generate_filename("test-id", "png")
        assert result == "test-id.png"

    @patch("sanzaru.utils.time.time", return_value=1234567890.0)
    def test_filename_with_timestamp(self, mock_time):
        result = generate_filename("img", "png", use_timestamp=True)
        assert result == "img_1234567890.png"

    @patch("sanzaru.utils.time.time", return_value=9999999999.0)
    def test_timestamp_precision(self, mock_time):
        """Verify timestamp is converted to int (no decimal)."""
        result = generate_filename("test", "jpg", use_timestamp=True)
        assert result == "test_9999999999.jpg"
        assert "." not in result.split("_")[1].split(".")[0]  # No decimal in timestamp part
