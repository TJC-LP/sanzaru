# SPDX-License-Identifier: MIT
"""Unit tests for image encoding utilities."""

import base64

import pytest

from sanzaru.tools.image import _encode_image_base64, _get_mime_type


@pytest.mark.unit
class TestEncodeImageBase64:
    """Test base64 encoding of image bytes."""

    def test_encode_success(self):
        """Test successful base64 encoding of image bytes."""
        test_data = b"fake png data"

        result = _encode_image_base64(test_data)

        # Verify it's valid base64
        assert isinstance(result, str)
        # Verify it decodes to original data
        decoded = base64.b64decode(result)
        assert decoded == test_data

    def test_encode_empty_bytes(self):
        """Test encoding empty bytes returns valid base64."""
        result = _encode_image_base64(b"")
        assert isinstance(result, str)
        assert base64.b64decode(result) == b""

    def test_encode_large_data(self):
        """Test encoding larger data works correctly."""
        test_data = b"\x89PNG" + b"\x00" * 10000
        result = _encode_image_base64(test_data)
        assert base64.b64decode(result) == test_data


@pytest.mark.unit
class TestGetMimeType:
    """Test MIME type detection from filename extensions."""

    def test_jpeg_extension(self):
        """Test MIME type for .jpg extension."""
        assert _get_mime_type("test.jpg") == "image/jpeg"

    def test_jpeg_extension_uppercase(self):
        """Test MIME type for .JPG extension."""
        assert _get_mime_type("test.JPG") == "image/jpeg"

    def test_jpeg_long_extension(self):
        """Test MIME type for .jpeg extension."""
        assert _get_mime_type("test.jpeg") == "image/jpeg"

    def test_png_extension(self):
        """Test MIME type for .png extension."""
        assert _get_mime_type("test.png") == "image/png"

    def test_png_extension_uppercase(self):
        """Test MIME type for .PNG extension."""
        assert _get_mime_type("test.PNG") == "image/png"

    def test_webp_extension(self):
        """Test MIME type for .webp extension."""
        assert _get_mime_type("test.webp") == "image/webp"

    def test_unknown_extension_defaults_to_jpeg(self):
        """Test that unknown extensions default to JPEG."""
        assert _get_mime_type("test.gif") == "image/jpeg"
        assert _get_mime_type("test.bmp") == "image/jpeg"
