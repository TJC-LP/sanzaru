# SPDX-License-Identifier: MIT
"""Unit tests for image encoding utilities."""

import base64
import pathlib

import pytest

from sora_mcp_server.tools.image import _encode_image_base64, _get_mime_type


@pytest.mark.unit
class TestEncodeImageBase64:
    """Test base64 encoding of image files."""

    def test_encode_success(self, tmp_path):
        """Test successful base64 encoding of image file."""
        # Create a fake image file
        test_file = tmp_path / "test.png"
        test_data = b"fake png data"
        test_file.write_bytes(test_data)

        result = _encode_image_base64(test_file)

        # Verify it's valid base64
        assert isinstance(result, str)
        # Verify it decodes to original data
        decoded = base64.b64decode(result)
        assert decoded == test_data

    def test_encode_file_not_found(self, tmp_path):
        """Test error handling when image file doesn't exist."""
        nonexistent = tmp_path / "missing.png"

        with pytest.raises(ValueError, match="Image file not found: missing.png"):
            _encode_image_base64(nonexistent)

    def test_encode_permission_error(self, tmp_path):
        """Test error handling when permission denied."""
        # This test is platform-dependent
        # We'll create a file and make it unreadable
        test_file = tmp_path / "forbidden.png"
        test_file.write_bytes(b"data")

        # Try to make it unreadable (may not work on all platforms)
        try:
            test_file.chmod(0o000)
            with pytest.raises(ValueError, match="Permission denied reading image"):
                _encode_image_base64(test_file)
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)


@pytest.mark.unit
class TestGetMimeType:
    """Test MIME type detection from file extensions."""

    def test_jpeg_extension(self):
        """Test MIME type for .jpg extension."""
        assert _get_mime_type(pathlib.Path("test.jpg")) == "image/jpeg"

    def test_jpeg_extension_uppercase(self):
        """Test MIME type for .JPG extension."""
        assert _get_mime_type(pathlib.Path("test.JPG")) == "image/jpeg"

    def test_jpeg_long_extension(self):
        """Test MIME type for .jpeg extension."""
        assert _get_mime_type(pathlib.Path("test.jpeg")) == "image/jpeg"

    def test_png_extension(self):
        """Test MIME type for .png extension."""
        assert _get_mime_type(pathlib.Path("test.png")) == "image/png"

    def test_png_extension_uppercase(self):
        """Test MIME type for .PNG extension."""
        assert _get_mime_type(pathlib.Path("test.PNG")) == "image/png"

    def test_webp_extension(self):
        """Test MIME type for .webp extension."""
        assert _get_mime_type(pathlib.Path("test.webp")) == "image/webp"

    def test_unknown_extension_defaults_to_jpeg(self):
        """Test that unknown extensions default to JPEG."""
        assert _get_mime_type(pathlib.Path("test.gif")) == "image/jpeg"
        assert _get_mime_type(pathlib.Path("test.bmp")) == "image/jpeg"
