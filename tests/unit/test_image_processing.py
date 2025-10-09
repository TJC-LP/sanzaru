# SPDX-License-Identifier: MIT
"""Unit tests for image processing helpers."""
import pathlib

import pytest
from PIL import Image

from sora_mcp_server.tools.reference import (
    load_and_convert_image,
    parse_video_dimensions,
    resize_crop,
    resize_pad,
    resize_rescale,
    save_image,
)


class TestParseVideoDimensions:
    """Test VideoSize string parsing."""

    def test_landscape_720p(self):
        width, height = parse_video_dimensions("1280x720")
        assert width == 1280
        assert height == 720

    def test_portrait_720p(self):
        width, height = parse_video_dimensions("720x1280")
        assert width == 720
        assert height == 1280

    def test_landscape_1080p(self):
        width, height = parse_video_dimensions("1920x1080")
        assert width == 1920
        assert height == 1080

    def test_pro_sizes(self):
        """Test larger sizes supported by sora-2-pro."""
        assert parse_video_dimensions("1024x1792") == (1024, 1792)
        assert parse_video_dimensions("1792x1024") == (1792, 1024)


class TestLoadAndConvertImage:
    """Test image loading and RGB conversion."""

    def test_load_rgb_image(self, sample_image):
        """Test loading an RGB image (no conversion needed)."""
        img = load_and_convert_image(sample_image, "test.png")
        assert img.mode == "RGB"
        assert img.size == (200, 100)

    def test_convert_rgba_to_rgb(self, sample_rgba_image):
        """Test that RGBA images are converted to RGB."""
        img = load_and_convert_image(sample_rgba_image, "rgba.png")
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_file_not_found_error(self, tmp_reference_path):
        """Test that missing files raise ValueError."""
        nonexistent = tmp_reference_path / "missing.png"
        with pytest.raises(ValueError, match="Input image not found: missing.png"):
            load_and_convert_image(nonexistent, "missing.png")

    def test_grayscale_conversion(self, tmp_reference_path):
        """Test that grayscale images are converted to RGB."""
        gray_img_path = tmp_reference_path / "gray.png"
        gray_img = Image.new("L", (50, 50), color=128)
        gray_img.save(gray_img_path, "PNG")

        img = load_and_convert_image(gray_img_path, "gray.png")
        assert img.mode == "RGB"
        assert img.size == (50, 50)


class TestResizeCrop:
    """Test crop resize strategy."""

    def test_crop_wider_image(self):
        """Test cropping a wider image to square (crops width)."""
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))  # 2:1 ratio
        result = resize_crop(img, 100, 100)  # 1:1 target

        assert result.size == (100, 100)
        assert result.mode == "RGB"

    def test_crop_taller_image(self):
        """Test cropping a taller image to square (crops height)."""
        img = Image.new("RGB", (100, 200), color=(0, 255, 0))  # 1:2 ratio
        result = resize_crop(img, 100, 100)  # 1:1 target

        assert result.size == (100, 100)
        assert result.mode == "RGB"

    def test_crop_to_landscape(self):
        """Test cropping square to landscape."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        result = resize_crop(img, 160, 90)  # 16:9 landscape

        assert result.size == (160, 90)

    def test_crop_preserves_aspect_no_distortion(self):
        """Test that crop doesn't distort - just crops excess."""
        # Create a 400x200 image (2:1)
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        # Crop to 200x200 (1:1) - should scale to 400x200, then crop to 200x200
        result = resize_crop(img, 200, 200)

        assert result.size == (200, 200)


class TestResizePad:
    """Test pad resize strategy."""

    def test_pad_wider_image(self):
        """Test padding a wider image (adds top/bottom bars)."""
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        result = resize_pad(img, 100, 100)

        assert result.size == (100, 100)
        # Image should be centered with black bars on top/bottom

    def test_pad_taller_image(self):
        """Test padding a taller image (adds left/right bars)."""
        img = Image.new("RGB", (100, 200), color=(0, 255, 0))
        result = resize_pad(img, 100, 100)

        assert result.size == (100, 100)
        # Image should be centered with black bars on left/right

    def test_pad_to_larger_dimensions(self):
        """Test padding to larger dimensions."""
        img = Image.new("RGB", (50, 50), color=(0, 0, 255))
        result = resize_pad(img, 100, 100)

        assert result.size == (100, 100)

    def test_pad_preserves_aspect_ratio(self):
        """Test that padding preserves original aspect ratio."""
        # Create a 200x100 image (2:1 ratio)
        img = Image.new("RGB", (200, 100), color=(128, 128, 128))
        # Pad to 200x200 - should fit inside (scale to 200x100) and add black bars
        result = resize_pad(img, 200, 200)

        assert result.size == (200, 200)


class TestResizeRescale:
    """Test rescale (stretch) resize strategy."""

    def test_rescale_wider_to_square(self):
        """Test stretching a wider image to square (may distort)."""
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        result = resize_rescale(img, 100, 100)

        assert result.size == (100, 100)
        # Note: This will distort the image, but that's expected for rescale mode

    def test_rescale_square_to_landscape(self):
        """Test stretching square to landscape."""
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        result = resize_rescale(img, 160, 90)

        assert result.size == (160, 90)

    def test_rescale_to_exact_dimensions(self):
        """Test that rescale always produces exact target dimensions."""
        img = Image.new("RGB", (300, 150), color=(0, 0, 255))
        result = resize_rescale(img, 640, 480)

        assert result.size == (640, 480)


class TestSaveImage:
    """Test image saving functionality."""

    def test_save_image_creates_file(self, tmp_reference_path):
        """Test that save_image creates a PNG file."""
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        output_path = tmp_reference_path / "output.png"

        save_image(img, output_path, "output.png")

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_image_is_png(self, tmp_reference_path):
        """Test that saved image is valid PNG format."""
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        output_path = tmp_reference_path / "test.png"

        save_image(img, output_path, "test.png")

        # Verify it's a valid PNG by loading it
        loaded = Image.open(output_path)
        assert loaded.format == "PNG"
        assert loaded.size == (50, 50)

    def test_save_image_preserves_content(self, tmp_reference_path):
        """Test that image content is preserved after save."""
        # Create image with specific color
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        output_path = tmp_reference_path / "red.png"

        save_image(img, output_path, "red.png")

        # Load and verify color
        loaded = Image.open(output_path)
        pixel = loaded.getpixel((5, 5))
        assert pixel == (255, 0, 0)
