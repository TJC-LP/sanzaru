# SPDX-License-Identifier: MIT
"""Unit tests for image processing helpers."""

import pytest
from PIL import Image

from sanzaru.tools.reference import (
    load_and_convert_image,
    parse_size,
    resize_crop,
    resize_pad,
    resize_rescale,
    save_image,
    target_dimensions,
)


@pytest.mark.unit
class TestParseSize:
    """ "WxH" parsing for an exact target size."""

    def test_landscape(self):
        assert parse_size("1280x720") == (1280, 720)

    def test_portrait_and_case(self):
        assert parse_size("720X1280") == (720, 1280)

    @pytest.mark.parametrize("bad", ["1280", "1280x", "x720", "12a0x720", "1280x720x3", "-1x720", ""])
    def test_malformed_is_refused(self, bad):
        with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
            parse_size(bad)

    @pytest.mark.parametrize("bad", ["63x720", "1280x4097", "0x0"])
    def test_edges_out_of_range_are_refused(self, bad):
        with pytest.raises(ValueError, match="64-4096"):
            parse_size(bad)

    def test_bounds_are_inclusive(self):
        assert parse_size("64x4096") == (64, 4096)


@pytest.mark.unit
class TestTargetDimensions:
    """Exactly one of aspect_ratio / size; aspect ratios map onto the 720p frame."""

    @pytest.mark.parametrize(
        ("aspect", "expected"),
        [("16:9", (1280, 720)), ("9:16", (720, 1280)), ("1:1", (960, 960))],
    )
    def test_aspect_ratio_uses_the_shared_720p_table(self, aspect, expected):
        assert target_dimensions(aspect, None) == expected

    def test_every_aspect_ratio_has_a_frame(self):
        for aspect in ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9"):
            width, height = target_dimensions(aspect, None)
            assert width > 0 and height > 0

    def test_size_passes_through(self):
        assert target_dimensions(None, "1000x500") == (1000, 500)

    @pytest.mark.parametrize(("aspect", "size"), [(None, None), ("16:9", "1280x720")])
    def test_exactly_one_is_required(self, aspect, size):
        with pytest.raises(ValueError, match="exactly one"):
            target_dimensions(aspect, size)

    def test_unknown_aspect_is_refused(self):
        with pytest.raises(ValueError, match="not supported"):
            target_dimensions("5:4", None)  # type: ignore[arg-type]


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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
