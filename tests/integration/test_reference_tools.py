# SPDX-License-Identifier: MIT
"""Integration tests for reference image tools (file-based, no API mocking)."""

import pytest
from PIL import Image

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.reference import list_reference_images, prepare_reference_image


@pytest.mark.integration
async def test_sora_list_references_with_multiple_files(mocker, tmp_reference_path):
    """Test listing reference images with real files."""
    mocker.patch(
        "sanzaru.tools.reference.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )

    # Create test image files
    Image.new("RGB", (100, 100)).save(tmp_reference_path / "cat.png")
    Image.new("RGB", (200, 100)).save(tmp_reference_path / "dog.jpg")
    Image.new("RGB", (50, 50)).save(tmp_reference_path / "bird.webp")
    # Create a non-image file (should be ignored)
    (tmp_reference_path / "readme.txt").write_text("ignore me")

    result = await list_reference_images()

    assert len(result["data"]) == 3
    filenames = [r["filename"] for r in result["data"]]
    assert "cat.png" in filenames
    assert "dog.jpg" in filenames
    assert "bird.webp" in filenames
    assert "readme.txt" not in filenames

    # Verify metadata structure
    for img_data in result["data"]:
        assert "filename" in img_data
        assert "size_bytes" in img_data
        assert "modified_timestamp" in img_data
        assert "file_type" in img_data


@pytest.mark.integration
async def test_sora_prepare_reference_end_to_end(mocker, tmp_reference_path):
    """Test complete image preparation workflow with real files."""
    mocker.patch(
        "sanzaru.tools.reference.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )

    # Create source image (different aspect ratio from target)
    source = tmp_reference_path / "source.png"
    Image.new("RGB", (400, 200)).save(source, "PNG")

    # Prepare for 1280x720 (landscape) using crop mode
    result = await prepare_reference_image("source.png", "1280x720", resize_mode="crop")

    assert result["output_filename"] == "source_1280x720.png"
    assert result["original_size"] == (400, 200)
    assert result["target_size"] == (1280, 720)
    assert result["resize_mode"] == "crop"

    # Verify output file created
    output_file = tmp_reference_path / "source_1280x720.png"
    assert output_file.exists()

    # Verify dimensions are correct
    img = Image.open(output_file)
    assert img.size == (1280, 720)
    assert img.mode == "RGB"


@pytest.mark.integration
async def test_sora_list_references_with_filtering(mocker, tmp_reference_path):
    """Test reference listing with pattern and type filters."""
    mocker.patch(
        "sanzaru.tools.reference.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )

    # Create various image files
    Image.new("RGB", (100, 100)).save(tmp_reference_path / "cat_001.png")
    Image.new("RGB", (100, 100)).save(tmp_reference_path / "cat_002.png")
    Image.new("RGB", (100, 100)).save(tmp_reference_path / "dog_001.png")
    Image.new("RGB", (100, 100)).save(tmp_reference_path / "bird.jpg")

    # Filter for cat* pattern and png only
    result = await list_reference_images(pattern="cat*.png", file_type="png")

    assert len(result["data"]) == 2
    filenames = [r["filename"] for r in result["data"]]
    assert "cat_001.png" in filenames
    assert "cat_002.png" in filenames
    assert "dog_001.png" not in filenames
    assert "bird.jpg" not in filenames
