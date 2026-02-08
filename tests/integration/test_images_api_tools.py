# SPDX-License-Identifier: MIT
"""Integration tests for Images API tools (generate_image, edit_image).

These tests use mocked OpenAI client responses to verify the Images API
integration works correctly without making actual API calls.
"""

import base64

import pytest
from PIL import Image

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.images_api import edit_image, generate_image

# =============================================================================
# generate_image Tests
# =============================================================================


@pytest.mark.integration
async def test_generate_image_basic(mocker, tmp_reference_path):
    """Test basic image generation via Images API."""
    # Mock OpenAI response - usage must be None or real Usage type (Pydantic validation)
    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake png data").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None  # Pydantic requires None or real Usage type

    # Patch dependencies
    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.generate = mocker.AsyncMock(return_value=mock_response)

    # Mock PIL for dimensions
    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    # Call function
    result = await generate_image(prompt="test image", model="gpt-image-1.5")

    # Assert results
    assert result.model == "gpt-image-1.5"
    assert result.size == (1024, 1024)
    assert result.format == "png"
    assert result.usage is None  # No usage in this test

    # Verify API call
    call_kwargs = mock_get_client.return_value.images.generate.call_args.kwargs
    assert call_kwargs["prompt"] == "test image"
    assert call_kwargs["model"] == "gpt-image-1.5"
    assert call_kwargs["n"] == 1


@pytest.mark.integration
async def test_generate_image_with_custom_filename(mocker, tmp_reference_path):
    """Test image generation with custom output filename."""
    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake png data").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.generate = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    result = await generate_image(prompt="test", filename="custom_name.png")

    assert result.filename == "custom_name.png"


@pytest.mark.integration
async def test_generate_image_all_params(mocker, tmp_reference_path):
    """Test image generation with all parameters specified."""
    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake png data").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.generate = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1536, 1024), format="WEBP")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    result = await generate_image(
        prompt="detailed test image",
        model="gpt-image-1.5",
        size="1536x1024",
        quality="high",
        background="transparent",
        output_format="webp",
        moderation="low",
        filename="test_all_params.webp",
    )

    # Verify all params passed to API
    call_kwargs = mock_get_client.return_value.images.generate.call_args.kwargs
    assert call_kwargs["prompt"] == "detailed test image"
    assert call_kwargs["model"] == "gpt-image-1.5"
    assert call_kwargs["size"] == "1536x1024"
    assert call_kwargs["quality"] == "high"
    assert call_kwargs["background"] == "transparent"
    assert call_kwargs["output_format"] == "webp"
    assert call_kwargs["moderation"] == "low"

    assert result.size == (1536, 1024)
    assert result.format == "webp"


@pytest.mark.integration
async def test_generate_image_usage_tracking(mocker, tmp_reference_path):
    """Test that usage information is properly returned."""
    from openai.types.images_response import Usage, UsageInputTokensDetails

    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake png data").decode()
    mock_response.data = [mock_data]

    # Create real Usage object (Pydantic model requires real type, not MagicMock)
    mock_response.usage = Usage(
        input_tokens=100,
        output_tokens=5000,
        total_tokens=5100,
        input_tokens_details=UsageInputTokensDetails(image_tokens=0, text_tokens=100),
        output_tokens_details=None,
    )

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.generate = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    result = await generate_image(prompt="test")

    assert result.usage is not None
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 5000
    assert result.usage.total_tokens == 5100


@pytest.mark.integration
async def test_generate_image_no_data_error(mocker, tmp_reference_path):
    """Test error when API returns empty data."""
    mock_response = mocker.MagicMock()
    mock_response.data = []  # Empty data

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.generate = mocker.AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="No image data returned from API"):
        await generate_image(prompt="test")


@pytest.mark.integration
async def test_generate_image_no_b64_error(mocker, tmp_reference_path):
    """Test error when b64_json is missing from response."""
    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = None  # Missing b64_json
    mock_response.data = [mock_data]

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.generate = mocker.AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="No base64 image data returned"):
        await generate_image(prompt="test")


# =============================================================================
# edit_image Tests - Success Paths
# =============================================================================


@pytest.mark.integration
async def test_edit_image_single_input(mocker, tmp_reference_path):
    """Test editing with a single input image."""
    # Create test input image
    input_file = tmp_reference_path / "input.png"
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(input_file, "PNG")

    # Mock response - usage must be None (Pydantic requires real type or None)
    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake edited png").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.edit = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    result = await edit_image(prompt="add a hat", input_images=["input.png"])

    assert result.model == "gpt-image-1.5"
    assert result.size == (1024, 1024)

    # Verify API call - single image passed as tuple, not list
    call_kwargs = mock_get_client.return_value.images.edit.call_args.kwargs
    assert call_kwargs["prompt"] == "add a hat"
    # Single image should be passed as tuple (filename, bytes, mime)
    assert isinstance(call_kwargs["image"], tuple)
    assert len(call_kwargs["image"]) == 3


@pytest.mark.integration
async def test_edit_image_multiple_inputs(mocker, tmp_reference_path):
    """Test editing with multiple input images (composition)."""
    # Create test input images
    for i in range(3):
        img_file = tmp_reference_path / f"img{i}.png"
        Image.new("RGB", (100, 100), color=(i * 50, 0, 0)).save(img_file, "PNG")

    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake composed png").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.edit = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    result = await edit_image(
        prompt="combine into collage",
        input_images=["img0.png", "img1.png", "img2.png"],
    )

    assert result.model == "gpt-image-1.5"

    # Verify API call - multiple images passed as list of tuples
    call_kwargs = mock_get_client.return_value.images.edit.call_args.kwargs
    assert isinstance(call_kwargs["image"], list)
    assert len(call_kwargs["image"]) == 3
    for img_tuple in call_kwargs["image"]:
        assert isinstance(img_tuple, tuple)
        assert len(img_tuple) == 3


@pytest.mark.integration
async def test_edit_image_with_mask(mocker, tmp_reference_path):
    """Test inpainting with mask file."""
    # Create input image and mask
    input_file = tmp_reference_path / "input.png"
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(input_file, "PNG")

    mask_file = tmp_reference_path / "mask.png"
    Image.new("RGBA", (100, 100), color=(0, 0, 0, 0)).save(mask_file, "PNG")

    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake inpainted png").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.edit = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    result = await edit_image(
        prompt="add flamingo in masked area",
        input_images=["input.png"],
        mask_filename="mask.png",
    )

    assert result.model == "gpt-image-1.5"

    # Verify mask was passed
    call_kwargs = mock_get_client.return_value.images.edit.call_args.kwargs
    assert "mask" in call_kwargs
    assert isinstance(call_kwargs["mask"], tuple)
    assert call_kwargs["mask"][0] == "mask.png"
    assert call_kwargs["mask"][2] == "image/png"


@pytest.mark.integration
async def test_edit_image_with_input_fidelity(mocker, tmp_reference_path):
    """Test input_fidelity parameter."""
    input_file = tmp_reference_path / "face.png"
    Image.new("RGB", (100, 100)).save(input_file, "PNG")

    mock_response = mocker.MagicMock()
    mock_data = mocker.MagicMock()
    mock_data.b64_json = base64.b64encode(b"fake png").decode()
    mock_response.data = [mock_data]
    mock_response.usage = None

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.edit = mocker.AsyncMock(return_value=mock_response)

    mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
    mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

    await edit_image(
        prompt="change hair color",
        input_images=["face.png"],
        input_fidelity="high",
    )

    call_kwargs = mock_get_client.return_value.images.edit.call_args.kwargs
    assert call_kwargs["input_fidelity"] == "high"


@pytest.mark.integration
async def test_edit_image_mime_types(mocker, tmp_reference_path):
    """Test correct MIME types are passed for different image formats."""
    test_cases = [
        ("test.jpg", "image/jpeg"),
        ("test.jpeg", "image/jpeg"),
        ("test.png", "image/png"),
        ("test.webp", "image/webp"),
    ]

    for filename, expected_mime in test_cases:
        # Create test file
        img_file = tmp_reference_path / filename
        img_file.write_bytes(b"fake image data")

        mock_response = mocker.MagicMock()
        mock_data = mocker.MagicMock()
        mock_data.b64_json = base64.b64encode(b"fake png").decode()
        mock_response.data = [mock_data]
        mock_response.usage = None

        mocker.patch(
            "sanzaru.tools.images_api.get_storage",
            return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
        )
        mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
        mock_get_client.return_value.images.edit = mocker.AsyncMock(return_value=mock_response)

        mock_img = mocker.MagicMock(size=(1024, 1024), format="PNG")
        mocker.patch("sanzaru.tools.images_api.Image.open", return_value=mock_img)

        await edit_image(prompt="test", input_images=[filename])

        call_kwargs = mock_get_client.return_value.images.edit.call_args.kwargs
        image_tuple = call_kwargs["image"]
        assert image_tuple[2] == expected_mime, f"Expected {expected_mime} for {filename}"


# =============================================================================
# edit_image Tests - Validation and Errors
# =============================================================================


@pytest.mark.integration
async def test_edit_image_empty_list_error(mocker, tmp_reference_path):
    """Test error on empty input_images list."""
    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="At least one input image is required"):
        await edit_image(prompt="test", input_images=[])


@pytest.mark.integration
async def test_edit_image_exceeds_max_images(mocker, tmp_reference_path):
    """Test error when more than 16 images provided."""
    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    too_many_images = [f"img{i}.png" for i in range(17)]

    with pytest.raises(ValueError, match="Maximum 16 input images allowed"):
        await edit_image(prompt="test", input_images=too_many_images)


@pytest.mark.integration
async def test_edit_image_unsupported_format(mocker, tmp_reference_path):
    """Test error on unsupported file extension."""
    # Create file with unsupported extension
    bad_file = tmp_reference_path / "image.gif"
    bad_file.write_bytes(b"fake gif data")

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="Unsupported image format.*use JPEG, PNG, WEBP"):
        await edit_image(prompt="test", input_images=["image.gif"])


@pytest.mark.integration
async def test_edit_image_mask_non_png_error(mocker, tmp_reference_path):
    """Test error when mask is not PNG format."""
    input_file = tmp_reference_path / "input.png"
    Image.new("RGB", (100, 100)).save(input_file, "PNG")

    mask_file = tmp_reference_path / "mask.jpg"
    mask_file.write_bytes(b"fake jpg mask")

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="Mask must be PNG format"):
        await edit_image(prompt="test", input_images=["input.png"], mask_filename="mask.jpg")


@pytest.mark.integration
async def test_edit_image_file_not_found(mocker, tmp_reference_path):
    """Test error when input image doesn't exist."""
    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="File not found"):
        await edit_image(prompt="test", input_images=["nonexistent.png"])


@pytest.mark.integration
async def test_edit_image_no_data_error(mocker, tmp_reference_path):
    """Test error when edit API returns empty data."""
    input_file = tmp_reference_path / "input.png"
    Image.new("RGB", (100, 100)).save(input_file, "PNG")

    mock_response = mocker.MagicMock()
    mock_response.data = []  # Empty

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mock_get_client = mocker.patch("sanzaru.tools.images_api.get_client")
    mock_get_client.return_value.images.edit = mocker.AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="No image data returned from API"):
        await edit_image(prompt="test", input_images=["input.png"])


# =============================================================================
# Security Tests
# =============================================================================


@pytest.mark.integration
async def test_edit_image_path_traversal(mocker, tmp_reference_path):
    """Test path traversal is rejected."""
    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="path traversal detected"):
        await edit_image(prompt="test", input_images=["../../../etc/passwd.png"])


@pytest.mark.integration
async def test_edit_image_symlink_rejected(mocker, tmp_reference_path):
    """Test symlink detection rejects symlinked images."""
    # Create real file and symlink
    real_file = tmp_reference_path / "real.png"
    Image.new("RGB", (100, 100)).save(real_file, "PNG")

    link = tmp_reference_path / "link.png"
    link.symlink_to(real_file)

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        await edit_image(prompt="test", input_images=["link.png"])


@pytest.mark.integration
async def test_edit_image_mask_symlink_rejected(mocker, tmp_reference_path):
    """Test symlink detection rejects symlinked mask file."""
    input_file = tmp_reference_path / "input.png"
    Image.new("RGB", (100, 100)).save(input_file, "PNG")

    real_mask = tmp_reference_path / "real_mask.png"
    Image.new("RGBA", (100, 100)).save(real_mask, "PNG")

    link_mask = tmp_reference_path / "link_mask.png"
    link_mask.symlink_to(real_mask)

    mocker.patch(
        "sanzaru.tools.images_api.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    mocker.patch("sanzaru.tools.images_api.get_client")

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        await edit_image(prompt="test", input_images=["input.png"], mask_filename="link_mask.png")
