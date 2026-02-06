# SPDX-License-Identifier: MIT
"""Integration tests for image input functionality."""

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.image import create_image


@pytest.mark.integration
class TestImageInput:
    """Test image input parameters for create_image."""

    async def test_create_image_with_single_input(self, mocker, tmp_reference_path, sample_image):
        """Test creating image with single reference image."""
        # Mock the OpenAI client
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.id = "resp_test123"
        mock_response.status = "queued"
        mock_response.created_at = 1234567890.0

        mock_client.responses.create = mocker.AsyncMock(return_value=mock_response)
        mocker.patch("sanzaru.tools.image.get_client", return_value=mock_client)
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Create test image
        test_img = tmp_reference_path / "test.png"
        test_img.write_bytes(b"fake png data")

        result = await create_image(prompt="add a flamingo", input_images=["test.png"])

        assert result["id"] == "resp_test123"
        assert result["status"] == "queued"

        # Verify API was called with structured input
        call_args = mock_client.responses.create.call_args
        assert call_args.kwargs["input"] is not None
        input_param = call_args.kwargs["input"]
        assert isinstance(input_param, list)
        assert len(input_param) == 1
        assert input_param[0]["role"] == "user"

    async def test_create_image_with_multiple_inputs(self, mocker, tmp_reference_path):
        """Test creating image with multiple reference images."""
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.id = "resp_multi"
        mock_response.status = "queued"
        mock_response.created_at = 1234567890.0

        mock_client.responses.create = mocker.AsyncMock(return_value=mock_response)
        mocker.patch("sanzaru.tools.image.get_client", return_value=mock_client)
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Create test images
        for i in range(3):
            img = tmp_reference_path / f"img{i}.png"
            img.write_bytes(b"fake data")

        result = await create_image(prompt="combine these images", input_images=["img0.png", "img1.png", "img2.png"])

        assert result["id"] == "resp_multi"

        # Verify content has 1 text + 3 images
        call_args = mock_client.responses.create.call_args
        input_param = call_args.kwargs["input"]
        content = input_param[0]["content"]
        assert len(content) == 4  # 1 text + 3 images
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"
        assert content[2]["type"] == "input_image"
        assert content[3]["type"] == "input_image"

    async def test_create_image_with_tool_config(self, mocker, tmp_reference_path):
        """Test that custom tool_config is passed through correctly."""
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.id = "resp_config"
        mock_response.status = "queued"
        mock_response.created_at = 1234567890.0

        mock_client.responses.create = mocker.AsyncMock(return_value=mock_response)
        mocker.patch("sanzaru.tools.image.get_client", return_value=mock_client)
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        test_img = tmp_reference_path / "face.png"
        test_img.write_bytes(b"fake data")

        custom_config = {
            "type": "image_generation",
            "model": "gpt-image-1-mini",
            "moderation": "low",
            "input_fidelity": "high",
            "quality": "medium",
        }

        result = await create_image(prompt="add logo to shirt", input_images=["face.png"], tool_config=custom_config)

        assert result["id"] == "resp_config"

        # Verify custom config was passed through
        call_args = mock_client.responses.create.call_args
        tools = call_args.kwargs["tools"]
        assert tools[0]["model"] == "gpt-image-1-mini"
        assert tools[0]["moderation"] == "low"
        assert tools[0]["input_fidelity"] == "high"
        assert tools[0]["quality"] == "medium"

    async def test_create_image_with_mask(self, mocker, tmp_reference_path):
        """Test masked inpainting with alpha channel."""
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.id = "resp_mask"
        mock_response.status = "queued"
        mock_response.created_at = 1234567890.0

        mock_file_obj = mocker.MagicMock()
        mock_file_obj.id = "file_mask123"

        mock_client.responses.create = mocker.AsyncMock(return_value=mock_response)
        mock_client.files.create = mocker.AsyncMock(return_value=mock_file_obj)
        mocker.patch("sanzaru.tools.image.get_client", return_value=mock_client)
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Create test files
        test_img = tmp_reference_path / "pool.png"
        test_img.write_bytes(b"fake image data")
        mask_img = tmp_reference_path / "mask.png"
        mask_img.write_bytes(b"fake mask data")

        result = await create_image(prompt="add flamingo", input_images=["pool.png"], mask_filename="mask.png")

        assert result["id"] == "resp_mask"

        # Verify mask was uploaded
        assert mock_client.files.create.called

        # Verify mask file_id in tool config
        call_args = mock_client.responses.create.call_args
        tools = call_args.kwargs["tools"]
        assert tools[0]["input_image_mask"] == {"file_id": "file_mask123"}

    async def test_create_image_mask_without_images_error(self, mocker, tmp_reference_path):
        """Test error when mask provided without input_images."""
        # Mock to prevent OPENAI_API_KEY check
        mocker.patch("sanzaru.tools.image.get_client")
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        with pytest.raises(ValueError, match="mask_filename requires input_images parameter"):
            await create_image(prompt="test", mask_filename="mask.png")

    async def test_create_image_invalid_filename_path_traversal(self, mocker, tmp_reference_path):
        """Test path traversal protection for input images.

        Path traversal filenames without valid extensions are caught by the
        extension validation first. Files with valid extensions but traversal
        paths are caught by the storage backend's path validation.
        """
        mocker.patch("sanzaru.tools.image.get_client")
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Without a valid extension, the extension check fires first
        with pytest.raises(ValueError, match="Unsupported image format"):
            await create_image(prompt="test", input_images=["../../../etc/passwd"])

        # With a valid extension, path traversal detection fires via storage backend
        with pytest.raises(ValueError, match="path traversal detected"):
            await create_image(prompt="test", input_images=["../../../etc/evil.png"])

    async def test_create_image_unsupported_format(self, mocker, tmp_reference_path):
        """Test error for unsupported image format (e.g., .gif)."""
        mocker.patch("sanzaru.tools.image.get_client")
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Create a .gif file
        gif_file = tmp_reference_path / "test.gif"
        gif_file.write_bytes(b"fake gif")

        with pytest.raises(ValueError, match="Unsupported image format.*use JPEG, PNG, WEBP"):
            await create_image(prompt="test", input_images=["test.gif"])

    async def test_create_image_symlink_rejected(self, mocker, tmp_reference_path):
        """Test symlink detection rejects symlinked images."""
        mock_client = mocker.MagicMock()
        mock_client.responses.create = mocker.AsyncMock()

        mocker.patch("sanzaru.tools.image.get_client", return_value=mock_client)
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Create a real file INSIDE reference path and symlink to it
        real_file = tmp_reference_path / "real.png"
        real_file.write_bytes(b"real data")

        link = tmp_reference_path / "link.png"
        link.symlink_to(real_file)

        with pytest.raises(ValueError, match="cannot be a symbolic link"):
            await create_image(prompt="test", input_images=["link.png"])

    async def test_create_image_mask_non_png_rejected(self, mocker, tmp_reference_path):
        """Test that non-PNG masks are rejected."""
        mocker.patch("sanzaru.tools.image.get_client")
        storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
        mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)

        # Create image and non-PNG mask
        img = tmp_reference_path / "img.png"
        img.write_bytes(b"image")
        mask = tmp_reference_path / "mask.jpg"
        mask.write_bytes(b"mask")

        with pytest.raises(ValueError, match="Mask must be PNG format"):
            await create_image(prompt="test", input_images=["img.png"], mask_filename="mask.jpg")
