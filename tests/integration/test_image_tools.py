# SPDX-License-Identifier: MIT
"""Integration tests for image generation tools with mocked OpenAI client."""

import base64

import pytest

from sora_mcp_server.tools.image import image_create, image_download, image_get_status


@pytest.mark.integration
async def test_image_create(mocker):
    """Test image generation job creation."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_test123"
    mock_response.status = "queued"
    mock_response.created_at = 1234567890.0

    mock_get_client = mocker.patch("sora_mcp_server.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock(return_value=mock_response)

    result = await image_create(prompt="test image", model="gpt-5", size="1024x1024", quality="high")

    assert result["id"] == "resp_test123"
    assert result["status"] == "queued"
    assert result["created_at"] == 1234567890.0

    # Verify tool config built correctly
    call_kwargs = mock_get_client.return_value.responses.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5"
    assert call_kwargs["input"] == "test image"
    assert call_kwargs["background"] is True
    assert "image_generation" in str(call_kwargs["tools"])


@pytest.mark.integration
async def test_image_get_status(mocker):
    """Test image generation status retrieval."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_test123"
    mock_response.status = "completed"
    mock_response.created_at = 1234567890.0

    mock_get_client = mocker.patch("sora_mcp_server.tools.image.get_client")
    mock_get_client.return_value.responses.retrieve = mocker.AsyncMock(return_value=mock_response)

    result = await image_get_status("resp_test123")

    assert result["id"] == "resp_test123"
    assert result["status"] == "completed"
    mock_get_client.return_value.responses.retrieve.assert_called_once_with("resp_test123")


@pytest.mark.integration
async def test_image_download(mocker, tmp_reference_path):
    """Test image download decodes base64 and writes file."""
    # Create fake base64 data (doesn't need to be valid PNG since we mock Image.open)
    fake_base64 = base64.b64encode(b"fake png data").decode()

    # Mock image generation call result
    mock_img_call = mocker.MagicMock()
    mock_img_call.type = "image_generation_call"
    mock_img_call.result = fake_base64
    mock_img_call.status = "completed"

    mock_response = mocker.MagicMock()
    mock_response.id = "resp_test123"
    mock_response.output = [mock_img_call]

    mocker.patch("sora_mcp_server.tools.image.get_path", return_value=tmp_reference_path)
    mock_get_client = mocker.patch("sora_mcp_server.tools.image.get_client")
    mock_get_client.return_value.responses.retrieve = mocker.AsyncMock(return_value=mock_response)

    # Mock PIL Image.open to avoid trying to parse fake PNG
    mock_img = mocker.MagicMock()
    mock_img.size = (1024, 1024)
    mock_img.format = "PNG"
    mocker.patch("sora_mcp_server.tools.image.Image.open", return_value=mock_img)

    result = await image_download("resp_test123", filename="test.png")

    assert result["filename"] == "test.png"
    assert result["size"] == (1024, 1024)
    assert result["format"] == "png"

    # Verify file was written
    output_file = tmp_reference_path / "test.png"
    assert output_file.exists()
