# SPDX-License-Identifier: MIT
"""Integration tests for image generation tools with mocked OpenAI client."""

import base64

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.image import create_image, download_image, get_image_status


@pytest.mark.integration
async def test_image_create(mocker, tmp_reference_path):
    """Test image generation job creation."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_test123"
    mock_response.status = "queued"
    mock_response.created_at = 1234567890.0

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock(return_value=mock_response)

    result = await create_image(
        prompt="test image",
        model="gpt-5",
        tool_config={"type": "image_generation", "size": "1024x1024", "quality": "high"},
    )

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
async def test_image_create_defaults_to_gpt_image_2(mocker, tmp_reference_path):
    """create_image injects model=gpt-image-2 into the tool config when the caller omits it."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_default"
    mock_response.status = "queued"
    mock_response.created_at = 1234567890.0

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock(return_value=mock_response)

    await create_image(prompt="a cat")

    tools = mock_get_client.return_value.responses.create.call_args.kwargs["tools"]
    assert tools[0]["type"] == "image_generation"
    assert tools[0]["model"] == "gpt-image-2"


@pytest.mark.integration
async def test_image_create_preserves_explicit_model(mocker, tmp_reference_path):
    """create_image keeps a caller-specified image model instead of overriding it."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_explicit"
    mock_response.status = "queued"
    mock_response.created_at = 1234567890.0

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock(return_value=mock_response)

    await create_image(
        prompt="a cat",
        tool_config={"type": "image_generation", "model": "gpt-image-1-mini", "quality": "low"},
    )

    tools = mock_get_client.return_value.responses.create.call_args.kwargs["tools"]
    assert tools[0]["model"] == "gpt-image-1-mini"


@pytest.mark.integration
async def test_image_create_transparent_without_model_raises(mocker, tmp_reference_path):
    """create_image rejects transparent background when gpt-image-2 is the injected default."""
    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock()

    with pytest.raises(ValueError, match="gpt-image-1.5"):
        await create_image(
            prompt="a logo",
            tool_config={"type": "image_generation", "background": "transparent"},
        )

    mock_get_client.return_value.responses.create.assert_not_called()


@pytest.mark.integration
async def test_image_create_transparent_explicit_gpt_image_2_raises(mocker, tmp_reference_path):
    """The transparent guard also fires when gpt-image-2 is passed explicitly."""
    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock()

    with pytest.raises(ValueError, match="transparent"):
        await create_image(
            prompt="a logo",
            tool_config={"type": "image_generation", "model": "gpt-image-2", "background": "transparent"},
        )


@pytest.mark.integration
async def test_image_create_transparent_with_gpt_image_1_5_allowed(mocker, tmp_reference_path):
    """gpt-image-1.5 supports transparent, so it must not be blocked by the guard."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_t"
    mock_response.status = "queued"
    mock_response.created_at = 1234567890.0

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock(return_value=mock_response)

    await create_image(
        prompt="a logo",
        tool_config={"type": "image_generation", "model": "gpt-image-1.5", "background": "transparent"},
    )

    tools = mock_get_client.return_value.responses.create.call_args.kwargs["tools"]
    assert tools[0]["model"] == "gpt-image-1.5"
    assert tools[0]["background"] == "transparent"


@pytest.mark.integration
async def test_image_create_does_not_mutate_caller_tool_config(mocker, tmp_reference_path):
    """create_image injects the default model into a copy, never the caller's dict."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_m"
    mock_response.status = "queued"
    mock_response.created_at = 1234567890.0

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.create = mocker.AsyncMock(return_value=mock_response)

    caller_config = {"type": "image_generation", "quality": "high"}
    await create_image(prompt="a cat", tool_config=caller_config)

    assert "model" not in caller_config  # caller's dict stays clean
    tools = mock_get_client.return_value.responses.create.call_args.kwargs["tools"]
    assert tools[0]["model"] == "gpt-image-2"


@pytest.mark.integration
async def test_image_get_status(mocker):
    """Test image generation status retrieval."""
    mock_response = mocker.MagicMock()
    mock_response.id = "resp_test123"
    mock_response.status = "completed"
    mock_response.created_at = 1234567890.0

    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.retrieve = mocker.AsyncMock(return_value=mock_response)

    result = await get_image_status("resp_test123")

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

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.image.get_storage", return_value=storage)
    mock_get_client = mocker.patch("sanzaru.tools.image.get_client")
    mock_get_client.return_value.responses.retrieve = mocker.AsyncMock(return_value=mock_response)

    # Mock PIL Image.open to avoid trying to parse fake PNG
    # Now called with io.BytesIO(image_bytes) instead of a file path
    mock_img = mocker.MagicMock()
    mock_img.size = (1024, 1024)
    mock_img.format = "PNG"
    mocker.patch("sanzaru.tools.image.Image.open", return_value=mock_img)

    result = await download_image("resp_test123", filename="test.png")

    assert result["filename"] == "test.png"
    assert result["size"] == (1024, 1024)
    assert result["format"] == "png"

    # Verify file was written
    output_file = tmp_reference_path / "test.png"
    assert output_file.exists()
