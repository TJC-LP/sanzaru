# SPDX-License-Identifier: MIT
"""Image generation tools using OpenAI's Responses API.

This module handles image generation operations:
- Creating image generation jobs
- Checking generation status
- Downloading completed images to reference path
"""

import base64
from typing import Literal

from openai._types import Omit, omit
from openai.types.responses.response_output_item import ImageGenerationCall
from PIL import Image

from ..config import get_client, get_path, logger
from ..security import validate_safe_path
from ..types import ImageDownloadResult, ImageResponse
from ..utils import generate_filename


async def image_create(
    prompt: str,
    model: str = "gpt-5",
    size: Literal["auto", "1024x1024", "1024x1536", "1536x1024"] | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["transparent", "opaque", "auto"] | None = None,
    previous_response_id: str | None = None,
) -> ImageResponse:
    """Create a new image generation job using Responses API.

    Args:
        prompt: Text description of image to generate
        model: Model to use (gpt-5, gpt-4.1, etc.)
        size: Output resolution
        quality: Image quality level
        output_format: File format for output
        background: Background transparency setting
        previous_response_id: Optional ID to refine previous generation

    Returns:
        ImageResponse with response ID, status, and creation timestamp

    Raises:
        RuntimeError: If OPENAI_API_KEY not set
    """
    client = get_client()

    # Build image generation tool configuration
    tool_config: dict = {"type": "image_generation"}

    if size is not None:
        tool_config["size"] = size
    if quality is not None:
        tool_config["quality"] = quality
    if output_format is not None:
        tool_config["output_format"] = output_format
    if background is not None:
        tool_config["background"] = background

    # Create response with image generation tool
    prev_resp_param: str | Omit = omit if previous_response_id is None else previous_response_id
    response = await client.responses.create(
        model=model,
        input=prompt,
        tools=[tool_config],
        previous_response_id=prev_resp_param,
        background=True,
    )

    logger.info(
        "Started image generation %s (%s)%s",
        response.id,
        response.status,
        f" from {previous_response_id}" if previous_response_id else "",
    )

    return {
        "id": response.id,
        "status": str(response.status) if response.status else "unknown",
        "created_at": response.created_at,
    }


async def image_get_status(response_id: str) -> ImageResponse:
    """Get current status of an image generation job.

    Args:
        response_id: The response ID from image_create

    Returns:
        ImageResponse with current status and metadata

    Raises:
        RuntimeError: If OPENAI_API_KEY not set
    """
    client = get_client()
    response = await client.responses.retrieve(response_id)

    return {
        "id": response.id,
        "status": str(response.status) if response.status else "unknown",
        "created_at": response.created_at,
    }


async def image_download(
    response_id: str,
    filename: str | None = None,
) -> ImageDownloadResult:
    """Download a completed generated image to disk.

    Args:
        response_id: Response ID from image_create
        filename: Optional custom filename

    Returns:
        ImageDownloadResult with filename, path, dimensions, and format

    Raises:
        RuntimeError: If SORA_REFERENCE_PATH not configured or OPENAI_API_KEY not set
        ValueError: If image generation not found or invalid filename
    """
    reference_image_path = get_path("reference")

    client = get_client()
    response = await client.responses.retrieve(response_id)

    # Find image generation call in output
    image_gen_call: ImageGenerationCall | None = None
    for output in response.output:
        if output.type == "image_generation_call":
            image_gen_call = output
            break

    if image_gen_call is None:
        raise ValueError(f"No image generation found in response {response_id}")

    if image_gen_call.result is None:
        raise ValueError(f"Image generation not completed (status: {image_gen_call.status})")

    # Decode base64 image
    image_base64 = image_gen_call.result
    image_bytes = base64.b64decode(image_base64)

    # Auto-generate filename if not provided
    if filename is None:
        # Default to png (we don't have direct access to tool config used)
        output_format = "png"
        filename = generate_filename("img", output_format, use_timestamp=True)

    # Security: validate filename and construct safe path
    output_path = validate_safe_path(reference_image_path, filename, allow_create=True)

    # Write image to disk with error handling
    try:
        with open(output_path, "wb") as f:
            f.write(image_bytes)
    except PermissionError as e:
        raise ValueError(f"Permission denied writing image: {filename}") from e
    except OSError as e:
        raise ValueError(f"Error writing image: {e}") from e

    # Get image dimensions using PIL
    try:
        img = Image.open(output_path)
        size = img.size  # (width, height)
        output_format = img.format.lower() if img.format else "unknown"
    except OSError as e:
        raise ValueError(f"Error reading saved image dimensions: {e}") from e

    logger.info("Downloaded image %s to %s (%dx%d, %s)", response_id, filename, size[0], size[1], output_format)

    return {
        "filename": filename,
        "path": str(output_path),
        "size": size,
        "format": output_format,
    }
