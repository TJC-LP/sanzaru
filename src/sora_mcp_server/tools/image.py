# SPDX-License-Identifier: MIT
"""Image generation tools using OpenAI's Responses API.

This module handles image generation operations:
- Creating image generation jobs
- Checking generation status
- Downloading completed images to reference path
"""

import base64
import pathlib
from typing import Literal

from openai._types import Omit, omit
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.tool_param import ImageGeneration
from PIL import Image

from ..config import get_client, get_path, logger
from ..security import check_not_symlink, validate_safe_path
from ..types import ImageDownloadResult, ImageResponse
from ..utils import generate_filename

# ==================== HELPER FUNCTIONS ====================


def _encode_image_base64(image_path: pathlib.Path) -> str:
    """Read image file and encode as base64 string.

    Args:
        image_path: Absolute path to validated image file

    Returns:
        Base64-encoded string (not data URL, just the base64 part)

    Raises:
        ValueError: If file cannot be read (with context)
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")
    except FileNotFoundError as e:
        raise ValueError(f"Image file not found: {image_path.name}") from e
    except PermissionError as e:
        raise ValueError(f"Permission denied reading image: {image_path.name}") from e
    except OSError as e:
        raise ValueError(f"Error reading image file: {e}") from e


def _get_mime_type(image_path: pathlib.Path) -> str:
    """Get MIME type from file extension.

    Args:
        image_path: Path with validated extension

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png")
    """
    ext = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")  # Default to jpeg


async def _upload_mask_file(mask_path: pathlib.Path) -> str:
    """Upload mask image to OpenAI Files API.

    Args:
        mask_path: Absolute path to validated PNG mask with alpha channel

    Returns:
        OpenAI file ID string

    Raises:
        ValueError: If upload fails
    """
    client = get_client()

    try:
        with open(mask_path, "rb") as f:
            file_obj = await client.files.create(file=f, purpose="vision")
        return file_obj.id
    except Exception as e:
        raise ValueError(f"Failed to upload mask file: {e}") from e


# ==================== PUBLIC API ====================


async def create_image(
    prompt: str,
    model: str = "gpt-5",
    size: Literal["auto", "1024x1024", "1024x1536", "1536x1024"] | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["transparent", "opaque", "auto"] | None = None,
    previous_response_id: str | None = None,
    input_images: list[str] | None = None,
    input_fidelity: Literal["low", "high"] | None = None,
    mask_filename: str | None = None,
) -> ImageResponse:
    """Create a new image generation job using Responses API.

    Args:
        prompt: Text description of image to generate (or edits to make if input_images provided)
        model: Model to use (gpt-5, gpt-4.1, etc.)
        size: Output resolution
        quality: Image quality level
        output_format: File format for output
        background: Background transparency setting
        previous_response_id: Optional ID to refine previous generation
        input_images: Optional list of reference image filenames from REFERENCE_IMAGE_PATH
        input_fidelity: Detail preservation level ("low" or "high") when using input_images
        mask_filename: Optional PNG mask with alpha channel for inpainting

    Returns:
        ImageResponse with response ID, status, and creation timestamp

    Raises:
        RuntimeError: If OPENAI_API_KEY not set or REFERENCE_IMAGE_PATH not configured
        ValueError: If invalid filename, path traversal, or mask without input_images
    """
    client = get_client()
    reference_path = get_path("reference")

    # Validate mask requires input images
    if mask_filename and not input_images:
        raise ValueError("mask_filename requires input_images parameter")

    # Build image generation tool configuration
    tool_config: ImageGeneration = {"type": "image_generation"}

    if size is not None:
        tool_config["size"] = size
    if quality is not None:
        tool_config["quality"] = quality
    if output_format is not None:
        tool_config["output_format"] = output_format
    if background is not None:
        tool_config["background"] = background
    if input_fidelity is not None:
        tool_config["input_fidelity"] = input_fidelity

    # Handle mask upload if provided
    if mask_filename:
        mask_path = validate_safe_path(reference_path, mask_filename)
        check_not_symlink(mask_path, "mask image")

        # Validate PNG format
        if mask_path.suffix.lower() != ".png":
            raise ValueError("Mask must be PNG format with alpha channel")

        # Upload to Files API
        mask_file_id = await _upload_mask_file(mask_path)
        tool_config["input_image_mask"] = {"file_id": mask_file_id}

        logger.info("Uploaded mask %s as file_id %s", mask_filename, mask_file_id)

    # Build input parameter
    input_param: ResponseInputParam | str

    if input_images:
        # Structured input with images
        content_items: ResponseInputMessageContentListParam = [ResponseInputTextParam(type="input_text", text=prompt)]

        for filename in input_images:
            # Validate filename and construct path
            img_path = validate_safe_path(reference_path, filename)
            check_not_symlink(img_path, "reference image")

            # Validate file extension
            ext = img_path.suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                raise ValueError(f"Unsupported image format: {filename} (use JPEG, PNG, WEBP)")

            # Encode to base64
            base64_data = _encode_image_base64(img_path)
            mime_type = _get_mime_type(img_path)

            # Add to content items
            image_item: ResponseInputImageParam = {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_data}",
                "detail": "auto",
            }
            content_items.append(image_item)

        # Build properly typed message
        message: EasyInputMessageParam = {"role": "user", "content": content_items}
        input_param = [message]

        logger.info(
            "Creating image with %d reference image(s)%s",
            len(input_images),
            f" (fidelity={input_fidelity})" if input_fidelity else "",
        )
    else:
        # Simple text-only input (existing behavior)
        input_param = prompt
        logger.info("Creating image from text prompt only")

    # Create response with image generation tool
    prev_resp_param: str | Omit = omit if previous_response_id is None else previous_response_id
    response = await client.responses.create(
        model=model,
        input=input_param,
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


async def get_image_status(response_id: str) -> ImageResponse:
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


async def download_image(
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
        RuntimeError: If REFERENCE_IMAGE_PATH not configured or OPENAI_API_KEY not set
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
