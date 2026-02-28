# SPDX-License-Identifier: MIT
"""Image generation tools using OpenAI's Responses API.

This module handles image generation operations:
- Creating image generation jobs
- Checking generation status
- Downloading completed images to reference path
"""

import base64
import io
from typing import Literal

import anyio
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

from ..config import get_client, get_google_client, logger
from ..storage import get_storage
from ..types import ImageDownloadResult, ImageResponse
from ..utils import generate_filename

# ==================== HELPER FUNCTIONS ====================


def _encode_image_base64(data: bytes) -> str:
    """Encode image bytes as base64 string.

    Args:
        data: Raw image bytes

    Returns:
        Base64-encoded string (not data URL, just the base64 part)
    """
    return base64.b64encode(data).decode("utf-8")


def _get_mime_type(filename: str) -> str:
    """Get MIME type from filename extension.

    Args:
        filename: Filename string with extension

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png")
    """
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")  # Default to jpeg


async def _upload_mask_file(data: bytes, filename: str) -> str:
    """Upload mask image to OpenAI Files API.

    Args:
        data: Raw mask image bytes (PNG with alpha channel)
        filename: Original filename for the upload

    Returns:
        OpenAI file ID string

    Raises:
        ValueError: If upload fails
    """
    client = get_client()

    try:
        file_obj = await client.files.create(file=(filename, data), purpose="vision")
        return file_obj.id
    except Exception as e:
        raise ValueError(f"Failed to upload mask file: {e}") from e


# ==================== GOOGLE NANO BANANA ====================

_GOOGLE_DEFAULT_MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana 2


async def _create_image_google(
    prompt: str,
    model: str,
    aspect_ratio: str,
    filename: str | None,
) -> ImageDownloadResult:
    """Generate an image using Google Nano Banana (Gemini image generation models).

    Wraps the synchronous Google Gen AI client in a thread pool for async compatibility.
    The Google API is synchronous and returns results immediately (no polling required).

    Args:
        prompt: Text description of the image to generate
        model: Google model ID (e.g., "gemini-3.1-flash-image-preview" for Nano Banana 2)
        aspect_ratio: Image aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4")
        filename: Optional custom filename for the saved image (auto-generated if None)

    Returns:
        ImageDownloadResult with filename, pixel dimensions, and format

    Raises:
        ImportError: If google-genai package is not installed
        RuntimeError: If Google credentials are not configured
        ValueError: If generation returns no images (e.g., blocked by safety filters)
    """
    try:
        from google.genai.types import GenerateImagesConfig
    except ImportError as e:
        raise ImportError("google-genai package is required. Install with: uv add 'sanzaru[google]'") from e

    storage = get_storage()
    google_client = get_google_client()

    config = GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio=aspect_ratio,
        output_mime_type="image/png",
    )

    logger.info("Generating Nano Banana image: model=%s aspect_ratio=%s", model, aspect_ratio)

    # Wrap synchronous Google API in thread pool (network I/O — avoids blocking the event loop)
    def _call_google() -> object:
        return google_client.models.generate_images(model=model, prompt=prompt, config=config)

    response = await anyio.to_thread.run_sync(_call_google)

    if not response.generated_images:
        raise ValueError("Google Nano Banana returned no images — prompt may have been blocked by safety filters")

    image_bytes: bytes = response.generated_images[0].image.image_bytes

    if filename is None:
        filename = generate_filename("nb", "png", use_timestamp=True)

    await storage.write("reference", filename, image_bytes)

    def _get_dimensions() -> tuple[tuple[int, int], str]:
        img = Image.open(io.BytesIO(image_bytes))
        return img.size, img.format.lower() if img.format else "png"

    size, fmt = await anyio.to_thread.run_sync(_get_dimensions)

    logger.info("Nano Banana image saved: %s (%dx%d, %s)", filename, size[0], size[1], fmt)

    return {"filename": filename, "size": size, "format": fmt}


# ==================== PUBLIC API ====================


async def create_image(
    prompt: str,
    provider: Literal["openai", "google"] = "openai",
    model: str | None = None,
    aspect_ratio: str = "1:1",
    filename: str | None = None,
    tool_config: ImageGeneration | None = None,
    previous_response_id: str | None = None,
    input_images: list[str] | None = None,
    mask_filename: str | None = None,
) -> ImageResponse | ImageDownloadResult:
    """Create a new image generation job using OpenAI Responses API or Google Nano Banana.

    Args:
        prompt: Text description of image to generate (or edits to make if input_images provided)
        provider: Image provider — "openai" (default) or "google" (Nano Banana / Gemini image models)
        model: Model ID. Defaults by provider:
            - openai: "gpt-5.2" (or any mainline model)
            - google: "gemini-3.1-flash-image-preview" (Nano Banana 2, recommended)
              Other Google models: "gemini-3-pro-image-preview" (Nano Banana Pro),
              "gemini-2.5-flash-image" (Nano Banana)
        aspect_ratio: Image aspect ratio for Google provider ("1:1", "16:9", "9:16", "4:3", "3:4").
            Ignored for OpenAI (use tool_config.size instead).
        filename: Output filename for Google provider (auto-generated if None).
            Ignored for OpenAI (use download_image after polling).
        tool_config: OpenAI ImageGeneration tool configuration (size, quality, model, moderation, etc.).
            Only used for OpenAI provider.
        previous_response_id: Refine a previous OpenAI generation iteratively. Only used for OpenAI.
        input_images: List of reference image filenames from IMAGE_PATH. Only used for OpenAI.
        mask_filename: PNG mask with alpha channel for inpainting. Only used for OpenAI.

    Returns:
        - OpenAI provider: ImageResponse with {id, status, created_at} — poll with get_image_status,
          then download with download_image.
        - Google provider: ImageDownloadResult with {filename, size, format} — image is ready
          immediately, no polling required.

    Raises:
        RuntimeError: If required API credentials are not set
        ValueError: If invalid filename, path traversal, or mask without input_images
        ImportError: If google-genai is not installed and provider="google"

    Example tool_config (OpenAI):
        {
            "type": "image_generation",
            "model": "gpt-image-1.5",  # recommended (or "gpt-image-1", "gpt-image-1-mini")
            "size": "1024x1024",
            "quality": "high",
            "moderation": "low",  # or "auto"
            "input_fidelity": "high",  # or "low"
            "output_format": "png",
            "background": "transparent"
        }
    """
    if provider == "google":
        return await _create_image_google(
            prompt=prompt,
            model=model or _GOOGLE_DEFAULT_MODEL,
            aspect_ratio=aspect_ratio,
            filename=filename,
        )
    # OpenAI provider path
    openai_model = model or "gpt-5.2"
    client = get_client()
    storage = get_storage()

    # Validate mask requires input images
    if mask_filename and not input_images:
        raise ValueError("mask_filename requires input_images parameter")

    # Build or use provided tool configuration
    config: ImageGeneration = tool_config if tool_config else {"type": "image_generation"}

    # Handle mask upload if provided
    if mask_filename:
        # Validate PNG format from filename
        mask_ext = ("." + mask_filename.rsplit(".", 1)[-1].lower()) if "." in mask_filename else ""
        if mask_ext != ".png":
            raise ValueError("Mask must be PNG format with alpha channel")

        # Read mask via storage backend (handles path validation + security)
        mask_bytes = await storage.read("reference", mask_filename)

        # Upload to Files API
        mask_file_id = await _upload_mask_file(mask_bytes, mask_filename)
        config["input_image_mask"] = {"file_id": mask_file_id}

        logger.info("Uploaded mask %s as file_id %s", mask_filename, mask_file_id)

    # Build input parameter
    input_param: ResponseInputParam | str

    if input_images:
        # Structured input with images
        content_items: ResponseInputMessageContentListParam = [ResponseInputTextParam(type="input_text", text=prompt)]

        for filename in input_images:
            # Validate file extension
            ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError(f"Unsupported image format: {filename} (use JPEG, PNG, WEBP)")

            # Read image via storage backend (handles path validation + security)
            img_bytes = await storage.read("reference", filename)

            # Encode to base64 (in thread pool to avoid blocking event loop)
            base64_data = await anyio.to_thread.run_sync(_encode_image_base64, img_bytes)
            mime_type = _get_mime_type(filename)

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
            " (config provided)" if tool_config else "",
        )
    else:
        # Simple text-only input (existing behavior)
        input_param = prompt
        logger.info("Creating image from text prompt only")

    # Create response with image generation tool
    prev_resp_param: str | Omit = omit if previous_response_id is None else previous_response_id
    response = await client.responses.create(
        model=openai_model,
        input=input_param,
        tools=[config],
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
        RuntimeError: If IMAGE_PATH not configured or OPENAI_API_KEY not set
        ValueError: If image generation not found or invalid filename
    """
    storage = get_storage()

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
        # Check for error or refusal in the image generation call
        error_msg = f"Image generation not completed (status: {image_gen_call.status})"

        # Check if there's an error field
        if hasattr(image_gen_call, "error") and image_gen_call.error:
            error_msg += f"\nError: {image_gen_call.error}"

        # Check for text response explaining the issue
        text_outputs = [out for out in response.output if hasattr(out, "content")]
        if text_outputs:
            error_msg += f"\nResponse: {text_outputs[0].content if hasattr(text_outputs[0], 'content') else 'See response output'}"

        raise ValueError(error_msg)

    # Decode base64 in thread pool (CPU-bound for large images)
    image_base64 = image_gen_call.result
    image_bytes = await anyio.to_thread.run_sync(base64.b64decode, image_base64)

    # Auto-generate filename if not provided
    if filename is None:
        # Default to png (we don't have direct access to tool config used)
        output_format = "png"
        filename = generate_filename("img", output_format, use_timestamp=True)

    # Write image via storage backend (handles path validation + security)
    await storage.write("reference", filename, image_bytes)

    # Get dimensions in thread pool (PIL operations) - use in-memory bytes
    def _get_dimensions() -> tuple[tuple[int, int], str]:
        img = Image.open(io.BytesIO(image_bytes))
        return img.size, img.format.lower() if img.format else "unknown"

    size, output_format = await anyio.to_thread.run_sync(_get_dimensions)

    logger.info("Downloaded image %s to %s (%dx%d, %s)", response_id, filename, size[0], size[1], output_format)

    return {
        "filename": filename,
        "size": size,
        "format": output_format,
    }
