# SPDX-License-Identifier: MIT
"""Image generation tools using OpenAI's direct Images API.

This module provides tools that use the Images API directly (client.images.generate/edit)
rather than the Responses API with tools. Key differences:
- Supports gpt-image-1.5 model (latest, recommended)
- Synchronous - returns image immediately (no polling)
- Returns token usage information for cost tracking
"""

import base64
from typing import Literal

import anyio
from openai.types import ImageModel
from PIL import Image

from ..config import get_client, get_path, logger
from ..security import async_safe_open_file, check_not_symlink, validate_safe_path
from ..types import ImageGenerateResult
from ..utils import generate_filename

# Type aliases for clarity
ImageSize = Literal["auto", "1024x1024", "1536x1024", "1024x1536"]
ImageQuality = Literal["auto", "low", "medium", "high"]
ImageBackground = Literal["auto", "transparent", "opaque"]
ImageOutputFormat = Literal["png", "jpeg", "webp"]


async def generate_image(
    prompt: str,
    model: ImageModel = "gpt-image-1.5",
    size: ImageSize = "auto",
    quality: ImageQuality = "auto",
    background: ImageBackground = "auto",
    output_format: ImageOutputFormat = "png",
    moderation: Literal["auto", "low"] = "auto",
    filename: str | None = None,
) -> ImageGenerateResult:
    """Generate an image using OpenAI's Images API directly.

    This function uses client.images.generate() for synchronous image generation.
    Returns immediately with the generated image (no polling required).

    Args:
        prompt: Text description of the image to generate (max 32k chars for GPT models)
        model: Image generation model. Default: "gpt-image-1.5" (recommended)
        size: Image dimensions. Default: "auto"
        quality: Generation quality. Default: "auto"
        background: Background type. Default: "auto"
        output_format: Output format. Default: "png"
        moderation: Content moderation level. Default: "auto"
        filename: Custom output filename (optional, auto-generated if not provided)

    Returns:
        ImageGenerateResult with filename, path, size, format, model, and usage

    Raises:
        RuntimeError: If OPENAI_API_KEY not set or IMAGE_PATH not configured
        ValueError: If API returns error or invalid filename
    """
    client = get_client()
    reference_path = get_path("reference")

    logger.info("Generating image with %s (size=%s, quality=%s)", model, size, quality)

    # Call Images API directly - synchronous, returns immediately
    response = await client.images.generate(
        prompt=prompt,
        model=model,
        size=size,
        quality=quality,
        background=background,
        output_format=output_format,
        moderation=moderation,
        n=1,  # Generate single image
    )

    # Extract image data
    if not response.data or len(response.data) == 0:
        raise ValueError("No image data returned from API")

    image_data = response.data[0]
    if not image_data.b64_json:
        raise ValueError("No base64 image data returned (GPT models always return b64_json)")

    # Decode base64 in thread pool (CPU-bound)
    image_bytes = await anyio.to_thread.run_sync(base64.b64decode, image_data.b64_json)

    # Generate filename if not provided
    if filename is None:
        filename = generate_filename("gen", output_format, use_timestamp=True)

    # Security: validate filename and construct safe path
    output_path = validate_safe_path(reference_path, filename, allow_create=True)

    # Write image to disk asynchronously
    async with async_safe_open_file(output_path, "wb", "image file", check_symlink=False) as f:
        await f.write(image_bytes)

    # Get dimensions in thread pool (PIL operations)
    def _get_dimensions() -> tuple[tuple[int, int], str]:
        img = Image.open(output_path)
        return img.size, img.format.lower() if img.format else "unknown"

    dimensions, detected_format = await anyio.to_thread.run_sync(_get_dimensions)

    logger.info(
        "Generated image %s (%dx%d, %s) with %s",
        filename,
        dimensions[0],
        dimensions[1],
        detected_format,
        model,
    )

    return ImageGenerateResult(
        filename=filename,
        path=str(output_path),
        size=dimensions,
        format=detected_format,
        model=str(model),
        usage=response.usage,
    )


async def edit_image(
    prompt: str,
    input_images: list[str],
    model: ImageModel = "gpt-image-1.5",
    mask_filename: str | None = None,
    size: ImageSize = "auto",
    quality: ImageQuality = "auto",
    background: ImageBackground = "auto",
    output_format: ImageOutputFormat = "png",
    input_fidelity: Literal["high", "low"] | None = None,
    filename: str | None = None,
) -> ImageGenerateResult:
    """Edit images using OpenAI's Images API directly.

    This function uses client.images.edit() for image editing/composition.
    Returns immediately with the edited image (no polling required).

    Args:
        prompt: Text description of desired edits (max 32k chars for GPT models)
        input_images: List of image filenames in IMAGE_PATH (up to 16 images)
        model: Image generation model. Default: "gpt-image-1.5"
        mask_filename: Optional PNG mask with alpha channel for inpainting
        size: Output image dimensions. Default: "auto"
        quality: Generation quality. Default: "auto"
        background: Background type. Default: "auto"
        output_format: Output format. Default: "png"
        input_fidelity: Control fidelity to input images (gpt-image-1 only). Default: None
        filename: Custom output filename (optional, auto-generated if not provided)

    Returns:
        ImageGenerateResult with filename, path, size, format, model, and usage

    Raises:
        RuntimeError: If OPENAI_API_KEY not set or IMAGE_PATH not configured
        ValueError: If API returns error, invalid filename, or image not found
    """
    client = get_client()
    reference_path = get_path("reference")

    if not input_images:
        raise ValueError("At least one input image is required")

    if len(input_images) > 16:
        raise ValueError("Maximum 16 input images allowed for GPT image models")

    # Mime type mapping
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    # Load and validate input images as tuples (filename, bytes, content_type)
    image_files: list[tuple[str, bytes, str]] = []
    for img_filename in input_images:
        # Check for symlink BEFORE resolution
        original_path = reference_path / img_filename
        check_not_symlink(original_path, "reference image")

        # Validate filename and construct safe path
        img_path = validate_safe_path(reference_path, img_filename)

        # Validate file extension
        ext = img_path.suffix.lower()
        if ext not in mime_types:
            raise ValueError(f"Unsupported image format: {img_filename} (use JPEG, PNG, WEBP)")

        # Read image file with mime type
        async with async_safe_open_file(img_path, "rb", "reference image") as f:
            image_bytes = await f.read()
            image_files.append((img_filename, image_bytes, mime_types[ext]))

    # Load mask if provided (as tuple with mime type)
    mask_file: tuple[str, bytes, str] | None = None
    if mask_filename:
        original_mask_path = reference_path / mask_filename
        check_not_symlink(original_mask_path, "mask image")
        mask_path = validate_safe_path(reference_path, mask_filename)

        if mask_path.suffix.lower() != ".png":
            raise ValueError("Mask must be PNG format with alpha channel")

        async with async_safe_open_file(mask_path, "rb", "mask image") as f:
            mask_bytes = await f.read()
            mask_file = (mask_filename, mask_bytes, "image/png")

    logger.info(
        "Editing %d image(s) with %s%s",
        len(input_images),
        model,
        f" (mask: {mask_filename})" if mask_filename else "",
    )

    # Build API call arguments
    # For single image, pass bytes directly; for multiple, pass list
    image_arg = image_files[0] if len(image_files) == 1 else image_files

    # Build kwargs, omitting None values (SDK doesn't accept None for optional params)
    edit_kwargs: dict = {
        "image": image_arg,
        "prompt": prompt,
        "model": model,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "n": 1,
    }
    if mask_file:
        edit_kwargs["mask"] = mask_file
    if input_fidelity:
        edit_kwargs["input_fidelity"] = input_fidelity

    # Call Images API edit endpoint
    response = await client.images.edit(**edit_kwargs)

    # Extract image data
    if not response.data or len(response.data) == 0:
        raise ValueError("No image data returned from API")

    image_data = response.data[0]
    if not image_data.b64_json:
        raise ValueError("No base64 image data returned")

    # Decode base64 in thread pool
    image_bytes = await anyio.to_thread.run_sync(base64.b64decode, image_data.b64_json)

    # Generate filename if not provided
    if filename is None:
        filename = generate_filename("edit", output_format, use_timestamp=True)

    # Security: validate filename and construct safe path
    output_path = validate_safe_path(reference_path, filename, allow_create=True)

    # Write image to disk asynchronously
    async with async_safe_open_file(output_path, "wb", "image file", check_symlink=False) as f:
        await f.write(image_bytes)

    # Get dimensions in thread pool
    def _get_dimensions() -> tuple[tuple[int, int], str]:
        img = Image.open(output_path)
        return img.size, img.format.lower() if img.format else "unknown"

    dimensions, detected_format = await anyio.to_thread.run_sync(_get_dimensions)

    logger.info(
        "Edited image -> %s (%dx%d, %s) with %s",
        filename,
        dimensions[0],
        dimensions[1],
        detected_format,
        model,
    )

    return ImageGenerateResult(
        filename=filename,
        path=str(output_path),
        size=dimensions,
        format=detected_format,
        model=str(model),
        usage=response.usage,
    )
