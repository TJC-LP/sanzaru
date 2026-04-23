# SPDX-License-Identifier: MIT
"""Image generation tools using OpenAI's direct Images API.

This module provides tools that use the Images API directly (client.images.generate/edit)
rather than the Responses API with tools. Key differences from create_image:
- Synchronous — blocks until image is ready (no polling)
- Returns token usage information for cost tracking
- Does NOT support iterative refinement (no previous_response_id)

For non-blocking generation, prefer create_image (Responses API) instead.
"""

import base64
import io
from typing import Literal, cast

import anyio
from openai.types import ImageModel
from PIL import Image

from ..config import get_client, logger
from ..storage import get_storage
from ..types import ImageGenerateResult
from ..utils import generate_filename

# SDK (openai 2.32) size Literal still reflects the pre-gpt-image-2 world and
# omits every 2K and 4K option. We accept the broader set on the public
# `ImageSize` alias below and cast through this narrower alias when calling
# into the SDK. Drop both the alias and the casts once the SDK ships an
# `ImageGenerateParams` that includes gpt-image-2's documented sizes.
_SDKImageSize = Literal["1024x1024", "1024x1536", "1536x1024", "auto"]

# Public size alias. Covers the "popular sizes" documented in OpenAI's
# gpt-image-2 cookbook (April 2026). The API actually accepts any resolution
# that satisfies: max edge 3840px, multiples of 16px, ratio ≤3:1,
# 655,360 ≤ pixels ≤ 8,294,400.
#
# OpenAI's own guidance: 2560x1440 is the reliability ceiling; anything above
# that (3840x2160 and friends) should be treated as experimental — results
# get more variable and the client cost (memory, decode time) climbs fast.
ImageSize = Literal[
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "2560x1440",
    "1440x2560",
    "3840x2160",
    "2160x3840",
]
ImageQuality = Literal["auto", "low", "medium", "high"]
ImageBackground = Literal["auto", "transparent", "opaque"]
ImageOutputFormat = Literal["png", "jpeg", "webp"]

# Models that restrict some parameters. gpt-image-2 processes inputs at high
# fidelity automatically (no `input_fidelity` knob) and does not support
# transparent backgrounds.
_GPT_IMAGE_2 = "gpt-image-2"


async def generate_image(
    prompt: str,
    model: str = _GPT_IMAGE_2,
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
        model: Image generation model. Default: "gpt-image-2" (state-of-the-art)
        size: Image dimensions. Default: "auto"
        quality: Generation quality. Default: "auto"
        background: Background type. Default: "auto" (transparent unsupported on gpt-image-2)
        output_format: Output format. Default: "png"
        moderation: Content moderation level. Default: "auto"
        filename: Custom output filename (optional, auto-generated if not provided)

    Returns:
        ImageGenerateResult with filename, path, size, format, model, and usage

    Raises:
        RuntimeError: If OPENAI_API_KEY not set or IMAGE_PATH not configured
        ValueError: If API returns error, invalid filename, or transparent background
            is requested with gpt-image-2 (unsupported — use gpt-image-1.5)
    """
    if model == _GPT_IMAGE_2 and background == "transparent":
        raise ValueError(
            "gpt-image-2 does not support transparent backgrounds. Use gpt-image-1.5 for transparent output."
        )

    client = get_client()
    storage = get_storage()

    logger.info("Generating image with %s (size=%s, quality=%s)", model, size, quality)

    # SDK 2.32's `ImageModel` Literal does not yet include `gpt-image-2` and
    # its `size` Literal does not include any 2K/4K option. The API accepts
    # both at runtime; the casts here satisfy the type checker. Drop them
    # once a newer SDK widens `ImageModel` and `ImageGenerateParams.size`.
    response = await client.images.generate(
        prompt=prompt,
        model=cast(ImageModel, model),
        size=cast(_SDKImageSize, size),
        quality=quality,
        background=background,
        output_format=output_format,
        moderation=moderation,
        n=1,
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

    # Write image via storage backend
    await storage.write("reference", filename, image_bytes)

    # Get dimensions in thread pool (PIL operations)
    def _get_dimensions() -> tuple[tuple[int, int], str]:
        img = Image.open(io.BytesIO(image_bytes))
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
        size=dimensions,
        format=detected_format,
        model=str(model),
        usage=response.usage,
    )


async def edit_image(
    prompt: str,
    input_images: list[str],
    model: str = _GPT_IMAGE_2,
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
        model: Image generation model. Default: "gpt-image-2" (state-of-the-art)
        mask_filename: Optional PNG mask with alpha channel for inpainting
        size: Output image dimensions. Default: "auto"
        quality: Generation quality. Default: "auto"
        background: Background type. Default: "auto" (transparent unsupported on gpt-image-2)
        output_format: Output format. Default: "png"
        input_fidelity: Control fidelity to input images. Only supported on
            gpt-image-1 / gpt-image-1.5; ignored for gpt-image-2 (always high).
        filename: Custom output filename (optional, auto-generated if not provided)

    Returns:
        ImageGenerateResult with filename, path, size, format, model, and usage

    Raises:
        RuntimeError: If OPENAI_API_KEY not set or IMAGE_PATH not configured
        ValueError: If API returns error, invalid filename, image not found, or
            transparent background is requested with gpt-image-2.
    """
    if model == _GPT_IMAGE_2 and background == "transparent":
        raise ValueError(
            "gpt-image-2 does not support transparent backgrounds. Use gpt-image-1.5 for transparent output."
        )

    client = get_client()
    storage = get_storage()

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
        # Validate extension from filename string
        ext = "." + img_filename.rsplit(".", 1)[-1].lower() if "." in img_filename else ""
        if ext not in mime_types:
            raise ValueError(f"Unsupported image format: {img_filename} (use JPEG, PNG, WEBP)")

        # Read image file via storage backend (handles path validation + security)
        image_bytes = await storage.read("reference", img_filename)
        image_files.append((img_filename, image_bytes, mime_types[ext]))

    # Load mask if provided (as tuple with mime type)
    mask_file: tuple[str, bytes, str] | None = None
    if mask_filename:
        # Validate extension
        mask_ext = "." + mask_filename.rsplit(".", 1)[-1].lower() if "." in mask_filename else ""
        if mask_ext != ".png":
            raise ValueError("Mask must be PNG format with alpha channel")

        mask_bytes = await storage.read("reference", mask_filename)
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

    # Build kwargs, omitting None values (SDK doesn't accept None for optional params).
    # SDK 2.32's `ImageModel` Literal omits `gpt-image-2` and its size Literal
    # omits 2K/4K options. The API accepts both at runtime; drop the casts
    # once a newer SDK catches up.
    edit_kwargs: dict = {
        "image": image_arg,
        "prompt": prompt,
        "model": cast(ImageModel, model),
        "size": cast(_SDKImageSize, size),
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "n": 1,
    }
    if mask_file:
        edit_kwargs["mask"] = mask_file
    if input_fidelity:
        # gpt-image-2 always processes inputs at high fidelity and rejects the flag.
        if model == _GPT_IMAGE_2:
            logger.debug("Ignoring input_fidelity=%s for gpt-image-2 (always high)", input_fidelity)
        else:
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

    # Write image via storage backend
    await storage.write("reference", filename, image_bytes)

    # Get dimensions in thread pool
    def _get_dimensions() -> tuple[tuple[int, int], str]:
        img = Image.open(io.BytesIO(image_bytes))
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
        size=dimensions,
        format=detected_format,
        model=str(model),
        usage=response.usage,
    )
