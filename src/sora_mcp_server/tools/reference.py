# SPDX-License-Identifier: MIT
"""Reference image management tools.

This module handles reference image operations:
- Listing available reference images
- Preparing/resizing images for video generation
"""

import os
import pathlib
from typing import Literal

from openai.types import VideoSize
from PIL import Image

from ..config import get_path, logger
from ..security import validate_safe_path
from ..types import PrepareResult, ReferenceImage


async def sora_list_references(
    pattern: str | None = None,
    file_type: Literal["jpeg", "png", "webp", "all"] = "all",
    sort_by: Literal["name", "size", "modified"] = "modified",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
) -> dict:
    """List reference images available for video generation.

    Args:
        pattern: Glob pattern to filter filenames (e.g., "*.png", "cat*")
        file_type: Filter by image type
        sort_by: Sort criterion (name, size, or modified timestamp)
        order: Sort order (asc or desc)
        limit: Maximum number of results to return

    Returns:
        Dict with "data" key containing list of ReferenceImage objects

    Raises:
        RuntimeError: If SORA_REFERENCE_PATH not configured
    """
    reference_image_path = get_path("reference")

    # Map file_type to extensions
    type_to_extensions = {
        "jpeg": {".jpg", ".jpeg"},
        "png": {".png"},
        "webp": {".webp"},
        "all": {".jpg", ".jpeg", ".png", ".webp"},
    }
    allowed_extensions = type_to_extensions[file_type]

    # Collect matching files
    glob_pattern = pattern if pattern else "*"
    files: list[tuple[pathlib.Path, os.stat_result]] = []

    for file_path in reference_image_path.glob(glob_pattern):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            # Security: ensure file is within reference_image_path
            try:
                file_path.resolve().relative_to(reference_image_path)
            except ValueError:
                continue  # Skip files outside reference path
            files.append((file_path, file_path.stat()))

    # Sort files
    if sort_by == "name":
        files.sort(key=lambda x: x[0].name, reverse=(order == "desc"))
    elif sort_by == "size":
        files.sort(key=lambda x: x[1].st_size, reverse=(order == "desc"))
    elif sort_by == "modified":
        files.sort(key=lambda x: x[1].st_mtime, reverse=(order == "desc"))

    # Build result list
    results: list[ReferenceImage] = []
    for file_path, stat in files[:limit]:
        # Determine file type
        ext = file_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            img_type = "jpeg"
        elif ext == ".png":
            img_type = "png"
        else:
            img_type = "webp"

        results.append(
            {
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "modified_timestamp": int(stat.st_mtime),
                "file_type": img_type,
            }
        )

    logger.info("Listed %d reference images (pattern=%s, type=%s)", len(results), glob_pattern, file_type)
    return {"data": results}


async def sora_prepare_reference(
    input_filename: str,
    target_size: VideoSize,
    output_filename: str | None = None,
    resize_mode: Literal["crop", "pad", "rescale"] = "crop",
) -> PrepareResult:
    """Prepare a reference image by resizing to match Sora dimensions.

    Args:
        input_filename: Source image filename (not path) in SORA_REFERENCE_PATH
        target_size: Target Sora video size
        output_filename: Optional custom output name (defaults to auto-generated)
        resize_mode: Resizing strategy - "crop" (cover + crop), "pad" (fit + letterbox), or "rescale" (stretch to fit)

    Returns:
        PrepareResult with output filename, sizes, mode, and absolute path

    Raises:
        RuntimeError: If SORA_REFERENCE_PATH not configured
        ValueError: If input file invalid or path traversal detected
    """
    reference_image_path = get_path("reference")

    # Security: validate input filename and construct safe path
    input_path = validate_safe_path(reference_image_path, input_filename)

    # Parse target dimensions from VideoSize string (e.g., "1280x720" -> (1280, 720))
    width_str, height_str = target_size.split("x")
    target_width, target_height = int(width_str), int(height_str)

    # Generate output filename if not provided
    if output_filename is None:
        input_stem = input_path.stem
        output_filename = f"{input_stem}_{target_size}.png"

    # Security: validate output filename and construct safe path
    output_path = validate_safe_path(reference_image_path, output_filename, allow_create=True)

    # Load image with Pillow
    try:
        img = Image.open(input_path)
        original_size = img.size  # (width, height)
    except FileNotFoundError as e:
        raise ValueError(f"Input image not found: {input_filename}") from e
    except PermissionError as e:
        raise ValueError(f"Permission denied reading input image: {input_filename}") from e
    except OSError as e:
        raise ValueError(f"Error reading input image: {e}") from e

    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize based on mode
    if resize_mode == "crop":
        # Scale to cover target dimensions, then center crop
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            # Image is wider than target - fit height, crop width
            new_height = target_height
            new_width = int(img.width * (target_height / img.height))
        else:
            # Image is taller than target - fit width, crop height
            new_width = target_width
            new_height = int(img.height * (target_width / img.width))

        # Resize
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        img = img.crop((left, top, right, bottom))

    elif resize_mode == "pad":
        # Scale to fit inside target dimensions, then pad with black bars
        img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

        # Create black background
        result = Image.new("RGB", (target_width, target_height), (0, 0, 0))

        # Paste resized image centered
        paste_x = (target_width - img.width) // 2
        paste_y = (target_height - img.height) // 2
        result.paste(img, (paste_x, paste_y))
        img = result
    else:  # resize_mode == "rescale"
        # Simple stretch/squash to exact dimensions (may distort)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Save as PNG with error handling
    try:
        img.save(output_path, "PNG")
    except PermissionError as e:
        raise ValueError(f"Permission denied writing output image: {output_filename}") from e
    except OSError as e:
        raise ValueError(f"Error writing output image: {e}") from e

    logger.info(
        "Prepared reference: %s -> %s (%s, %dx%d -> %dx%d)",
        input_filename,
        output_filename,
        resize_mode,
        original_size[0],
        original_size[1],
        target_width,
        target_height,
    )

    return {
        "output_filename": output_filename,
        "original_size": original_size,
        "target_size": (target_width, target_height),
        "resize_mode": resize_mode,
        "path": str(output_path),
    }
