# SPDX-License-Identifier: MIT
"""Reference image management tools.

This module handles reference image operations:
- Listing available reference images
- Preparing/resizing images for video generation
"""

import os
import pathlib
from typing import Literal

import anyio
from openai.types import VideoSize
from PIL import Image

from ..config import get_path, logger
from ..security import validate_safe_path
from ..types import PrepareResult, ReferenceImage

# ==================== Helper Functions for Image Processing ====================


def parse_video_dimensions(size: VideoSize) -> tuple[int, int]:
    """Parse VideoSize string to width/height tuple.

    Args:
        size: VideoSize string like "1280x720"

    Returns:
        Tuple of (width, height)

    Example:
        >>> parse_video_dimensions("1920x1080")
        (1920, 1080)
    """
    width_str, height_str = size.split("x")
    return int(width_str), int(height_str)


def load_and_convert_image(path: pathlib.Path, filename: str) -> Image.Image:
    """Load image and convert to RGB if needed.

    Args:
        path: Path to image file
        filename: Original filename for error messages

    Returns:
        PIL Image in RGB mode

    Raises:
        ValueError: If file can't be loaded
    """
    try:
        img: Image.Image = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except FileNotFoundError as e:
        raise ValueError(f"Input image not found: {filename}") from e
    except PermissionError as e:
        raise ValueError(f"Permission denied reading input image: {filename}") from e
    except OSError as e:
        raise ValueError(f"Error reading input image: {e}") from e


def resize_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image using crop strategy (cover target, center crop excess).

    This strategy scales the image to cover the target dimensions completely,
    then crops the excess from the center. No distortion occurs, but edges may be lost.

    Args:
        img: Source PIL Image
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Resized and cropped PIL Image
    """
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
    return img.crop((left, top, right, bottom))


def resize_pad(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image using pad strategy (fit inside, add black bars).

    This strategy scales the image to fit inside the target dimensions,
    then adds black letterbox bars. No distortion occurs, full image is preserved.

    Args:
        img: Source PIL Image
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Resized and padded PIL Image
    """
    img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

    result = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - img.width) // 2
    paste_y = (target_height - img.height) // 2
    result.paste(img, (paste_x, paste_y))
    return result


def resize_rescale(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image using rescale strategy (stretch to exact dimensions).

    This strategy stretches or squashes the image to exactly match the target dimensions.
    May cause distortion if aspect ratios don't match.

    Args:
        img: Source PIL Image
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Resized PIL Image
    """
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def save_image(img: Image.Image, path: pathlib.Path, filename: str) -> None:
    """Save PIL Image to disk as PNG.

    Args:
        img: PIL Image to save
        path: Destination path
        filename: Original filename for error messages

    Raises:
        ValueError: If save fails
    """
    try:
        img.save(path, "PNG")
    except PermissionError as e:
        raise ValueError(f"Permission denied writing output image: {filename}") from e
    except OSError as e:
        raise ValueError(f"Error writing output image: {e}") from e


# Mapping of resize mode strings to functions
RESIZE_STRATEGIES = {
    "crop": resize_crop,
    "pad": resize_pad,
    "rescale": resize_rescale,
}


async def list_reference_images(
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
        RuntimeError: If IMAGE_PATH not configured
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


async def prepare_reference_image(
    input_filename: str,
    target_size: VideoSize,
    output_filename: str | None = None,
    resize_mode: Literal["crop", "pad", "rescale"] = "crop",
) -> PrepareResult:
    """Prepare a reference image by resizing to match Sora dimensions.

    Args:
        input_filename: Source image filename (not path) in IMAGE_PATH
        target_size: Target Sora video size
        output_filename: Optional custom output name (defaults to auto-generated)
        resize_mode: Resizing strategy - "crop" (cover + crop), "pad" (fit + letterbox), or "rescale" (stretch to fit)

    Returns:
        PrepareResult with output filename, sizes, mode, and absolute path

    Raises:
        RuntimeError: If IMAGE_PATH not configured
        ValueError: If input file invalid or path traversal detected
    """
    reference_image_path = get_path("reference")

    # Validate paths (fast, stays synchronous)
    input_path = validate_safe_path(reference_image_path, input_filename)
    target_width, target_height = parse_video_dimensions(target_size)

    # Generate output filename if needed
    if output_filename is None:
        output_filename = f"{input_path.stem}_{target_size}.png"

    output_path = validate_safe_path(reference_image_path, output_filename, allow_create=True)

    # Wrap entire PIL workflow in thread pool (CPU-intensive operations)
    def _process_image() -> tuple[tuple[int, int], str]:
        """Synchronous image processing in worker thread."""
        # Load and convert image
        img = load_and_convert_image(input_path, input_filename)
        original_size = img.size

        # Apply resize strategy
        resize_fn = RESIZE_STRATEGIES[resize_mode]
        img = resize_fn(img, target_width, target_height)

        # Save result
        save_image(img, output_path, output_filename)

        return original_size, output_filename

    # Run in thread pool - multiple image preps can now run concurrently
    original_size, final_filename = await anyio.to_thread.run_sync(_process_image)

    logger.info(
        "Prepared reference: %s -> %s (%s, %dx%d -> %dx%d)",
        input_filename,
        final_filename,
        resize_mode,
        original_size[0],
        original_size[1],
        target_width,
        target_height,
    )

    return {
        "output_filename": final_filename,
        "original_size": original_size,
        "target_size": (target_width, target_height),
        "resize_mode": resize_mode,
        "path": str(output_path),
    }
