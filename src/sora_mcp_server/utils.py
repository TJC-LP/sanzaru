# SPDX-License-Identifier: MIT
"""Shared utility functions for the Sora MCP server."""

import time
from typing import Literal


def suffix_for_variant(variant: Literal["video", "thumbnail", "spritesheet"]) -> str:
    """Get the file extension for a video asset variant.

    Args:
        variant: Asset type

    Returns:
        File extension without dot (e.g., "mp4", "webp", "jpg")
    """
    return {"video": "mp4", "thumbnail": "webp", "spritesheet": "jpg"}[variant]


def generate_filename(base_id: str, suffix: str, *, use_timestamp: bool = False) -> str:
    """Generate a filename with optional timestamp.

    Args:
        base_id: Base identifier for the file (e.g., video_id, "img")
        suffix: File extension without dot (e.g., "mp4", "png")
        use_timestamp: If True, append current Unix timestamp to base_id

    Returns:
        Generated filename (e.g., "abc123.mp4" or "img_1234567890.png")
    """
    if use_timestamp:
        timestamp = int(time.time())
        return f"{base_id}_{timestamp}.{suffix}"
    return f"{base_id}.{suffix}"
