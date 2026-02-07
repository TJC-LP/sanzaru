# SPDX-License-Identifier: MIT
"""Media viewer tools for MCP App-based media playback.

Provides two tools:
- view_media: Returns metadata for a media file and triggers the MCP App UI
- get_media_data: Returns base64-encoded chunks of media data (called by the MCP App)
"""

import base64
import mimetypes
from typing import Literal, TypedDict

from ..config import logger
from ..storage.factory import get_storage
from ..storage.protocol import PathType

# Media type → storage PathType mapping
MEDIA_TYPE_TO_PATH_TYPE: dict[str, PathType] = {
    "video": "video",
    "image": "reference",
    "audio": "audio",
}

MediaType = Literal["video", "audio", "image"]

DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB


class ViewMediaResult(TypedDict):
    """Result from view_media tool."""

    filename: str
    media_type: str
    size_bytes: int
    mime_type: str


class MediaDataResult(TypedDict):
    """Result from get_media_data tool."""

    data: str  # base64-encoded
    offset: int
    chunk_size: int
    total_size: int
    is_last: bool
    mime_type: str


def _resolve_path_type(media_type: MediaType) -> PathType:
    """Map a user-facing media type to a storage PathType."""
    path_type = MEDIA_TYPE_TO_PATH_TYPE.get(media_type)
    if path_type is None:
        raise ValueError(f"Unknown media_type: {media_type!r}. Must be one of: video, audio, image")
    return path_type


def _guess_mime_type(filename: str, media_type: MediaType) -> str:
    """Guess MIME type from filename, with sensible fallbacks per media type."""
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    # Fallbacks by media type
    fallbacks: dict[str, str] = {
        "video": "video/mp4",
        "audio": "audio/mpeg",
        "image": "image/png",
    }
    return fallbacks.get(media_type, "application/octet-stream")


async def view_media(media_type: MediaType, filename: str) -> ViewMediaResult:
    """Return metadata for a media file, triggering the MCP App viewer.

    Args:
        media_type: Type of media — "video", "audio", or "image"
        filename: Name of the file within the configured media directory

    Returns:
        ViewMediaResult with filename, media_type, size_bytes, and mime_type

    Raises:
        ValueError: If media_type is invalid or file does not exist
    """
    path_type = _resolve_path_type(media_type)
    storage = get_storage()

    if not await storage.exists(path_type, filename):
        raise ValueError(f"File not found: {filename}")

    info = await storage.stat(path_type, filename)
    mime_type = _guess_mime_type(filename, media_type)

    logger.info("view_media: %s/%s (%s, %d bytes)", media_type, filename, mime_type, info.size_bytes)

    return ViewMediaResult(
        filename=filename,
        media_type=media_type,
        size_bytes=info.size_bytes,
        mime_type=mime_type,
    )


async def get_media_data(
    media_type: MediaType,
    filename: str,
    offset: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> MediaDataResult:
    """Read a chunk of media data and return it as base64.

    Called by the MCP App via callServerTool to load media content in chunks.

    .. note::

        This reads the full file via ``storage.read()`` and slices in memory.
        For the local backend the OS page cache makes repeated reads fast.
        For remote backends (Databricks), prefer using the HTTP route
        ``/media/{type}/{name}`` which streams bytes directly without base64
        overhead.  This tool exists as the universal fallback that works over
        both stdio and HTTP transports.

    Args:
        media_type: Type of media — "video", "audio", or "image"
        filename: Name of the file within the configured media directory
        offset: Byte offset to start reading from (default: 0)
        chunk_size: Number of bytes to read (default: 2 MB)

    Returns:
        MediaDataResult with base64 data, offset info, and completion status

    Raises:
        ValueError: If media_type is invalid, file does not exist, or offset is negative
    """
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    path_type = _resolve_path_type(media_type)
    storage = get_storage()

    # Read entire file, then slice (storage.read returns full bytes)
    file_data = await storage.read(path_type, filename)
    total_size = len(file_data)

    # Slice the requested chunk
    chunk = file_data[offset : offset + chunk_size]
    actual_chunk_size = len(chunk)
    is_last = (offset + actual_chunk_size) >= total_size

    mime_type = _guess_mime_type(filename, media_type)
    encoded = base64.b64encode(chunk).decode("ascii")

    logger.debug(
        "get_media_data: %s/%s offset=%d chunk=%d total=%d is_last=%s",
        media_type,
        filename,
        offset,
        actual_chunk_size,
        total_size,
        is_last,
    )

    return MediaDataResult(
        data=encoded,
        offset=offset,
        chunk_size=actual_chunk_size,
        total_size=total_size,
        is_last=is_last,
        mime_type=mime_type,
    )
