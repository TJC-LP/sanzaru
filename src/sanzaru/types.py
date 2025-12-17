# SPDX-License-Identifier: MIT
"""Type definitions for Sora MCP server.

This module contains TypedDict and Pydantic model definitions used across the server.
"""

from typing import Literal, TypedDict

from openai.types import VideoModel, VideoSeconds, VideoSize
from openai.types.images_response import Usage as ImageUsage
from pydantic import BaseModel


class DownloadResult(TypedDict):
    """Result from downloading a video asset."""

    filename: str
    path: str
    variant: Literal["video", "thumbnail", "spritesheet"]


class VideoSummary(TypedDict):
    """Summary of a video for list results."""

    id: str
    status: Literal["queued", "in_progress", "completed", "failed"]
    created_at: int
    seconds: VideoSeconds
    size: VideoSize
    model: VideoModel
    progress: int


class ListResult(TypedDict):
    """Paginated list of videos."""

    data: list[VideoSummary]
    has_more: bool | None
    last: str | None


class ReferenceImage(TypedDict):
    """Metadata for a reference image file."""

    filename: str
    size_bytes: int
    modified_timestamp: int
    file_type: str


class PrepareResult(TypedDict):
    """Result from preparing a reference image."""

    output_filename: str
    original_size: tuple[int, int]
    target_size: tuple[int, int]
    resize_mode: str
    path: str


class ImageResponse(TypedDict):
    """Response from creating an image generation job."""

    id: str
    status: str
    created_at: float


class ImageDownloadResult(TypedDict):
    """Result from downloading a generated image."""

    filename: str
    path: str
    size: tuple[int, int]
    format: str


class ImageGenerateResult(BaseModel):
    """Result from generating an image via Images API."""

    filename: str
    path: str
    size: tuple[int, int]
    format: str
    model: str
    usage: ImageUsage | None = None
