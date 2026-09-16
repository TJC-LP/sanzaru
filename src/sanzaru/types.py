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
    variant: Literal["video", "thumbnail", "spritesheet"]


class VideoSummary(TypedDict):
    """Summary of a video for list results."""

    id: str
    status: Literal["queued", "in_progress", "completed", "failed"]
    created_at: int
    seconds: str | VideoSeconds
    size: VideoSize
    model: VideoModel
    progress: int


class ListResult(TypedDict):
    """Paginated list of videos."""

    data: list[VideoSummary]
    has_more: bool | None
    last: str | None


class VideoFile(TypedDict):
    """Metadata for a local video file."""

    filename: str
    size_bytes: int
    modified_timestamp: int
    file_type: str


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


class ImageResponse(TypedDict):
    """Response from creating an image generation job."""

    id: str
    status: str
    created_at: float


class ImageDownloadResult(TypedDict):
    """Result from downloading a generated image."""

    filename: str
    size: tuple[int, int]
    format: str


class ImageGenerateResult(BaseModel):
    """Result from generating an image via Images API."""

    filename: str
    size: tuple[int, int]
    format: str
    model: str
    usage: ImageUsage | None = None


class WaitJob(TypedDict):
    """One job's outcome from `wait_for`."""

    id: str
    kind: Literal["video", "image"]
    status: str
    """Last-seen status: terminal when `done`, the in-flight one when `timed_out`,
    "unknown" if the job was never successfully fetched."""
    done: bool
    """Reached a terminal state (completed, failed, cancelled, incomplete) or
    errored; either way there is nothing left to wait for."""
    timed_out: bool
    """Still running at the deadline. The job continues server-side; waiting
    again on the same id resumes."""
    progress: int | None
    """0-100 for video jobs; None for images (the Responses API reports none)."""
    error: str | None
    """A non-retryable API error for this id (e.g. an unknown id), when any."""
    download: DownloadResult | ImageDownloadResult | None
    """Set when `download=True` and the job completed within the deadline."""


class WaitResult(TypedDict):
    """Result of `wait_for`: every job, in input order."""

    jobs: list[WaitJob]
    all_done: bool
    timed_out: bool
    timeout_s: float
