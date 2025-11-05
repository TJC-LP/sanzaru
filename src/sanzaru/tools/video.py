# SPDX-License-Identifier: MIT
"""Video generation tools using OpenAI's Sora API.

This module contains all video-related operations:
- Creating video generation jobs
- Checking status and progress
- Downloading completed videos
- Listing and managing videos
- Remixing existing videos
"""

from typing import Literal

from openai._types import Omit, omit
from openai.types import Video, VideoDeleteResponse, VideoModel, VideoSeconds, VideoSize

from ..config import get_client, get_path, logger
from ..security import async_safe_open_file, validate_safe_path
from ..types import DownloadResult, ListResult, VideoSummary
from ..utils import generate_filename, suffix_for_variant


async def create_video(
    prompt: str,
    model: VideoModel = "sora-2",
    seconds: VideoSeconds | None = None,
    size: VideoSize | None = None,
    input_reference_filename: str | None = None,
) -> Video:
    """Create a new video generation job.

    Args:
        prompt: Text description of video content
        model: Video generation model to use
        seconds: Duration as string literal "4", "8", or "12"
        size: Output resolution (width x height)
        input_reference_filename: Filename of reference image (not full path)

    Returns:
        Video object with job details (id, status, progress)

    Raises:
        RuntimeError: If OPENAI_API_KEY not set or IMAGE_PATH not configured
        ValueError: If reference image invalid or path traversal detected
    """
    client = get_client()

    # Convert None to omit for OpenAI SDK
    seconds_param = omit if seconds is None else seconds
    size_param = omit if size is None else size

    if input_reference_filename:
        # Get reference image path at runtime
        reference_image_path = get_path("reference")

        # Security: validate filename and construct safe path
        reference_file = validate_safe_path(reference_image_path, input_reference_filename)

        # Validate file extension (Sora supports JPEG, PNG, WEBP)
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        if reference_file.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {reference_file.suffix}. Use: JPEG, PNG, or WEBP")

        # Open and read reference image asynchronously with security checks
        async with async_safe_open_file(reference_file, "rb", "reference image") as f:
            file_content = await f.read()

            # Determine MIME type from file extension
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            # Defensive fallback - should be unreachable due to extension validation above (line 60-62)
            mime_type = mime_type_map.get(reference_file.suffix.lower(), "application/octet-stream")

            # Pass as tuple (filename, bytes, content_type) so SDK can detect MIME type
            video = await client.videos.create(
                model=model,
                prompt=prompt,
                seconds=seconds_param,
                size=size_param,
                input_reference=(input_reference_filename, file_content, mime_type),
            )
        logger.info("Started job %s (%s) with reference: %s", video.id, video.status, input_reference_filename)
    else:
        video = await client.videos.create(
            model=model,
            prompt=prompt,
            seconds=seconds_param,
            size=size_param,
        )
        logger.info("Started job %s (%s)", video.id, video.status)

    return video


async def get_video_status(video_id: str) -> Video:
    """Get current status and progress of a video job.

    Args:
        video_id: The video ID from sora_create_video or sora_remix

    Returns:
        Video object with current status, progress, and metadata

    Raises:
        RuntimeError: If OPENAI_API_KEY not set
    """
    client = get_client()
    video = await client.videos.retrieve(video_id)
    return video


async def download_video(
    video_id: str,
    filename: str | None = None,
    variant: Literal["video", "thumbnail", "spritesheet"] = "video",
) -> DownloadResult:
    """Download a completed video asset to disk.

    Args:
        video_id: Video ID from sora_create_video or sora_remix
        filename: Optional custom filename
        variant: Asset type to download (video, thumbnail, or spritesheet)

    Returns:
        DownloadResult with filename, absolute path, and variant

    Raises:
        RuntimeError: If VIDEO_PATH not configured or OPENAI_API_KEY not set
        ValueError: If invalid filename or path traversal detected
    """
    video_download_path = get_path("video")
    client = get_client()
    suffix = suffix_for_variant(variant)

    # Auto-generate filename if not provided
    if filename is None:
        filename = generate_filename(video_id, suffix)

    # Security: validate filename and construct safe path
    out_path = validate_safe_path(video_download_path, filename, allow_create=True)

    # Stream video to disk asynchronously
    async with (
        client.with_streaming_response.videos.download_content(video_id, variant=variant) as response,
        async_safe_open_file(out_path, "wb", "video file", check_symlink=False) as f,
    ):
        async for chunk in response.iter_bytes():
            await f.write(chunk)

    logger.info("Wrote %s (%s)", out_path, variant)
    return {"filename": filename, "path": str(out_path), "variant": variant}


async def list_videos(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc") -> ListResult:
    """List video jobs with pagination.

    Args:
        limit: Maximum videos to return (default 20)
        after: Cursor for pagination (ID of last item from previous page)
        order: Sort order by creation time (desc=newest first, asc=oldest first)

    Returns:
        ListResult with data array, has_more flag, and last ID for pagination

    Raises:
        RuntimeError: If OPENAI_API_KEY not set
    """
    client = get_client()
    # Convert None to omit for OpenAI SDK (omit = field not sent in API request)
    after_param: str | Omit = omit if after is None else after
    page = await client.videos.list(limit=limit, after=after_param, order=order)
    items: list[VideoSummary] = []
    for v in page.data:
        items.append(
            {
                "id": v.id,
                "status": v.status,
                "created_at": v.created_at,
                "seconds": v.seconds,
                "size": v.size,
                "model": v.model,
                "progress": v.progress,
            }
        )
    return {"data": items, "has_more": page.has_more, "last": items[-1]["id"] if items else None}


async def delete_video(video_id: str) -> VideoDeleteResponse:
    """Permanently delete a video from OpenAI storage.

    Args:
        video_id: Video ID to delete

    Returns:
        VideoDeleteResponse with deleted=true confirmation

    Raises:
        RuntimeError: If OPENAI_API_KEY not set
    """
    client = get_client()
    resp = await client.videos.delete(video_id)
    logger.info("Deleted %s", video_id)
    return resp


async def remix_video(previous_video_id: str, prompt: str) -> Video:
    """Create a new video by remixing an existing one.

    Args:
        previous_video_id: ID of completed video to remix
        prompt: New prompt to guide the remix

    Returns:
        NEW Video object with different video_id and status='queued'

    Raises:
        RuntimeError: If OPENAI_API_KEY not set
    """
    client = get_client()
    video = await client.videos.remix(previous_video_id, prompt=prompt)
    logger.info("Started remix %s (from %s)", video.id, previous_video_id)
    return video
