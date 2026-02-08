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

from ..config import get_client, logger
from ..storage import get_storage
from ..types import DownloadResult, ListResult, VideoFile, VideoSummary
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
        storage = get_storage()

        # Validate file extension (Sora supports JPEG, PNG, WEBP)
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        ext = "." + input_reference_filename.rsplit(".", 1)[-1].lower() if "." in input_reference_filename else ""
        if ext not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {ext}. Use: JPEG, PNG, or WEBP")

        # Read reference image via storage backend (handles path validation + security)
        file_content = await storage.read("reference", input_reference_filename)

        # Determine MIME type from file extension
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_type_map.get(ext, "application/octet-stream")

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
    storage = get_storage()
    client = get_client()
    suffix = suffix_for_variant(variant)

    # Auto-generate filename if not provided
    if filename is None:
        filename = generate_filename(video_id, suffix)

    # Stream video to storage backend
    async with client.with_streaming_response.videos.download_content(video_id, variant=variant) as response:
        display_path = await storage.write_stream("video", filename, response.iter_bytes())

    logger.info("Wrote %s (%s)", display_path, variant)
    return {"filename": filename, "variant": variant}


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


async def list_local_videos(
    pattern: str | None = None,
    file_type: Literal["mp4", "webm", "mov", "all"] = "all",
    sort_by: Literal["name", "size", "modified"] = "modified",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
) -> dict:
    """List locally downloaded video files.

    Args:
        pattern: Glob pattern to filter filenames (e.g., "*.mp4", "sora*")
        file_type: Filter by video type
        sort_by: Sort criterion (name, size, or modified timestamp)
        order: Sort order (asc or desc)
        limit: Maximum number of results to return

    Returns:
        Dict with "data" key containing list of VideoFile objects

    Raises:
        RuntimeError: If VIDEO_PATH not configured
    """
    storage = get_storage()

    # Map file_type to extensions
    type_to_extensions: dict[str, set[str]] = {
        "mp4": {".mp4"},
        "webm": {".webm"},
        "mov": {".mov"},
        "all": {".mp4", ".webm", ".mov"},
    }
    allowed_extensions = type_to_extensions[file_type]

    # Collect matching files via storage backend
    glob_pattern = pattern if pattern else "*"
    file_infos = await storage.list_files("video", pattern=glob_pattern, extensions=allowed_extensions)

    # Sort files
    if sort_by == "name":
        file_infos.sort(key=lambda x: x.name, reverse=(order == "desc"))
    elif sort_by == "size":
        file_infos.sort(key=lambda x: x.size_bytes, reverse=(order == "desc"))
    elif sort_by == "modified":
        file_infos.sort(key=lambda x: x.modified_timestamp, reverse=(order == "desc"))

    # Build result list
    results: list[VideoFile] = []
    for info in file_infos[:limit]:
        # Determine file type from extension
        ext = ("." + info.name.rsplit(".", 1)[-1].lower()) if "." in info.name else ""
        if ext == ".mp4":
            vid_type = "mp4"
        elif ext == ".webm":
            vid_type = "webm"
        else:
            vid_type = "mov"

        results.append(
            {
                "filename": info.name,
                "size_bytes": info.size_bytes,
                "modified_timestamp": int(info.modified_timestamp),
                "file_type": vid_type,
            }
        )

    logger.info("Listed %d local videos (pattern=%s, type=%s)", len(results), glob_pattern, file_type)
    return {"data": results}
