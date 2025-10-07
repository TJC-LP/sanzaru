# SPDX-License-Identifier: MIT
import logging
import os
import pathlib
import sys
from typing import Literal, TypedDict

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from openai._types import Omit, omit
from openai.types import Video, VideoDeleteResponse, VideoModel, VideoSeconds, VideoSize


# ---------- TypedDict definitions ----------
class DownloadResult(TypedDict):
    """Result from downloading a video asset."""

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


# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # Log to stderr to avoid interfering with stdio MCP transport
)
logger = logging.getLogger("sora-mcp")


# ---------- OpenAI client (stateless) ----------
def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)


# ---------- MCP server ----------
mcp = FastMCP("Sora MCP")

# ---------- Global video download path ----------
VIDEO_DOWNLOAD_PATH: pathlib.Path | None = None

# ---------- Global reference image path ----------
REFERENCE_IMAGE_PATH: pathlib.Path | None = None


def _suffix_for_variant(variant: str) -> str:
    return {"video": "mp4", "thumbnail": "webp", "spritesheet": "jpg"}[variant]


@mcp.tool(
    description="""Create a new Sora video generation job. This starts an async job and returns immediately with a video_id.

The video is NOT ready immediately - use sora_get_status(video_id) to poll for completion.
Status will be 'queued' -> 'in_progress' -> 'completed' or 'failed'.
Once status='completed', use sora_download(video_id) to save the video to disk.

Parameters:
- prompt: Text description of the video to generate (required)
- model: "sora-2" (faster, cheaper) or "sora-2-pro" (higher quality). Default: "sora-2"
- seconds: Duration as string "4", "8", or "12" (NOT an integer). Default: varies by model
- size: Resolution as "720x1280" (portrait), "1280x720" (landscape), "1024x1792", or "1792x1024". Default: "720x1280"
- input_reference_filename: Filename of reference image in SORA_REFERENCE_PATH (e.g., "cat.png"). Use sora_list_references to find available images. Image must match target size. Supported: JPEG, PNG, WEBP. Optional.

Returns Video object with fields: id, status, progress, model, seconds, size."""
)
async def sora_create_video(
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
        RuntimeError: If OPENAI_API_KEY not set or REFERENCE_IMAGE_PATH not initialized
        ValueError: If reference image invalid or path traversal detected
    """
    if REFERENCE_IMAGE_PATH is None:
        raise RuntimeError("REFERENCE_IMAGE_PATH not initialized")

    client = get_client()

    # Convert None to omit for OpenAI SDK
    seconds_param = omit if seconds is None else seconds
    size_param = omit if size is None else size

    if input_reference_filename:
        # Security: validate filename and construct safe path
        reference_file = REFERENCE_IMAGE_PATH / input_reference_filename
        reference_file = reference_file.resolve()

        # Security: prevent path traversal - ensure resolved path is within REFERENCE_IMAGE_PATH
        if not str(reference_file).startswith(str(REFERENCE_IMAGE_PATH)):
            raise ValueError("Invalid reference filename: path traversal detected")

        # Validate file exists
        if not reference_file.exists():
            raise ValueError(f"Reference image not found: {input_reference_filename}")

        # Validate file extension (Sora supports JPEG, PNG, WEBP)
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        if reference_file.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {reference_file.suffix}. Use: JPEG, PNG, or WEBP")

        # Sora expects the input reference to match the target video size.
        with open(reference_file, "rb") as f:
            video = await client.videos.create(
                model=model,
                prompt=prompt,
                seconds=seconds_param,
                size=size_param,
                input_reference=f,
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


@mcp.tool(
    description="""Check the status and progress of a video generation job.

Use this to poll for completion after calling sora_create_video or sora_remix.
Call this repeatedly (e.g. every 5-10 seconds) until status changes from 'queued'/'in_progress' to 'completed' or 'failed'.

The returned Video object contains:
- status: "queued" | "in_progress" | "completed" | "failed"
- progress: Integer 0-100 showing completion percentage
- id: The video_id for use with other tools
- Other metadata: model, seconds, size, created_at, etc.

Typical workflow:
1. Create video with sora_create_video() -> get video_id
2. Poll with sora_get_status(video_id) until status='completed'
3. Download with sora_download(video_id)"""
)
async def sora_get_status(video_id: str) -> Video:
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


@mcp.tool(
    description="""Download a completed video to disk.

IMPORTANT: Only call this AFTER sora_get_status shows status='completed'.
If the video is not completed, this will fail.

The video is automatically saved to the directory configured in SORA_VIDEO_PATH.
Returns the absolute path to the downloaded file.

Parameters:
- video_id: The ID from sora_create_video or sora_remix (required)
- variant: What to download (default: "video")
  * "video" -> MP4 video file
  * "thumbnail" -> WEBP thumbnail image
  * "spritesheet" -> JPG spritesheet of frames

Typical workflow:
1. Create: sora_create_video() -> video_id
2. Poll: sora_get_status(video_id) until status='completed'
3. Download: sora_download(video_id) -> returns local file path"""
)
async def sora_download(
    video_id: str,
    variant: Literal["video", "thumbnail", "spritesheet"] = "video",
) -> DownloadResult:
    """Download a completed video asset to disk.

    Args:
        video_id: Video ID from sora_create_video or sora_remix
        variant: Asset type to download (video, thumbnail, or spritesheet)

    Returns:
        DownloadResult with absolute path and variant

    Raises:
        RuntimeError: If VIDEO_DOWNLOAD_PATH not initialized or OPENAI_API_KEY not set
    """
    if VIDEO_DOWNLOAD_PATH is None:
        raise RuntimeError("VIDEO_DOWNLOAD_PATH not initialized")

    client = get_client()
    content = await client.videos.download_content(video_id, variant=variant)
    suffix = _suffix_for_variant(variant)
    out_path = VIDEO_DOWNLOAD_PATH / f"{video_id}.{suffix}"
    content.write_to_file(str(out_path))
    logger.info("Wrote %s (%s)", out_path, variant)
    return {"path": str(out_path), "variant": variant}


@mcp.tool(
    description="""List all video jobs in your OpenAI account with pagination support.

Returns a paginated list of all videos (completed, in-progress, failed, etc.).
Each video summary includes: id, status, progress, created_at, model, seconds, size.

Parameters:
- limit: Max number of videos to return (default: 20, max: 100)
- after: For pagination, pass the 'last' id from previous response (optional)
- order: "desc" for newest first (default) or "asc" for oldest first

Returns:
- data: Array of video summaries
- has_more: Boolean indicating if more results exist
- last: The ID of the last video (use this as 'after' for next page)

Pagination example:
1. page1 = sora_list(limit=20) -> get page1.last
2. page2 = sora_list(limit=20, after=page1.last)
3. Continue until has_more=false"""
)
async def sora_list(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc") -> ListResult:
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


@mcp.tool(
    description="""Permanently delete a video from OpenAI's cloud storage.

WARNING: This is permanent and cannot be undone! The video will be deleted from OpenAI's servers.
This does NOT delete any local files you may have downloaded with sora_download.

Use this to:
- Clean up test videos
- Remove unwanted content
- Free up storage quota

Parameters:
- video_id: The ID of the video to delete (required)

Returns confirmation with the deleted video_id and deleted=true."""
)
async def sora_delete(video_id: str) -> VideoDeleteResponse:
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


@mcp.tool(
    description="""Create a NEW video by remixing an existing completed video with a different prompt.

This creates a brand new video generation job (with a new video_id) based on an existing video.
The original video must have status='completed' for remix to work.

Like sora_create_video, this returns immediately with a new video_id - the remix is NOT instant.
You must poll the NEW video_id with sora_get_status until it completes.

Parameters:
- previous_video_id: ID of the completed video to use as a base (required)
- prompt: New text prompt to guide the remix (required)

Returns a NEW Video object with a different video_id, status='queued', progress=0.

Typical workflow:
1. Create original: sora_create_video("a cat") -> video_id_1
2. Wait: Poll sora_get_status(video_id_1) until completed
3. Remix: sora_remix(video_id_1, "a dog") -> video_id_2 (NEW ID!)
4. Wait: Poll sora_get_status(video_id_2) until completed
5. Download: sora_download(video_id_2)"""
)
async def sora_remix(previous_video_id: str, prompt: str) -> Video:
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


@mcp.tool(
    description="""Search and list reference images available for video generation.

Use this to discover what reference images are available in the SORA_REFERENCE_PATH directory.
These images can be used with sora_create_video's input_reference_filename parameter.

The reference image must match your target video size:
- "720x1280" or "1280x720" videos -> use 720x1280 or 1280x720 images
- "1024x1792" or "1792x1024" videos -> use 1024x1792 or 1792x1024 images

Parameters:
- pattern: Glob pattern to filter filenames (e.g., "cat*.png", "*.jpg"). Default: all files
- file_type: Filter by type: "jpeg", "png", "webp", or "all". Default: "all"
- sort_by: Sort results by "name", "size", or "modified". Default: "modified"
- order: "desc" for newest/largest/Z-A first, "asc" for oldest/smallest/A-Z. Default: "desc"
- limit: Max results to return. Default: 50

Returns list of ReferenceImage objects with: filename, size_bytes, modified_timestamp, file_type.

Example workflow:
1. sora_list_references(pattern="dog*", file_type="png") -> find dog images
2. Choose "dog_1280x720.png" from results
3. sora_create_video(prompt="...", size="1280x720", input_reference_filename="dog_1280x720.png")"""
)
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
        RuntimeError: If REFERENCE_IMAGE_PATH not initialized
    """
    if REFERENCE_IMAGE_PATH is None:
        raise RuntimeError("REFERENCE_IMAGE_PATH not initialized")

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

    for file_path in REFERENCE_IMAGE_PATH.glob(glob_pattern):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            # Security: ensure file is within REFERENCE_IMAGE_PATH
            try:
                file_path.resolve().relative_to(REFERENCE_IMAGE_PATH)
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


# -------- Entrypoint --------
def main():
    from dotenv import load_dotenv

    load_dotenv()
    global VIDEO_DOWNLOAD_PATH, REFERENCE_IMAGE_PATH

    # Get and validate video download path
    video_path_str = os.getenv("SORA_VIDEO_PATH", "./sora-videos")
    video_path = pathlib.Path(video_path_str).resolve()

    if not video_path.exists():
        logger.error("Video download directory does not exist: %s", video_path)
        logger.error("Please create the directory or set SORA_VIDEO_PATH to an existing directory")
        raise RuntimeError(f"Video download directory does not exist: {video_path}")

    if not video_path.is_dir():
        logger.error("SORA_VIDEO_PATH is not a directory: %s", video_path)
        raise RuntimeError(f"SORA_VIDEO_PATH is not a directory: {video_path}")

    VIDEO_DOWNLOAD_PATH = video_path
    logger.info("Video download path: %s", VIDEO_DOWNLOAD_PATH)

    # Get and validate reference image path
    reference_path_str = os.getenv("SORA_REFERENCE_PATH", "./sora-references")
    reference_path = pathlib.Path(reference_path_str).resolve()

    if not reference_path.exists():
        logger.error("Reference image directory does not exist: %s", reference_path)
        logger.error("Please create the directory or set SORA_REFERENCE_PATH to an existing directory")
        raise RuntimeError(f"Reference image directory does not exist: {reference_path}")

    if not reference_path.is_dir():
        logger.error("SORA_REFERENCE_PATH is not a directory: %s", reference_path)
        raise RuntimeError(f"SORA_REFERENCE_PATH is not a directory: {reference_path}")

    REFERENCE_IMAGE_PATH = reference_path
    logger.info("Reference image path: %s", REFERENCE_IMAGE_PATH)
    logger.info("Starting MCP server over stdio")
    mcp.run()


if __name__ == "__main__":
    main()
