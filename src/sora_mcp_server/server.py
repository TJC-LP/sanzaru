# SPDX-License-Identifier: MIT
"""Sora MCP Server - FastMCP server for OpenAI Sora video generation.

This module initializes the FastMCP server and registers all tools.
Business logic is organized into submodules under tools/.
"""

from typing import Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai.types import VideoModel, VideoSeconds, VideoSize

from .config import logger
from .tools import image, reference, video

# Load environment variables for mcp run compatibility
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Sora")


# ==================== VIDEO TOOLS ====================
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
):
    return await video.sora_create_video(prompt, model, seconds, size, input_reference_filename)


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
async def sora_get_status(video_id: str):
    return await video.sora_get_status(video_id)


@mcp.tool(
    description="""Download a completed video to disk.

IMPORTANT: Only call this AFTER sora_get_status shows status='completed'.
If the video is not completed, this will fail.

The video is automatically saved to the directory configured in SORA_VIDEO_PATH.
Returns the absolute path to the downloaded file.

Parameters:
- video_id: The ID from sora_create_video or sora_remix (required)
- filename: Custom filename (optional, defaults to video_id with appropriate extension)
- variant: What to download (default: "video")
  * "video" -> MP4 video file
  * "thumbnail" -> WEBP thumbnail image
  * "spritesheet" -> JPG spritesheet of frames

Typical workflow:
1. Create: sora_create_video() -> video_id
2. Poll: sora_get_status(video_id) until status='completed'
3. Download: sora_download(video_id, filename="my_video.mp4") -> returns local file path

Returns DownloadResult with: filename, path, variant"""
)
async def sora_download(
    video_id: str,
    filename: str | None = None,
    variant: Literal["video", "thumbnail", "spritesheet"] = "video",
):
    return await video.sora_download(video_id, filename, variant)


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
async def sora_list(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc"):
    return await video.sora_list(limit, after, order)


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
async def sora_delete(video_id: str):
    return await video.sora_delete(video_id)


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
async def sora_remix(previous_video_id: str, prompt: str):
    return await video.sora_remix(previous_video_id, prompt)


# ==================== REFERENCE IMAGE TOOLS ====================
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
):
    return await reference.sora_list_references(pattern, file_type, sort_by, order, limit)


@mcp.tool(
    description="""Automatically resize a reference image to match Sora's required dimensions.

This tool prepares images for use with sora_create_video by resizing them to exact Sora dimensions.
The original image is preserved; a new resized copy is created.

Parameters:
- input_filename: Source image filename in SORA_REFERENCE_PATH (required)
- target_size: Target video size: "720x1280", "1280x720", "1024x1792", or "1792x1024" (required)
- output_filename: Custom output filename (optional, defaults to "{original_name}_{width}x{height}.png")
- resize_mode: How to handle aspect ratio (default: "crop")
  * "crop": Scale to cover target, center crop excess (no distortion, may lose edges)
  * "pad": Scale to fit inside target, add black bars (no distortion, preserves full image)
  * "rescale": Stretch/squash to exact dimensions (may distort, no cropping/padding)

Returns PrepareResult with: output_filename, original_size, target_size, resize_mode, path

Example workflow:
1. sora_list_references() -> find "photo.jpg"
2. sora_prepare_reference("photo.jpg", "1280x720", resize_mode="crop") -> "photo_1280x720.png"
3. sora_create_video(prompt="...", size="1280x720", input_reference_filename="photo_1280x720.png")"""
)
async def sora_prepare_reference(
    input_filename: str,
    target_size: VideoSize,
    output_filename: str | None = None,
    resize_mode: Literal["crop", "pad", "rescale"] = "crop",
):
    return await reference.sora_prepare_reference(input_filename, target_size, output_filename, resize_mode)


# ==================== IMAGE GENERATION TOOLS ====================
@mcp.tool(
    description="""Generate an image using OpenAI's Responses API and save to reference path.

Creates an image from a text prompt using GPT-5 or newer models with image generation capability.
Returns immediately with a response_id - use image_get_status() to poll for completion.

The image is saved to SORA_REFERENCE_PATH and can be used with sora_create_video.

Parameters:
- prompt: Text description of the image to generate (required)
- model: Model to use - "gpt-5", "gpt-4.1", etc. Default: "gpt-5"
- size: Resolution - "1024x1024", "1024x1536", "1536x1024", or "auto". Default: "auto"
- quality: "low", "medium", "high", "auto". Default: "high"
- output_format: "png", "jpeg", "webp". Default: "png"
- background: "transparent", "opaque", "auto". Default: "auto"
- previous_response_id: Optional response ID to refine previous image

Returns ImageResponse with: id (response_id), status, created_at

Typical workflow:
1. image_create(prompt="sunset over mountains") -> response_id
2. image_get_status(response_id) -> poll until status='completed'
3. image_download(response_id, filename="sunset.png") -> saves to reference path
4. sora_create_video(..., input_reference_filename="sunset.png")

Iterative refinement:
1. resp1 = image_create(prompt="a cat") -> response_id_1
2. Wait for completion
3. resp2 = image_create(prompt="make it more realistic", previous_response_id=response_id_1) -> response_id_2
4. Download response_id_2"""
)
async def image_create(
    prompt: str,
    model: str = "gpt-5",
    size: Literal["auto", "1024x1024", "1024x1536", "1536x1024"] | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["transparent", "opaque", "auto"] | None = None,
    previous_response_id: str | None = None,
):
    return await image.image_create(prompt, model, size, quality, output_format, background, previous_response_id)


@mcp.tool(
    description="""Check status and progress of image generation.

Use this to poll for completion after calling image_create.
Call repeatedly until status changes from 'queued'/'in_progress' to 'completed' or 'failed'.

Returns ImageResponse with: id, status, created_at"""
)
async def image_get_status(response_id: str):
    return await image.image_get_status(response_id)


@mcp.tool(
    description="""Download a completed generated image to reference path.

IMPORTANT: Only call AFTER image_get_status shows status='completed'.

The image is saved to SORA_REFERENCE_PATH and can immediately be used with sora_create_video.

Parameters:
- response_id: The response ID from image_create (required)
- filename: Custom filename (optional, auto-generates if not provided)

Returns ImageDownloadResult with: filename, path, size, format"""
)
async def image_download(response_id: str, filename: str | None = None):
    return await image.image_download(response_id, filename)


# ==================== SERVER ENTRYPOINT ====================
def main():
    """Run the MCP server.

    Paths are validated lazily at runtime when tools are called.
    This allows the server to work with both `uv run sora-mcp-server` and `mcp run`.
    """
    logger.info("Starting Sora MCP server over stdio")
    mcp.run()


if __name__ == "__main__":
    main()
