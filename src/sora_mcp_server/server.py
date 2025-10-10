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
from .descriptions import (
    CREATE_IMAGE,
    CREATE_VIDEO,
    DELETE_VIDEO,
    DOWNLOAD_IMAGE,
    DOWNLOAD_VIDEO,
    GET_IMAGE_STATUS,
    GET_VIDEO_STATUS,
    LIST_REFERENCE_IMAGES,
    LIST_VIDEOS,
    PREPARE_REFERENCE_IMAGE,
    REMIX_VIDEO,
)
from .tools import image, reference, video

# Initialize FastMCP server
mcp = FastMCP("sora-mcp-server")  # Consistent naming with repo


# ==================== VIDEO TOOLS ====================
@mcp.tool(description=CREATE_VIDEO)
async def create_video(
    prompt: str,
    model: VideoModel = "sora-2",
    seconds: VideoSeconds | None = None,
    size: VideoSize | None = None,
    input_reference_filename: str | None = None,
):
    return await video.create_video(prompt, model, seconds, size, input_reference_filename)


@mcp.tool(description=GET_VIDEO_STATUS)
async def get_video_status(video_id: str):
    return await video.get_video_status(video_id)


@mcp.tool(description=DOWNLOAD_VIDEO)
async def download_video(
    video_id: str,
    filename: str | None = None,
    variant: Literal["video", "thumbnail", "spritesheet"] = "video",
):
    return await video.download_video(video_id, filename, variant)


@mcp.tool(description=LIST_VIDEOS)
async def list_videos(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc"):
    return await video.list_videos(limit, after, order)


@mcp.tool(description=DELETE_VIDEO)
async def delete_video(video_id: str):
    return await video.delete_video(video_id)


@mcp.tool(description=REMIX_VIDEO)
async def remix_video(previous_video_id: str, prompt: str):
    return await video.remix_video(previous_video_id, prompt)


# ==================== REFERENCE IMAGE TOOLS ====================
@mcp.tool(description=LIST_REFERENCE_IMAGES)
async def list_reference_images(
    pattern: str | None = None,
    file_type: Literal["jpeg", "png", "webp", "all"] = "all",
    sort_by: Literal["name", "size", "modified"] = "modified",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
):
    return await reference.list_reference_images(pattern, file_type, sort_by, order, limit)


@mcp.tool(description=PREPARE_REFERENCE_IMAGE)
async def prepare_reference_image(
    input_filename: str,
    target_size: VideoSize,
    output_filename: str | None = None,
    resize_mode: Literal["crop", "pad", "rescale"] = "crop",
):
    return await reference.prepare_reference_image(input_filename, target_size, output_filename, resize_mode)


# ==================== IMAGE GENERATION TOOLS ====================
@mcp.tool(description=CREATE_IMAGE)
async def create_image(
    prompt: str,
    model: str = "gpt-5",
    size: Literal["auto", "1024x1024", "1024x1536", "1536x1024"] | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["transparent", "opaque", "auto"] | None = None,
    previous_response_id: str | None = None,
    input_images: list[str] | None = None,
    input_fidelity: Literal["low", "high"] | None = None,
    mask_filename: str | None = None,
):
    return await image.create_image(
        prompt,
        model,
        size,
        quality,
        output_format,
        background,
        previous_response_id,
        input_images,
        input_fidelity,
        mask_filename,
    )


@mcp.tool(description=GET_IMAGE_STATUS)
async def get_image_status(response_id: str):
    return await image.get_image_status(response_id)


@mcp.tool(description=DOWNLOAD_IMAGE)
async def download_image(response_id: str, filename: str | None = None):
    return await image.download_image(response_id, filename)


# ==================== SERVER ENTRYPOINT ====================
def main():
    """Run the MCP server.

    Paths are validated lazily at runtime when tools are called.
    This allows the server to work with both `uv run sora-mcp-server` and `mcp run`.
    """
    logger.info("Starting Sora MCP server over stdio")
    load_dotenv()  # Load environment variables at runtime
    mcp.run()


if __name__ == "__main__":
    main()
