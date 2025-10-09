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
    IMAGE_CREATE,
    IMAGE_DOWNLOAD,
    IMAGE_GET_STATUS,
    SORA_CREATE_VIDEO,
    SORA_DELETE,
    SORA_DOWNLOAD,
    SORA_GET_STATUS,
    SORA_LIST,
    SORA_LIST_REFERENCES,
    SORA_PREPARE_REFERENCE,
    SORA_REMIX,
)
from .tools import image, reference, video


# Initialize FastMCP server
mcp = FastMCP("sora-mcp-server")  # Consistent naming with repo


# ==================== VIDEO TOOLS ====================
@mcp.tool(description=SORA_CREATE_VIDEO)
async def sora_create_video(
    prompt: str,
    model: VideoModel = "sora-2",
    seconds: VideoSeconds | None = None,
    size: VideoSize | None = None,
    input_reference_filename: str | None = None,
):
    return await video.sora_create_video(prompt, model, seconds, size, input_reference_filename)


@mcp.tool(description=SORA_GET_STATUS)
async def sora_get_status(video_id: str):
    return await video.sora_get_status(video_id)


@mcp.tool(description=SORA_DOWNLOAD)
async def sora_download(
    video_id: str,
    filename: str | None = None,
    variant: Literal["video", "thumbnail", "spritesheet"] = "video",
):
    return await video.sora_download(video_id, filename, variant)


@mcp.tool(description=SORA_LIST)
async def sora_list(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc"):
    return await video.sora_list(limit, after, order)


@mcp.tool(description=SORA_DELETE)
async def sora_delete(video_id: str):
    return await video.sora_delete(video_id)


@mcp.tool(description=SORA_REMIX)
async def sora_remix(previous_video_id: str, prompt: str):
    return await video.sora_remix(previous_video_id, prompt)


# ==================== REFERENCE IMAGE TOOLS ====================
@mcp.tool(description=SORA_LIST_REFERENCES)
async def sora_list_references(
    pattern: str | None = None,
    file_type: Literal["jpeg", "png", "webp", "all"] = "all",
    sort_by: Literal["name", "size", "modified"] = "modified",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
):
    return await reference.sora_list_references(pattern, file_type, sort_by, order, limit)


@mcp.tool(description=SORA_PREPARE_REFERENCE)
async def sora_prepare_reference(
    input_filename: str,
    target_size: VideoSize,
    output_filename: str | None = None,
    resize_mode: Literal["crop", "pad", "rescale"] = "crop",
):
    return await reference.sora_prepare_reference(input_filename, target_size, output_filename, resize_mode)


# ==================== IMAGE GENERATION TOOLS ====================
@mcp.tool(description=IMAGE_CREATE)
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


@mcp.tool(description=IMAGE_GET_STATUS)
async def image_get_status(response_id: str):
    return await image.image_get_status(response_id)


@mcp.tool(description=IMAGE_DOWNLOAD)
async def image_download(response_id: str, filename: str | None = None):
    return await image.image_download(response_id, filename)


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
