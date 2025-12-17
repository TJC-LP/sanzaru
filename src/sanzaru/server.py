# SPDX-License-Identifier: MIT
"""sanzaru MCP Server - Unified MCP server for OpenAI multimodal APIs.

This module initializes the FastMCP server and conditionally registers tools
based on installed optional dependencies (video, audio, image).

Business logic is organized into submodules under tools/.
"""

import argparse
from typing import Literal

from mcp.server.fastmcp import FastMCP
from openai.types import VideoModel, VideoSeconds, VideoSize
from openai.types.responses.tool_param import ImageGeneration

from .config import logger
from .features import check_audio_available, check_image_available, check_video_available

# Optional dotenv support for local development
try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# Initialize FastMCP server (stateless configuration set at runtime)
mcp = FastMCP("sanzaru")


# ==================== VIDEO TOOLS (CONDITIONAL) ====================
if check_video_available():
    from .descriptions import (
        CREATE_VIDEO,
        DELETE_VIDEO,
        DOWNLOAD_VIDEO,
        GET_VIDEO_STATUS,
        LIST_VIDEOS,
        REMIX_VIDEO,
    )
    from .tools import video

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

    logger.info("Video tools registered (6 tools)")


# ==================== IMAGE TOOLS (CONDITIONAL) ====================
if check_image_available():
    from openai.types import ImageModel

    from .descriptions import (
        CREATE_IMAGE,
        DOWNLOAD_IMAGE,
        EDIT_IMAGE,
        GENERATE_IMAGE,
        GET_IMAGE_STATUS,
        LIST_REFERENCE_IMAGES,
        PREPARE_REFERENCE_IMAGE,
    )
    from .tools import image, images_api, reference

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

    @mcp.tool(description=CREATE_IMAGE)
    async def create_image(
        prompt: str,
        model: str = "gpt-5.2",
        tool_config: ImageGeneration | None = None,
        previous_response_id: str | None = None,
        input_images: list[str] | None = None,
        mask_filename: str | None = None,
    ):
        return await image.create_image(prompt, model, tool_config, previous_response_id, input_images, mask_filename)

    @mcp.tool(description=GET_IMAGE_STATUS)
    async def get_image_status(response_id: str):
        return await image.get_image_status(response_id)

    @mcp.tool(description=DOWNLOAD_IMAGE)
    async def download_image(response_id: str, filename: str | None = None):
        return await image.download_image(response_id, filename)

    # Images API tools (direct API, supports gpt-image-1.5)
    @mcp.tool(description=GENERATE_IMAGE)
    async def generate_image(
        prompt: str,
        model: ImageModel = "gpt-image-1.5",
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536"] = "auto",
        quality: Literal["auto", "low", "medium", "high"] = "auto",
        background: Literal["auto", "transparent", "opaque"] = "auto",
        output_format: Literal["png", "jpeg", "webp"] = "png",
        moderation: Literal["auto", "low"] = "auto",
        filename: str | None = None,
    ):
        return await images_api.generate_image(
            prompt, model, size, quality, background, output_format, moderation, filename
        )

    @mcp.tool(description=EDIT_IMAGE)
    async def edit_image(
        prompt: str,
        input_images: list[str],
        model: ImageModel = "gpt-image-1.5",
        mask_filename: str | None = None,
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536"] = "auto",
        quality: Literal["auto", "low", "medium", "high"] = "auto",
        background: Literal["auto", "transparent", "opaque"] = "auto",
        output_format: Literal["png", "jpeg", "webp"] = "png",
        input_fidelity: Literal["high", "low"] | None = None,
        filename: str | None = None,
    ):
        return await images_api.edit_image(
            prompt,
            input_images,
            model,
            mask_filename,
            size,
            quality,
            background,
            output_format,
            input_fidelity,
            filename,
        )

    logger.info("Image tools registered (7 tools)")


# ==================== AUDIO TOOLS (CONDITIONAL) ====================
if check_audio_available():
    from openai.types import AudioModel, AudioResponseFormat
    from openai.types.audio.speech_model import SpeechModel

    from .audio.constants import AudioChatModel, EnhancementType, SortBy, TTSVoice
    from .descriptions import (
        CHAT_WITH_AUDIO,
        COMPRESS_AUDIO,
        CONVERT_AUDIO,
        CREATE_AUDIO,
        GET_LATEST_AUDIO,
        LIST_AUDIO_FILES,
        TRANSCRIBE_AUDIO,
        TRANSCRIBE_WITH_ENHANCEMENT,
    )
    from .tools import audio

    @mcp.tool(description=LIST_AUDIO_FILES)
    async def list_audio_files(
        pattern: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        min_duration_seconds: float | None = None,
        max_duration_seconds: float | None = None,
        min_modified_time: float | None = None,
        max_modified_time: float | None = None,
        format: str | None = None,
        sort_by: SortBy = SortBy.NAME,
        reverse: bool = False,
    ):
        return await audio.list_audio_files(
            pattern=pattern,
            min_size_bytes=min_size_bytes,
            max_size_bytes=max_size_bytes,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            min_modified_time=min_modified_time,
            max_modified_time=max_modified_time,
            format=format,
            sort_by=sort_by,
            reverse=reverse,
        )

    @mcp.tool(description=GET_LATEST_AUDIO)
    async def get_latest_audio():
        return await audio.get_latest_audio()

    @mcp.tool(description=CONVERT_AUDIO)
    async def convert_audio(input_path: str, output_format: Literal["mp3", "wav"]):
        return await audio.convert_audio(input_path, output_format)

    @mcp.tool(description=COMPRESS_AUDIO)
    async def compress_audio(input_path: str, max_mb: int = 25):
        return await audio.compress_audio(input_path, max_mb)

    @mcp.tool(description=TRANSCRIBE_AUDIO)
    async def transcribe_audio(
        file_path: str,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ):
        return await audio.transcribe_audio(file_path, model, response_format, prompt, timestamp_granularities)

    @mcp.tool(description=CHAT_WITH_AUDIO)
    async def chat_with_audio(
        file_path: str,
        model: AudioChatModel = "gpt-4o-audio-preview",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        return await audio.chat_with_audio(file_path, model, system_prompt, user_prompt)

    @mcp.tool(description=TRANSCRIBE_WITH_ENHANCEMENT)
    async def transcribe_with_enhancement(
        file_path: str,
        enhancement_type: EnhancementType = "detailed",
        model: AudioModel = "gpt-4o-mini-transcribe",
    ):
        return await audio.transcribe_with_enhancement(file_path, enhancement_type, model)

    @mcp.tool(description=CREATE_AUDIO)
    async def create_audio(
        text_prompt: str,
        model: SpeechModel = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: str | None = None,
        speed: float = 1.0,
        output_filename: str | None = None,
    ):
        return await audio.create_audio(text_prompt, model, voice, instructions, speed, output_filename)

    logger.info("Audio tools registered (8 tools)")


# ==================== SERVER ENTRYPOINT ====================
def main():
    """Run the MCP server.

    Tools are registered conditionally based on installed optional dependencies:
    - video: Sora video generation (no extra deps, always available)
    - audio: Whisper transcription, GPT-4o audio, TTS (requires pydub, ffmpeg-python)
    - image: GPT Vision image generation and reference management (requires pillow)

    Install with: uv add "sanzaru[video,audio,image]" or any combination.

    Paths are validated lazily at runtime when tools are called.

    Environment variables should be set explicitly in .mcp.json or passed via the calling environment.
    For local development with .env files, install python-dotenv: uv add --dev python-dotenv

    Transport options:
    - stdio (default): Standard I/O for Claude Desktop and MCP clients
    - http: Stateless HTTP streaming for web clients and remote access
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Sanzaru MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transport (default: 8000)",
    )
    args = parser.parse_args()

    # Optional: Load .env file if dotenv is installed (local development only)
    if _DOTENV_AVAILABLE:
        load_dotenv()
        logger.debug("Loaded environment variables from .env file")

    # Log enabled features
    enabled = []
    if check_video_available():
        enabled.append("video")
    if check_audio_available():
        enabled.append("audio")
    if check_image_available():
        enabled.append("image")

    if enabled:
        logger.info(f"Enabled features: {', '.join(enabled)}")
    else:
        logger.warning("No features enabled - install optional dependencies with: uv add 'sanzaru[all]'")

    # Run server with selected transport
    if args.transport == "http":
        logger.info(f"Starting sanzaru MCP server over HTTP at http://{args.host}:{args.port}/mcp")
        # Configure for stateless HTTP (no session IDs needed - all state in OpenAI cloud)
        mcp.settings.stateless_http = True
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        logger.info("Starting sanzaru MCP server over stdio")
        mcp.run()


if __name__ == "__main__":
    main()
