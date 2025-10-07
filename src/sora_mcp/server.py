# SPDX-License-Identifier: MIT
import asyncio
import logging
import os
import pathlib
import time
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


# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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


def _suffix_for_variant(variant: str) -> str:
    return {"video": "mp4", "thumbnail": "webp", "spritesheet": "jpg"}[variant]


@mcp.tool()
async def sora_create_video(
    prompt: str,
    model: VideoModel = "sora-2",
    seconds: VideoSeconds | None = None,
    size: VideoSize | None = None,
    input_reference_path: str | None = None,
) -> Video:
    """
    Start a Sora render job. Returns Video object with id/status/progress.
    Note: seconds must be a string ("4", "8", or "12"), not an integer.
    """
    client = get_client()

    # Convert None to omit for OpenAI SDK
    seconds_param = omit if seconds is None else seconds
    size_param = omit if size is None else size

    if input_reference_path:
        # Sora expects the input reference to match the target video size.
        with open(input_reference_path, "rb") as f:
            video = await client.videos.create(
                model=model,
                prompt=prompt,
                seconds=seconds_param,
                size=size_param,
                input_reference=f,
            )
    else:
        video = await client.videos.create(
            model=model,
            prompt=prompt,
            seconds=seconds_param,
            size=size_param,
        )

    logger.info("Started job %s (%s)", video.id, video.status)
    return video


@mcp.tool()
async def sora_get_status(video_id: str) -> Video:
    """Retrieve current job status for a given video_id."""
    client = get_client()
    video = await client.videos.retrieve(video_id)
    return video


@mcp.tool()
async def sora_wait(
    video_id: str,
    poll_every: int = 5,
    timeout: int = 600,
) -> Video:
    """
    Poll until job completes/fails or timeout.
    Returns the Video object. Check video.status to see if it completed or failed.
    """
    client = get_client()
    start = time.monotonic()
    last_progress = -1
    while True:
        video = await client.videos.retrieve(video_id)
        progress = video.progress
        if progress != last_progress:
            logger.info("Job %s progress: %s%% (status=%s)", video_id, progress, video.status)
            last_progress = progress

        if video.status in ("completed", "failed"):
            logger.info("Job %s finished: %s", video_id, video.status)
            return video

        if time.monotonic() - start > timeout:
            logger.warning("Job %s timed out after %ss", video_id, timeout)
            return video

        await asyncio.sleep(poll_every)


@mcp.tool()
async def sora_download(
    video_id: str,
    variant: Literal["video", "thumbnail", "spritesheet"] = "video",
    path: str | None = None,
) -> DownloadResult:
    """
    Download the asset to disk and return its absolute path.
    """
    client = get_client()
    content = await client.videos.download_content(video_id, variant=variant)
    suffix = _suffix_for_variant(variant)
    out_path = pathlib.Path(path or f"{video_id}.{suffix}").resolve()
    content.write_to_file(str(out_path))
    logger.info("Wrote %s (%s)", out_path, variant)
    return {"path": str(out_path), "variant": variant}


@mcp.tool()
async def sora_list(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc") -> ListResult:
    """List videos with pagination hints. Returns dict with 'data' (list of Videos), 'has_more', and 'last' (id)."""
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


@mcp.tool()
async def sora_delete(video_id: str) -> VideoDeleteResponse:
    """Delete a video from OpenAI storage."""
    client = get_client()
    resp = await client.videos.delete(video_id)
    logger.info("Deleted %s", video_id)
    return resp


@mcp.tool()
async def sora_remix(previous_video_id: str, prompt: str) -> Video:
    """
    Remix an existing video with a new prompt.
    """
    client = get_client()
    video = await client.videos.remix(previous_video_id, prompt=prompt)
    logger.info("Started remix %s (from %s)", video.id, previous_video_id)
    return video


# -------- Entrypoint --------
def main():
    logger.info("Starting MCP server over stdio")
    mcp.run()
