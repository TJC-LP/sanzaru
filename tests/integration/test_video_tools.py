# SPDX-License-Identifier: MIT
"""Integration tests for video tools with mocked OpenAI client."""

import pytest

from sora_mcp_server.tools.video import (
    create_video,
    delete_video,
    download_video,
    get_video_status,
    list_videos,
    remix_video,
)


@pytest.mark.integration
async def test_sora_create_video_without_reference(mocker, mock_video_queued):
    """Test video creation calls OpenAI API correctly."""
    mock_get_client = mocker.patch("sora_mcp_server.tools.video.get_client")
    mock_get_client.return_value.videos.create = mocker.AsyncMock(return_value=mock_video_queued)

    result = await create_video(prompt="test video", model="sora-2", seconds="8", size="1280x720")

    assert result.id == "vid_queued"
    assert result.status == "queued"
    assert result.progress == 0
    mock_get_client.return_value.videos.create.assert_called_once()


@pytest.mark.integration
async def test_sora_get_status(mocker, mock_video_response):
    """Test status retrieval from OpenAI API."""
    mock_get_client = mocker.patch("sora_mcp_server.tools.video.get_client")
    mock_get_client.return_value.videos.retrieve = mocker.AsyncMock(return_value=mock_video_response)

    result = await get_video_status("vid_test123")

    assert result.id == "vid_test123"
    assert result.status == "completed"
    assert result.progress == 100
    mock_get_client.return_value.videos.retrieve.assert_called_once_with("vid_test123")


@pytest.mark.integration
async def test_sora_download(mocker, tmp_video_path):
    """Test video download writes file correctly."""
    mocker.patch("sora_mcp_server.tools.video.get_path", return_value=tmp_video_path)

    # Mock streaming response with async iteration
    mock_response = mocker.MagicMock()

    # Mock iter_bytes to return async iterator of chunks
    async def mock_iter():
        yield b"chunk1"
        yield b"chunk2"

    mock_response.iter_bytes.return_value = mock_iter()

    # Mock the streaming context manager
    mock_stream_ctx = mocker.MagicMock()
    mock_stream_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = mocker.AsyncMock(return_value=None)

    mock_get_client = mocker.patch("sora_mcp_server.tools.video.get_client")
    mock_get_client.return_value.with_streaming_response.videos.download_content.return_value = mock_stream_ctx

    result = await download_video("vid_test123", filename="test.mp4", variant="video")

    assert result["filename"] == "test.mp4"
    assert result["variant"] == "video"
    assert "test.mp4" in result["path"]
    mock_get_client.return_value.with_streaming_response.videos.download_content.assert_called_once_with(
        "vid_test123", variant="video"
    )


@pytest.mark.integration
async def test_sora_list(mocker):
    """Test listing videos with pagination."""
    from openai.types import Video

    mock_page = mocker.MagicMock()
    mock_page.data = [
        Video(
            id="vid1",
            object="video",
            status="completed",
            progress=100,
            model="sora-2",
            seconds="8",
            size="1280x720",
            created_at=1000,
        ),
        Video(
            id="vid2",
            object="video",
            status="in_progress",
            progress=50,
            model="sora-2",
            seconds="8",
            size="720x1280",
            created_at=2000,
        ),
    ]
    mock_page.has_more = True

    mock_get_client = mocker.patch("sora_mcp_server.tools.video.get_client")
    mock_get_client.return_value.videos.list = mocker.AsyncMock(return_value=mock_page)

    result = await list_videos(limit=20, order="desc")

    assert len(result["data"]) == 2
    assert result["data"][0]["id"] == "vid1"
    assert result["data"][1]["status"] == "in_progress"
    assert result["has_more"] is True
    assert result["last"] == "vid2"


@pytest.mark.integration
async def test_sora_delete(mocker):
    """Test video deletion calls API correctly."""
    from openai.types import VideoDeleteResponse

    mock_response = VideoDeleteResponse(id="vid_test123", object="video.deleted", deleted=True)

    mock_get_client = mocker.patch("sora_mcp_server.tools.video.get_client")
    mock_get_client.return_value.videos.delete = mocker.AsyncMock(return_value=mock_response)

    result = await delete_video("vid_test123")

    assert result.id == "vid_test123"
    assert result.deleted is True
    mock_get_client.return_value.videos.delete.assert_called_once_with("vid_test123")


@pytest.mark.integration
async def test_sora_remix(mocker, mock_video_queued):
    """Test video remix creates new job."""
    mock_get_client = mocker.patch("sora_mcp_server.tools.video.get_client")
    mock_get_client.return_value.videos.remix = mocker.AsyncMock(return_value=mock_video_queued)

    result = await remix_video("vid_original", "new prompt")

    assert result.id == "vid_queued"
    assert result.status == "queued"
    mock_get_client.return_value.videos.remix.assert_called_once_with("vid_original", prompt="new prompt")
