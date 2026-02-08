# SPDX-License-Identifier: MIT
"""Integration tests for video tools with mocked OpenAI client."""

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.video import (
    create_video,
    delete_video,
    download_video,
    get_video_status,
    list_local_videos,
    list_videos,
    remix_video,
)


@pytest.mark.integration
async def test_sora_create_video_without_reference(mocker, mock_video_queued):
    """Test video creation calls OpenAI API correctly."""
    mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
    mock_get_client.return_value.videos.create = mocker.AsyncMock(return_value=mock_video_queued)

    result = await create_video(prompt="test video", model="sora-2", seconds="8", size="1280x720")

    assert result.id == "vid_queued"
    assert result.status == "queued"
    assert result.progress == 0
    mock_get_client.return_value.videos.create.assert_called_once()


@pytest.mark.integration
async def test_sora_get_status(mocker, mock_video_response):
    """Test status retrieval from OpenAI API."""
    mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
    mock_get_client.return_value.videos.retrieve = mocker.AsyncMock(return_value=mock_video_response)

    result = await get_video_status("vid_test123")

    assert result.id == "vid_test123"
    assert result.status == "completed"
    assert result.progress == 100
    mock_get_client.return_value.videos.retrieve.assert_called_once_with("vid_test123")


@pytest.mark.integration
async def test_sora_download(mocker, tmp_video_path):
    """Test video download writes file correctly."""
    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

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

    mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
    mock_get_client.return_value.with_streaming_response.videos.download_content.return_value = mock_stream_ctx

    result = await download_video("vid_test123", filename="test.mp4", variant="video")

    assert result["filename"] == "test.mp4"
    assert result["variant"] == "video"
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

    mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
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

    mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
    mock_get_client.return_value.videos.delete = mocker.AsyncMock(return_value=mock_response)

    result = await delete_video("vid_test123")

    assert result.id == "vid_test123"
    assert result.deleted is True
    mock_get_client.return_value.videos.delete.assert_called_once_with("vid_test123")


@pytest.mark.integration
async def test_sora_remix(mocker, mock_video_queued):
    """Test video remix creates new job."""
    mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
    mock_get_client.return_value.videos.remix = mocker.AsyncMock(return_value=mock_video_queued)

    result = await remix_video("vid_original", "new prompt")

    assert result.id == "vid_queued"
    assert result.status == "queued"
    mock_get_client.return_value.videos.remix.assert_called_once_with("vid_original", prompt="new prompt")


@pytest.mark.integration
async def test_create_video_with_reference_image_mime_types(mocker, mock_video_queued, tmp_path):
    """Verify MIME types are correctly passed for different image formats."""
    # Test cases: (extension, expected_mime_type)
    test_cases = [
        ("test.jpg", "image/jpeg"),
        ("test.jpeg", "image/jpeg"),
        ("test.png", "image/png"),
        ("test.webp", "image/webp"),
    ]

    for filename, expected_mime in test_cases:
        # Create a temporary reference image directory
        reference_path = tmp_path / "references"
        reference_path.mkdir(exist_ok=True)
        image_file = reference_path / filename
        image_file.write_bytes(b"fake image data")

        # Mock storage backend with temp directory
        storage = LocalStorageBackend(path_overrides={"reference": reference_path})
        mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

        # Mock the OpenAI client
        mock_get_client = mocker.patch("sanzaru.tools.video.get_client")
        mock_get_client.return_value.videos.create = mocker.AsyncMock(return_value=mock_video_queued)

        # Call create_video with reference image
        result = await create_video(
            prompt="test video",
            model="sora-2",
            seconds="8",
            size="1280x720",
            input_reference_filename=filename,
        )

        # Verify the video was created
        assert result.id == "vid_queued"
        assert result.status == "queued"

        # Verify the SDK was called with correct MIME type tuple
        call_args = mock_get_client.return_value.videos.create.call_args
        input_reference_arg = call_args.kwargs["input_reference"]

        # Should be a tuple: (filename, bytes, mime_type)
        assert isinstance(input_reference_arg, tuple), f"Expected tuple for {filename}"
        assert len(input_reference_arg) == 3, f"Expected 3-element tuple for {filename}"
        assert input_reference_arg[0] == filename, f"Filename mismatch for {filename}"
        assert isinstance(input_reference_arg[1], bytes), f"Expected bytes for {filename}"
        assert input_reference_arg[2] == expected_mime, f"MIME type mismatch for {filename}: {input_reference_arg[2]}"


# ==================== list_local_videos Tests ====================


@pytest.mark.integration
async def test_list_local_videos_basic(mocker, tmp_video_path):
    """Test listing local video files."""
    # Create test video files
    (tmp_video_path / "video1.mp4").write_bytes(b"x" * 100)
    (tmp_video_path / "video2.webm").write_bytes(b"y" * 200)
    (tmp_video_path / "video3.mov").write_bytes(b"z" * 300)

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

    result = await list_local_videos()

    assert len(result["data"]) == 3
    filenames = {v["filename"] for v in result["data"]}
    assert filenames == {"video1.mp4", "video2.webm", "video3.mov"}
    # Verify no path fields leaked
    for item in result["data"]:
        assert "path" not in item


@pytest.mark.integration
async def test_list_local_videos_filter_by_type(mocker, tmp_video_path):
    """Test filtering local videos by file type."""
    (tmp_video_path / "video1.mp4").write_bytes(b"x" * 100)
    (tmp_video_path / "video2.webm").write_bytes(b"y" * 200)
    (tmp_video_path / "video3.mp4").write_bytes(b"z" * 300)

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

    result = await list_local_videos(file_type="mp4")

    assert len(result["data"]) == 2
    for item in result["data"]:
        assert item["file_type"] == "mp4"


@pytest.mark.integration
async def test_list_local_videos_sort_by_size(mocker, tmp_video_path):
    """Test sorting local videos by size."""
    (tmp_video_path / "small.mp4").write_bytes(b"x" * 100)
    (tmp_video_path / "large.mp4").write_bytes(b"y" * 1000)
    (tmp_video_path / "medium.mp4").write_bytes(b"z" * 500)

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

    result = await list_local_videos(sort_by="size", order="asc")

    sizes = [item["size_bytes"] for item in result["data"]]
    assert sizes == sorted(sizes)


@pytest.mark.integration
async def test_list_local_videos_limit(mocker, tmp_video_path):
    """Test limiting the number of results."""
    for i in range(10):
        (tmp_video_path / f"video{i}.mp4").write_bytes(b"x" * (i + 1) * 100)

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

    result = await list_local_videos(limit=3)

    assert len(result["data"]) == 3


@pytest.mark.integration
async def test_list_local_videos_empty(mocker, tmp_video_path):
    """Test listing when no video files exist."""
    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=storage)

    result = await list_local_videos()

    assert result["data"] == []
