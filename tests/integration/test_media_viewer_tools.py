# SPDX-License-Identifier: MIT
"""Integration tests for media viewer tools with mocked storage."""

import base64

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.media_viewer import get_media_data, view_media


@pytest.mark.integration
async def test_view_media_then_get_data_roundtrip(mocker, tmp_video_path):
    """Full roundtrip: view_media returns metadata, then get_media_data returns the file."""
    content = b"fake video content for roundtrip test"
    test_file = tmp_video_path / "roundtrip.mp4"
    test_file.write_bytes(content)

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

    # Step 1: Get metadata
    metadata = await view_media("video", "roundtrip.mp4")
    assert metadata["filename"] == "roundtrip.mp4"
    assert metadata["size_bytes"] == len(content)

    # Step 2: Fetch data using metadata
    data_result = await get_media_data(
        metadata["media_type"],
        metadata["filename"],
        offset=0,
        chunk_size=metadata["size_bytes"],
    )

    assert data_result["is_last"] is True
    assert base64.b64decode(data_result["data"]) == content


@pytest.mark.integration
async def test_chunked_download_assembles_correctly(mocker, tmp_audio_path):
    """Simulate multi-chunk download like the MCP App would."""
    content = b"A" * 500 + b"B" * 500 + b"C" * 24
    test_file = tmp_audio_path / "long.mp3"
    test_file.write_bytes(content)

    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

    # Simulate the client-side chunked loop
    assembled = bytearray()
    offset = 0
    chunk_size = 512

    while True:
        result = await get_media_data("audio", "long.mp3", offset=offset, chunk_size=chunk_size)
        chunk_bytes = base64.b64decode(result["data"])
        assembled.extend(chunk_bytes)
        offset += result["chunk_size"]
        if result["is_last"]:
            break

    assert bytes(assembled) == content


@pytest.mark.integration
async def test_image_media_viewer(mocker, tmp_reference_path):
    """view_media + get_media_data work for image files."""
    content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    test_file = tmp_reference_path / "test.png"
    test_file.write_bytes(content)

    storage = LocalStorageBackend(path_overrides={"reference": tmp_reference_path})
    mocker.patch("sanzaru.tools.media_viewer.get_storage", return_value=storage)

    metadata = await view_media("image", "test.png")
    assert metadata["mime_type"] == "image/png"

    data = await get_media_data("image", "test.png")
    assert data["is_last"] is True
    assert base64.b64decode(data["data"]) == content


@pytest.mark.integration
async def test_serve_media_route_content_type(mocker, tmp_video_path):
    """Test the custom HTTP route returns correct content-type headers."""
    from starlette.testclient import TestClient

    content = b"fake mp4 data"
    test_file = tmp_video_path / "test.mp4"
    test_file.write_bytes(content)

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.server.get_storage", return_value=storage)

    # Import the mcp server and build a Starlette test client
    from sanzaru.server import mcp

    app = mcp.streamable_http_app()
    client = TestClient(app)

    response = client.get("/media/video/test.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == content


@pytest.mark.integration
async def test_serve_media_route_not_found(mocker, tmp_video_path):
    """Test the custom HTTP route returns 404 for missing files."""
    from starlette.testclient import TestClient

    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.server.get_storage", return_value=storage)

    from sanzaru.server import mcp

    app = mcp.streamable_http_app()
    client = TestClient(app)

    response = client.get("/media/video/nonexistent.mp4")
    assert response.status_code == 404


@pytest.mark.integration
async def test_serve_media_route_invalid_type(mocker):
    """Test the custom HTTP route returns 400 for invalid media type."""
    from starlette.testclient import TestClient

    from sanzaru.server import mcp

    app = mcp.streamable_http_app()
    client = TestClient(app)

    response = client.get("/media/invalid/file.txt")
    assert response.status_code == 400
