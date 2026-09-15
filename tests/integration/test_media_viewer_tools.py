# SPDX-License-Identifier: MIT
"""Integration tests for media viewer tools with mocked storage."""

import base64

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.media_viewer import get_media_data, view_media


@pytest.fixture(autouse=True)
def _isolated_http_env(monkeypatch):
    """The /media route reads its policy from the environment at request time.

    A developer who followed the README and exported SANZARU_HTTP_TOKEN got four
    unexplained 401s out of this module; the identity variables would skew the
    Databricks-shaped tests the same way. Nothing here may depend on the shell.
    """
    for name in (
        "SANZARU_HTTP_TOKEN",
        "SANZARU_ALLOW_UNAUTHENTICATED_HTTP",
        "SANZARU_IDENTITY_HEADER",
        "SANZARU_ALLOWED_HOSTS",
        "SANZARU_ALLOWED_ORIGINS",
        "SANZARU_REQUIRE_USER_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)


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


# The /media route now enforces the same Host allowlist as /mcp, and
# TestClient's default Host is "testserver" — which is exactly the header a
# DNS-rebound browser page would send. Legitimate requests must name a
# loopback host.
_LOCAL = {"host": "127.0.0.1:8000"}


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

    response = client.get("/media/video/test.mp4", headers=_LOCAL)
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

    response = client.get("/media/video/nonexistent.mp4", headers=_LOCAL)
    assert response.status_code == 404


@pytest.mark.integration
async def test_serve_media_route_invalid_type(mocker):
    """Test the custom HTTP route returns 400 for invalid media type."""
    from starlette.testclient import TestClient

    from sanzaru.server import mcp

    app = mcp.streamable_http_app()
    client = TestClient(app)

    response = client.get("/media/invalid/file.txt", headers=_LOCAL)
    assert response.status_code == 400


@pytest.mark.integration
async def test_serve_media_rejects_foreign_host_header(mocker, tmp_video_path):
    """A DNS-rebound page cannot read the media library (CWE-346).

    FastMCP appends custom routes outside the middleware that guards /mcp, so
    this route answered any Host at all while /mcp rejected it.
    """
    from starlette.testclient import TestClient

    (tmp_video_path / "secret.mp4").write_bytes(b"private")
    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.server.get_storage", return_value=storage)

    from sanzaru.server import mcp

    client = TestClient(mcp.streamable_http_app())

    response = client.get("/media/video/secret.mp4", headers={"host": "evil.example:8000"})

    assert response.status_code == 421
    assert b"private" not in response.content


@pytest.mark.integration
async def test_serve_media_never_returns_an_executable_content_type(mocker, tmp_audio_path):
    """A stored .html file is served inert, not as a document (CWE-79).

    The response type used to be `mimetypes.guess_type()` of a caller-chosen
    name, so text persisted under an .html name executed in the server's own
    origin — the origin the SDK's rebinding allowlist trusts for /mcp.
    """
    from starlette.testclient import TestClient

    (tmp_audio_path / "x.html").write_bytes(b"<script>alert(1)</script>")
    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.server.get_storage", return_value=storage)

    from sanzaru.server import mcp

    client = TestClient(mcp.streamable_http_app())

    response = client.get("/media/audio/x.html", headers=_LOCAL)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"] == "attachment"


@pytest.mark.integration
async def test_serve_media_requires_the_bearer_token_when_configured(mocker, tmp_video_path, monkeypatch):
    """With SANZARU_HTTP_TOKEN set, /media is closed to unauthenticated callers."""
    from starlette.testclient import TestClient

    (tmp_video_path / "clip.mp4").write_bytes(b"data")
    storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
    mocker.patch("sanzaru.server.get_storage", return_value=storage)
    monkeypatch.setenv("SANZARU_HTTP_TOKEN", "s3cret")

    from sanzaru.server import mcp

    client = TestClient(mcp.streamable_http_app())

    assert client.get("/media/video/clip.mp4", headers=_LOCAL).status_code == 401
    assert client.get("/media/video/clip.mp4", headers=_LOCAL | {"authorization": "Bearer wrong"}).status_code == 401

    allowed = client.get("/media/video/clip.mp4", headers=_LOCAL | {"authorization": "Bearer s3cret"})
    assert allowed.status_code == 200
    assert allowed.content == b"data"


@pytest.mark.integration
async def test_serve_media_answers_403_when_the_backend_requires_an_identity(mocker):
    """SANZARU_REQUIRE_USER_CONTEXT with no identity on the request is a refusal, not a crash.

    The Databricks backend raises UserContextRequiredError rather than fall
    back to the shared volume root; uncaught, that reached the error middleware
    as a 500 with a traceback. The storage side is pinned in its own tests —
    this pins only what the route does with the exception.
    """
    from starlette.testclient import TestClient

    from sanzaru.user_context import UserContextRequiredError

    storage = mocker.AsyncMock()
    storage.read.side_effect = UserContextRequiredError(
        "SANZARU_REQUIRE_USER_CONTEXT is set but this request carries no user identity"
    )
    mocker.patch("sanzaru.server.get_storage", return_value=storage)

    from sanzaru.server import mcp

    client = TestClient(mcp.streamable_http_app())

    response = client.get("/media/video/clip.mp4", headers=_LOCAL)

    assert response.status_code == 403
    assert response.text == "Forbidden"
