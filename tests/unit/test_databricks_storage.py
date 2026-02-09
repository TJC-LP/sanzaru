# SPDX-License-Identifier: MIT
"""Unit tests for DatabricksVolumesBackend with mocked httpx."""

import pathlib
import time

import httpx
import pytest

from sanzaru.storage.databricks import DatabricksVolumesBackend
from sanzaru.storage.protocol import FileInfo, StorageBackend
from sanzaru.user_context import UserContext, reset_user_context, set_user_context

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _databricks_env(monkeypatch):
    """Set required env vars for every test."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.net")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("DATABRICKS_VOLUME_PATH", "catalog/schema/vol")


@pytest.fixture
def backend():
    return DatabricksVolumesBackend()


def _resp(status_code: int, **kwargs) -> httpx.Response:
    """Create an httpx.Response with a dummy request (needed for raise_for_status)."""
    return httpx.Response(status_code, request=httpx.Request("GET", "https://test"), **kwargs)


@pytest.fixture
def mock_token_response():
    """Standard OAuth token response."""
    return _resp(200, json={"access_token": "tok_123", "expires_in": 3600})


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


@pytest.mark.unit
def test_databricks_is_storage_backend(backend):
    assert isinstance(backend, StorageBackend)


# ------------------------------------------------------------------
# Lifecycle / resource cleanup
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_aclose_closes_httpx_client(backend, mocker):
    mock_aclose = mocker.patch.object(backend._client, "aclose")
    await backend.aclose()
    mock_aclose.assert_awaited_once()


@pytest.mark.unit
async def test_context_manager_calls_aclose(backend, mocker):
    mock_aclose = mocker.patch.object(backend._client, "aclose")
    async with backend:
        pass
    mock_aclose.assert_awaited_once()


# ------------------------------------------------------------------
# Environment variable validation
# ------------------------------------------------------------------


@pytest.mark.unit
def test_missing_single_env_var_raises(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST")
    with pytest.raises(RuntimeError, match="DATABRICKS_HOST"):
        DatabricksVolumesBackend()


@pytest.mark.unit
def test_missing_multiple_env_vars_lists_all(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST")
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET")
    with pytest.raises(RuntimeError, match="DATABRICKS_HOST") as exc_info:
        DatabricksVolumesBackend()
    # Both missing vars are mentioned in one error
    assert "DATABRICKS_CLIENT_SECRET" in str(exc_info.value)


@pytest.mark.unit
def test_empty_env_var_raises(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "   ")
    with pytest.raises(RuntimeError, match="DATABRICKS_HOST"):
        DatabricksVolumesBackend()


@pytest.mark.unit
def test_host_without_scheme_gets_https(monkeypatch):
    """DATABRICKS_HOST without https:// gets it added automatically."""
    monkeypatch.setenv("DATABRICKS_HOST", "adb-123.azuredatabricks.net")

    backend = DatabricksVolumesBackend()

    assert backend._host == "https://adb-123.azuredatabricks.net"


@pytest.mark.unit
def test_host_with_scheme_preserved(monkeypatch):
    """DATABRICKS_HOST with https:// is preserved as-is."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-123.azuredatabricks.net")

    backend = DatabricksVolumesBackend()

    assert backend._host == "https://adb-123.azuredatabricks.net"


@pytest.mark.unit
def test_sanzaru_media_path_fallback(monkeypatch):
    """SANZARU_MEDIA_PATH is used when DATABRICKS_VOLUME_PATH is not set."""
    monkeypatch.delenv("DATABRICKS_VOLUME_PATH")
    monkeypatch.setenv("SANZARU_MEDIA_PATH", "catalog/schema/vol")

    backend = DatabricksVolumesBackend()

    assert backend._volume_path == "catalog/schema/vol"


@pytest.mark.unit
def test_databricks_volume_path_takes_precedence(monkeypatch):
    """DATABRICKS_VOLUME_PATH takes precedence over SANZARU_MEDIA_PATH."""
    monkeypatch.setenv("DATABRICKS_VOLUME_PATH", "catalog/schema/vol")
    monkeypatch.setenv("SANZARU_MEDIA_PATH", "other/path")

    backend = DatabricksVolumesBackend()

    assert backend._volume_path == "catalog/schema/vol"


@pytest.mark.unit
def test_missing_volume_path_and_media_path_raises(monkeypatch):
    """Error when neither DATABRICKS_VOLUME_PATH nor SANZARU_MEDIA_PATH is set."""
    monkeypatch.delenv("DATABRICKS_VOLUME_PATH")
    monkeypatch.delenv("SANZARU_MEDIA_PATH", raising=False)

    with pytest.raises(RuntimeError, match="DATABRICKS_VOLUME_PATH"):
        DatabricksVolumesBackend()


# ------------------------------------------------------------------
# OAuth
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_token_acquisition(backend, mocker, mock_token_response):
    mock_post = mocker.patch.object(backend._client, "post", return_value=mock_token_response)

    token = await backend._get_token()

    assert token == "tok_123"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/oidc/v1/token" in call_kwargs.args[0]
    assert call_kwargs.kwargs["data"]["grant_type"] == "client_credentials"


@pytest.mark.unit
async def test_token_caching(backend, mocker, mock_token_response):
    mock_post = mocker.patch.object(backend._client, "post", return_value=mock_token_response)

    await backend._get_token()
    await backend._get_token()

    assert mock_post.call_count == 1  # Only one POST — cached


@pytest.mark.unit
async def test_token_refresh_on_expiry(backend, mocker, mock_token_response):
    mock_post = mocker.patch.object(backend._client, "post", return_value=mock_token_response)

    await backend._get_token()
    # Simulate token expiry
    backend._token_expires_at = time.monotonic() - 1
    await backend._get_token()

    assert mock_post.call_count == 2  # Refreshed


# ------------------------------------------------------------------
# read
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_read_calls_get(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mock_get = mocker.patch.object(backend._client, "get", return_value=_resp(200, content=b"PNG_DATA"))

    data = await backend.read("reference", "hero.png")

    assert data == b"PNG_DATA"
    url = mock_get.call_args.args[0]
    assert "/api/2.0/fs/files/" in url
    assert "images" in url
    assert "hero.png" in url


@pytest.mark.unit
async def test_read_404_raises_file_not_found(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "get", return_value=_resp(404))

    with pytest.raises(FileNotFoundError, match="hero.png"):
        await backend.read("reference", "hero.png")


# ------------------------------------------------------------------
# read_range
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_read_range_sends_range_header(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mock_get = mocker.patch.object(backend._client, "get", return_value=_resp(206, content=b"CHUNK"))

    data = await backend.read_range("reference", "hero.png", offset=100, length=50)

    assert data == b"CHUNK"
    headers = mock_get.call_args.kwargs["headers"]
    assert headers["Range"] == "bytes=100-149"


@pytest.mark.unit
async def test_read_range_accepts_200(backend, mocker, mock_token_response):
    """Some servers return 200 instead of 206 for range requests."""
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "get", return_value=_resp(200, content=b"FULL"))

    data = await backend.read_range("reference", "hero.png", offset=0, length=100)
    assert data == b"FULL"


@pytest.mark.unit
async def test_read_range_404_raises(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "get", return_value=_resp(404))

    with pytest.raises(FileNotFoundError, match="hero.png"):
        await backend.read_range("reference", "hero.png", offset=0, length=100)


@pytest.mark.unit
async def test_read_range_negative_offset_raises(backend):
    with pytest.raises(ValueError, match="non-negative"):
        await backend.read_range("reference", "hero.png", offset=-1, length=100)


# ------------------------------------------------------------------
# write
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_write_calls_put(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mock_put = mocker.patch.object(backend._client, "put", return_value=_resp(200))

    display = await backend.write("video", "out.mp4", b"VIDEO_DATA")

    url = mock_put.call_args.args[0]
    assert "/api/2.0/fs/files/" in url
    assert "videos" in url
    assert "out.mp4" in url
    assert mock_put.call_args.kwargs["content"] == b"VIDEO_DATA"
    assert mock_put.call_args.kwargs["headers"]["Content-Type"] == "application/octet-stream"
    assert "/Volumes/" in display


@pytest.mark.unit
async def test_write_returns_display_path(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "put", return_value=_resp(200))

    display = await backend.write("reference", "result.png", b"DATA")

    assert display == "/Volumes/catalog/schema/vol/images/result.png"


@pytest.mark.unit
async def test_write_stream_accumulates_chunks(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mock_put = mocker.patch.object(backend._client, "put", return_value=_resp(200))

    async def chunks():
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    await backend.write_stream("video", "streamed.mp4", chunks())

    assert mock_put.call_args.kwargs["content"] == b"chunk1chunk2chunk3"


# ------------------------------------------------------------------
# list_files
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_list_files_parses_json(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    dir_response = _resp(
        200,
        json={
            "contents": [
                {"name": "a.png", "file_size": 100, "last_modified": 1000000, "is_directory": False},
                {"name": "b.jpg", "file_size": 200, "last_modified": 2000000, "is_directory": False},
                {"name": "subdir", "is_directory": True},
            ]
        },
    )
    mocker.patch.object(backend._client, "get", return_value=dir_response)

    files = await backend.list_files("reference")

    assert len(files) == 2
    assert all(isinstance(f, FileInfo) for f in files)
    names = {f.name for f in files}
    assert names == {"a.png", "b.jpg"}


@pytest.mark.unit
async def test_list_files_filters_extensions(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    dir_response = _resp(
        200,
        json={
            "contents": [
                {"name": "a.png", "file_size": 100, "last_modified": 1000, "is_directory": False},
                {"name": "b.jpg", "file_size": 200, "last_modified": 2000, "is_directory": False},
                {"name": "c.txt", "file_size": 50, "last_modified": 3000, "is_directory": False},
            ]
        },
    )
    mocker.patch.object(backend._client, "get", return_value=dir_response)

    files = await backend.list_files("reference", extensions={".png"})

    assert len(files) == 1
    assert files[0].name == "a.png"


@pytest.mark.unit
async def test_list_files_filters_pattern(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    dir_response = _resp(
        200,
        json={
            "contents": [
                {"name": "cat_01.png", "file_size": 100, "last_modified": 1000, "is_directory": False},
                {"name": "cat_02.png", "file_size": 200, "last_modified": 2000, "is_directory": False},
                {"name": "dog_01.png", "file_size": 300, "last_modified": 3000, "is_directory": False},
            ]
        },
    )
    mocker.patch.object(backend._client, "get", return_value=dir_response)

    files = await backend.list_files("reference", pattern="cat*")

    assert len(files) == 2
    names = {f.name for f in files}
    assert names == {"cat_01.png", "cat_02.png"}


# ------------------------------------------------------------------
# stat / exists
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_stat_uses_head(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "head", return_value=_resp(200, headers={"Content-Length": "12345"}))

    info = await backend.stat("reference", "img.png")

    assert info.name == "img.png"
    assert info.size_bytes == 12345


@pytest.mark.unit
async def test_stat_404_raises(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "head", return_value=_resp(404))

    with pytest.raises(FileNotFoundError):
        await backend.stat("reference", "nope.png")


@pytest.mark.unit
async def test_exists_true_on_200(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "head", return_value=_resp(200))

    assert await backend.exists("reference", "img.png") is True


@pytest.mark.unit
async def test_exists_false_on_404(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "head", return_value=_resp(404))

    assert await backend.exists("reference", "nope.png") is False


# ------------------------------------------------------------------
# Security
# ------------------------------------------------------------------


@pytest.mark.unit
def test_validate_filename_strips_dirs(backend):
    assert backend._validate_filename("subdir/file.png") == "file.png"


@pytest.mark.unit
def test_validate_filename_rejects_traversal(backend):
    with pytest.raises(ValueError, match="Path traversal"):
        backend._validate_filename("../../../etc/passwd")


@pytest.mark.unit
def test_validate_filename_rejects_empty(backend):
    with pytest.raises(ValueError, match="Invalid filename"):
        backend._validate_filename("")


@pytest.mark.unit
def test_validate_filename_rejects_dot(backend):
    with pytest.raises(ValueError, match="Invalid filename"):
        backend._validate_filename(".")


# ------------------------------------------------------------------
# local_path / local_tempfile
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_local_path_downloads_to_temp(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "get", return_value=_resp(200, content=b"IMAGE_DATA"))

    async with backend.local_path("reference", "hero.png") as p:
        assert isinstance(p, pathlib.Path)
        assert p.exists()
        assert p.read_bytes() == b"IMAGE_DATA"
        assert p.suffix == ".png"

    # Temp file cleaned up
    assert not p.exists()


@pytest.mark.unit
async def test_local_tempfile_uploads_on_exit(backend, mocker, mock_token_response):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mock_put = mocker.patch.object(backend._client, "put", return_value=_resp(200))

    async with backend.local_tempfile("reference", "output.png") as p:
        assert isinstance(p, pathlib.Path)
        p.write_bytes(b"GENERATED_IMAGE")

    # Verify upload happened with correct content
    assert mock_put.call_count == 1
    assert mock_put.call_args.kwargs["content"] == b"GENERATED_IMAGE"

    # Temp file cleaned up
    assert not p.exists()


# ------------------------------------------------------------------
# resolve_display_path
# ------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_display_path(backend):
    display = backend.resolve_display_path("video", "clip.mp4")
    assert display == "/Volumes/catalog/schema/vol/videos/clip.mp4"


@pytest.mark.unit
def test_resolve_display_path_reference(backend):
    display = backend.resolve_display_path("reference", "hero.png")
    assert display == "/Volumes/catalog/schema/vol/images/hero.png"


# ------------------------------------------------------------------
# Per-user path prefixing (multi-tenant)
# ------------------------------------------------------------------


@pytest.fixture
def user_ctx():
    """Set a user context for the duration of the test, then reset."""
    token = set_user_context(UserContext(email="rcaputo3@tjclp.com"))
    yield
    reset_user_context(token)


@pytest.mark.unit
def test_user_prefix_returns_empty_without_context(backend):
    assert backend._user_prefix() == ""


@pytest.mark.unit
def test_user_prefix_returns_slug_with_context(backend, user_ctx):
    assert backend._user_prefix() == "rcaputo3"


@pytest.mark.unit
def test_file_url_includes_user_prefix(backend, user_ctx):
    url = backend._file_url("video", "clip.mp4")
    assert "/catalog/schema/vol/rcaputo3/videos/clip.mp4" in url


@pytest.mark.unit
def test_file_url_no_user_prefix_without_context(backend):
    url = backend._file_url("video", "clip.mp4")
    assert "/catalog/schema/vol/videos/clip.mp4" in url
    assert "/rcaputo3/" not in url


@pytest.mark.unit
def test_dir_url_includes_user_prefix(backend, user_ctx):
    url = backend._dir_url("reference")
    assert "/catalog/schema/vol/rcaputo3/images" in url


@pytest.mark.unit
def test_dir_url_no_user_prefix_without_context(backend):
    url = backend._dir_url("reference")
    assert "/catalog/schema/vol/images" in url
    assert "/rcaputo3/" not in url


@pytest.mark.unit
def test_resolve_display_path_with_user_context(backend, user_ctx):
    display = backend.resolve_display_path("video", "clip.mp4")
    assert display == "/Volumes/catalog/schema/vol/rcaputo3/videos/clip.mp4"


@pytest.mark.unit
def test_resolve_display_path_without_user_context(backend):
    display = backend.resolve_display_path("video", "clip.mp4")
    assert display == "/Volumes/catalog/schema/vol/videos/clip.mp4"


@pytest.mark.unit
async def test_write_uses_user_prefixed_path(backend, mocker, mock_token_response, user_ctx):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mocker.patch.object(backend._client, "put", return_value=_resp(200))

    display = await backend.write("reference", "result.png", b"DATA")

    assert display == "/Volumes/catalog/schema/vol/rcaputo3/images/result.png"


@pytest.mark.unit
async def test_read_uses_user_prefixed_url(backend, mocker, mock_token_response, user_ctx):
    mocker.patch.object(backend._client, "post", return_value=mock_token_response)
    mock_get = mocker.patch.object(backend._client, "get", return_value=_resp(200, content=b"DATA"))

    await backend.read("video", "clip.mp4")

    url = mock_get.call_args.args[0]
    assert "/rcaputo3/videos/clip.mp4" in url
