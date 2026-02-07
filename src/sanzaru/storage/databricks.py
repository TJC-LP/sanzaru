# SPDX-License-Identifier: MIT
"""Databricks Unity Catalog Volumes storage backend.

Uses the Databricks Files API (REST) for all file operations, with
OAuth client credentials for authentication.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import pathlib
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx

from .protocol import FileInfo, PathType

logger = logging.getLogger("sanzaru")


class DatabricksVolumesBackend:
    """Databricks Unity Catalog Volumes storage backend.

    Reads/writes files via the Databricks Files API and lists directories
    via the Directories API.  Authentication uses OAuth 2.0 client
    credentials with automatic token caching and refresh.

    Required environment variables::

        DATABRICKS_HOST            https://adb-123.11.azuredatabricks.net
        DATABRICKS_CLIENT_ID       OAuth service principal client ID
        DATABRICKS_CLIENT_SECRET   OAuth service principal client secret
        DATABRICKS_VOLUME_PATH     Unity Catalog volume path (or SANZARU_MEDIA_PATH fallback)

    Optional environment variables::

        DATABRICKS_VIDEO_DIR       Subdirectory for videos (default: "videos")
        DATABRICKS_IMAGE_DIR       Subdirectory for images (default: "images")
        DATABRICKS_AUDIO_DIR       Subdirectory for audio  (default: "audio")
    """

    def __init__(self) -> None:
        # Volume path: DATABRICKS_VOLUME_PATH > SANZARU_MEDIA_PATH
        volume_path = os.getenv("DATABRICKS_VOLUME_PATH", "").strip() or os.getenv("SANZARU_MEDIA_PATH", "").strip()

        required = {
            "DATABRICKS_HOST": "Workspace URL (e.g. https://adb-123.azuredatabricks.net)",
            "DATABRICKS_CLIENT_ID": "OAuth service principal client ID",
            "DATABRICKS_CLIENT_SECRET": "OAuth service principal client secret",
        }
        missing = [name for name in required if name not in os.environ or not os.environ[name].strip()]
        if not volume_path:
            missing.append("DATABRICKS_VOLUME_PATH")
        if missing:
            details = "\n".join(
                f"  - {name}: {required.get(name, 'Unity Catalog volume path (or set SANZARU_MEDIA_PATH)')}"
                for name in missing
            )
            raise RuntimeError(f"Missing required Databricks environment variable(s):\n{details}")

        self._host = os.environ["DATABRICKS_HOST"].rstrip("/")
        self._client_id = os.environ["DATABRICKS_CLIENT_ID"]
        self._client_secret = os.environ["DATABRICKS_CLIENT_SECRET"]
        self._volume_path = volume_path.strip("/")

        self._subdirs: dict[str, str] = {
            "video": os.getenv("DATABRICKS_VIDEO_DIR", "videos"),
            "reference": os.getenv("DATABRICKS_IMAGE_DIR", "images"),
            "audio": os.getenv("DATABRICKS_AUDIO_DIR", "audio"),
        }

        self._client = httpx.AsyncClient(timeout=300.0)
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying httpx client and release resources.

        Should be called during application shutdown.  The backend is
        typically a singleton, so this is called once at process exit.
        """
        await self._client.aclose()

    async def __aenter__(self) -> DatabricksVolumesBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _get_token(self) -> str:
        """Get an OAuth token, refreshing if expired or near-expiry."""
        now = time.monotonic()
        if self._token and now < self._token_expires_at - 60:
            return self._token

        resp = await self._client.post(
            f"{self._host}/oidc/v1/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "all-apis",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        # Default to 1-hour expiry if not provided
        self._token_expires_at = now + payload.get("expires_in", 3600)
        logger.debug("Acquired Databricks OAuth token (expires in %ds)", payload.get("expires_in", 3600))
        return self._token

    async def _headers(self) -> dict[str, str]:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _validate_filename(self, filename: str) -> str:
        """Sanitise a user-provided filename, rejecting traversal attempts."""
        name = pathlib.PurePosixPath(filename).name
        if not name or name in (".", ".."):
            raise ValueError(f"Invalid filename: {filename}")
        if ".." in filename:
            raise ValueError(f"Path traversal detected: {filename}")
        return name

    def _file_url(self, path_type: PathType, filename: str) -> str:
        safe = self._validate_filename(filename)
        subdir = self._subdirs[path_type]
        path = f"/Volumes/{self._volume_path}/{subdir}/{safe}"
        return f"{self._host}/api/2.0/fs/files{quote(path)}"

    def _dir_url(self, path_type: PathType) -> str:
        subdir = self._subdirs[path_type]
        path = f"/Volumes/{self._volume_path}/{subdir}"
        return f"{self._host}/api/2.0/fs/directories{quote(path)}"

    # ------------------------------------------------------------------
    # Byte-level I/O
    # ------------------------------------------------------------------

    async def read(self, path_type: PathType, filename: str) -> bytes:
        headers = await self._headers()
        resp = await self._client.get(self._file_url(path_type, filename), headers=headers)
        if resp.status_code == 404:
            raise FileNotFoundError(f"File not found: {filename}")
        resp.raise_for_status()
        return resp.content

    async def write(self, path_type: PathType, filename: str, data: bytes) -> str:
        self._validate_filename(filename)
        headers = await self._headers()
        headers["Content-Type"] = "application/octet-stream"
        resp = await self._client.put(self._file_url(path_type, filename), headers=headers, content=data)
        resp.raise_for_status()
        return self.resolve_display_path(path_type, filename)

    async def write_stream(self, path_type: PathType, filename: str, chunks: AsyncIterator[bytes]) -> str:
        """Write file from an async byte-chunk stream.

        .. warning::

            **Full in-memory buffering.** The Databricks Files API requires a
            complete PUT request body — it does not support chunked transfer
            encoding or multipart uploads.  This method therefore buffers the
            entire stream in memory before uploading.

            For Sora videos (8-12 s at 720p ≈ 20-60 MB) this is acceptable.
            For very large files, monitor memory usage in your deployment.
        """
        # WARNING: entire stream buffered in memory (Databricks API limitation)
        buf = bytearray()
        async for chunk in chunks:
            buf.extend(chunk)
        return await self.write(path_type, filename, bytes(buf))

    # ------------------------------------------------------------------
    # Metadata / listing
    # ------------------------------------------------------------------

    async def list_files(
        self,
        path_type: PathType,
        pattern: str = "*",
        extensions: set[str] | None = None,
    ) -> list[FileInfo]:
        headers = await self._headers()
        resp = await self._client.get(self._dir_url(path_type), headers=headers)
        resp.raise_for_status()

        results: list[FileInfo] = []
        for entry in resp.json().get("contents", []):
            if entry.get("is_directory", False):
                continue
            name = entry.get("name", "")
            if not name:
                continue

            # Extension filter
            if extensions:
                ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
                if ext not in extensions:
                    continue

            # Glob pattern filter
            if pattern != "*" and not fnmatch.fnmatch(name, pattern):
                continue

            results.append(
                FileInfo(
                    name=name,
                    size_bytes=entry.get("file_size", 0),
                    modified_timestamp=entry.get("last_modified", 0) / 1000.0,
                )
            )
        return results

    async def stat(self, path_type: PathType, filename: str) -> FileInfo:
        headers = await self._headers()
        resp = await self._client.head(self._file_url(path_type, filename), headers=headers)
        if resp.status_code == 404:
            raise FileNotFoundError(f"File not found: {filename}")
        resp.raise_for_status()
        return FileInfo(
            name=self._validate_filename(filename),
            size_bytes=int(resp.headers.get("Content-Length", 0)),
            modified_timestamp=0.0,  # HEAD doesn't return mtime
        )

    async def exists(self, path_type: PathType, filename: str) -> bool:
        try:
            self._validate_filename(filename)
            headers = await self._headers()
            resp = await self._client.head(self._file_url(path_type, filename), headers=headers)
            return resp.status_code == 200
        except (ValueError, httpx.HTTPError):
            return False

    # ------------------------------------------------------------------
    # Local-path helpers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def local_path(self, path_type: PathType, filename: str):
        """Download file to a temp path for libraries that need local files."""
        data = await self.read(path_type, filename)
        suffix = pathlib.PurePosixPath(filename).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115
        tmp_path = pathlib.Path(tmp.name)
        try:
            tmp.write(data)
            tmp.close()
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)

    @asynccontextmanager
    async def local_tempfile(self, path_type: PathType, filename: str):
        """Yield a temp path for writing; upload to Volumes on context exit."""
        suffix = pathlib.PurePosixPath(filename).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115
        tmp_path = pathlib.Path(tmp.name)
        tmp.close()
        try:
            yield tmp_path
            # Upload the written file
            content = tmp_path.read_bytes()
            await self.write(path_type, filename, content)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def resolve_display_path(self, path_type: PathType, filename: str) -> str:
        safe = self._validate_filename(filename)
        subdir = self._subdirs[path_type]
        return f"/Volumes/{self._volume_path}/{subdir}/{safe}"
