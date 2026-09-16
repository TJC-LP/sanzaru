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

import aiofiles
import httpx

from ..user_context import UserContextRequiredError, get_user_context, user_slug
from .protocol import FileInfo, PathType

logger = logging.getLogger("sanzaru")

# How much of a failed response body reaches the log. Databricks puts its error
# code and message in the first line; the rest is rarely worth the log volume.
_LOG_BODY_CHARS = 500

#: Makes a missing per-request identity fatal instead of falling back to the
#: shared volume root. Off by default because this backend also serves
#: single-tenant deployments, where there is no identity to require and the
#: shared root is the right answer.
REQUIRE_USER_CONTEXT_ENV = "SANZARU_REQUIRE_USER_CONTEXT"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"", "0", "false", "no", "off"})


def require_user_context() -> bool:
    """Read ``SANZARU_REQUIRE_USER_CONTEXT`` as a strict boolean.

    Strict because the failure mode of this switch is "every user shares one
    namespace": a permissive parse that recognised only ``1/true/yes`` left
    ``=on``, ``=y`` and ``=enabled`` silently *off*, which is the one outcome an
    operator who set the variable at all cannot have meant. Anything outside
    the two spellings sets is a configuration error, never a default.

    Raises:
        RuntimeError: For an unrecognised value — mapped by the CLI to the
            ``config`` envelope (exit 3), like any other bad environment.
    """
    raw = os.environ.get(REQUIRE_USER_CONTEXT_ENV, "")
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise RuntimeError(
        f"{REQUIRE_USER_CONTEXT_ENV}={raw!r} is not a recognised boolean; "
        "use 1/true/yes/on to require a user identity or 0/false/no/off to allow the shared root"
    )


class StorageRequestError(httpx.HTTPError):
    """A backend request failed, described without describing the backend.

    Subclasses :class:`httpx.HTTPError` because that is already this backend's
    de-facto failure contract — :meth:`DatabricksVolumesBackend.exists` swallows
    it to answer ``False``, and callers upstream catch it the same way. Only the
    *message* changes.
    """

    def __init__(self, operation: str, status_code: int, filename: str | None = None) -> None:
        target = f" for {filename!r}" if filename else ""
        super().__init__(f"Storage {operation} failed{target} (HTTP {status_code})")
        self.operation = operation
        self.status_code = status_code
        self.filename = filename


def _check_response(resp: httpx.Response, operation: str, filename: str | None = None) -> None:
    """Raise a sanitised error for a failed response, sending the full one to the log.

    Stands in for ``resp.raise_for_status()``, whose message embeds the request
    URL — and that message travels: ``FileSystemRepository`` interpolates it into
    an ``AudioFileError`` and FastMCP hands the text to the MCP client. For this
    backend the URL is
    ``{host}/api/2.0/fs/files/Volumes/{catalog}/{schema}/{volume}/{prefix}/{subdir}/{name}``,
    so a caller who asked for one file learns the workspace host, the Unity
    Catalog topology, the media layout and the shape of the per-tenant prefix
    (CWE-209). The operator needs exactly that detail to debug, so it goes to the
    log; the caller gets a status code and the name they themselves supplied.

    The success test is 2xx rather than "not 4xx/5xx" so the substitution is
    exact: ``raise_for_status()`` also refuses a redirect, and a 3xx that
    silently returned its body here would hand the caller the wrong bytes.

    Scope: this closes the *error* path. ``write`` still returns
    ``resolve_display_path()`` — ``/Volumes/{catalog}/{schema}/{volume}/…`` —
    on success, because that display path is the tool contract callers use to
    find their file; it names the volume topology but never the host. httpx
    transport failures (``ConnectError``, timeouts) bypass this function, but
    their ``str()`` is the OS-level message, not the URL, so they do not leak
    it either.
    """
    if resp.is_success:
        return
    try:
        body = resp.text[:_LOG_BODY_CHARS]
    except httpx.ResponseNotRead:
        # Nothing here streams today, but an error path that raises its own
        # exception would bury the status code the caller is about to be told.
        body = "<body not read>"
    logger.error(
        "Databricks %s failed: HTTP %d for %s %s — %s",
        operation,
        resp.status_code,
        resp.request.method,
        resp.request.url,
        body,
    )
    raise StorageRequestError(operation, resp.status_code, filename)


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

        host = os.environ["DATABRICKS_HOST"].rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = f"https://{host}"
        self._host = host
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
        _check_response(resp, "authentication")
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

    def _user_prefix(self) -> str:
        """Return a per-user path segment derived from the current user context.

        When :func:`~sanzaru.user_context.get_user_context` returns a
        ``UserContext``, this method returns a readable-plus-hash segment
        (e.g. ``"rcaputo3-44827c88f857"``) suitable for inserting between the
        volume root and the media subdirectory.  When there is no user context
        (single-tenant mode), returns an empty string.

        This segment is the *only* thing separating one tenant's files from
        another's, so it has to be injective over identities rather than merely
        readable — see :func:`~sanzaru.user_context.user_slug` for what the hash
        half defends against.

        Falling back to the shared root when no identity is present is safe for
        the single-tenant case this backend also serves, and catastrophic for
        the multi-tenant one: with the contextvar unset every user resolved to
        the same namespace, so the isolation the README advertises was simply
        absent (CWE-862). ``SANZARU_REQUIRE_USER_CONTEXT=1`` turns that fallback
        into a refusal, which is what a shared deployment wants — better a
        failed request than one silently served out of everybody's directory.

        Raises:
            UserContextRequiredError: No identity and the switch is on.
            RuntimeError: The switch holds a value that is neither on nor off.
        """
        ctx = get_user_context()
        if ctx is None:
            if require_user_context():
                raise UserContextRequiredError(
                    f"{REQUIRE_USER_CONTEXT_ENV} is set but this request carries no user identity — "
                    "refusing to fall back to the shared volume root"
                )
            return ""
        return user_slug(ctx.email)

    def _validate_filename(self, filename: str) -> str:
        """Sanitise a user-provided filename, rejecting traversal attempts."""
        name = pathlib.PurePosixPath(filename).name
        if not name or name in (".", ".."):
            raise ValueError(f"Invalid filename: {filename}")
        if ".." in filename:
            raise ValueError(f"Path traversal detected: {filename}")
        return name

    def _volume_base(self) -> str:
        """Return the volume base path, optionally including a per-user prefix."""
        prefix = self._user_prefix()
        if prefix:
            return f"{self._volume_path}/{prefix}"
        return self._volume_path

    def _file_url(self, path_type: PathType, filename: str) -> str:
        safe = self._validate_filename(filename)
        subdir = self._subdirs[path_type]
        base = self._volume_base()
        path = f"/Volumes/{base}/{subdir}/{safe}"
        return f"{self._host}/api/2.0/fs/files{quote(path)}"

    def _dir_url(self, path_type: PathType) -> str:
        subdir = self._subdirs[path_type]
        base = self._volume_base()
        path = f"/Volumes/{base}/{subdir}"
        return f"{self._host}/api/2.0/fs/directories{quote(path)}"

    # ------------------------------------------------------------------
    # Byte-level I/O
    # ------------------------------------------------------------------

    async def read(self, path_type: PathType, filename: str) -> bytes:
        safe = self._validate_filename(filename)
        headers = await self._headers()
        resp = await self._client.get(self._file_url(path_type, filename), headers=headers)
        if resp.status_code == 404:
            raise FileNotFoundError(f"File not found: {filename}")
        _check_response(resp, "read", safe)
        return resp.content

    async def read_range(self, path_type: PathType, filename: str, offset: int, length: int) -> bytes:
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        safe = self._validate_filename(filename)
        headers = await self._headers()
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        resp = await self._client.get(self._file_url(path_type, filename), headers=headers)
        if resp.status_code == 404:
            raise FileNotFoundError(f"File not found: {filename}")
        # Accept both 200 (full content) and 206 (partial content). Any other
        # 2xx (a 204, say) is not a range answer and must not be returned as
        # one — `_check_response` lets 2xx through, so name it explicitly.
        if resp.status_code not in (200, 206):
            _check_response(resp, "range read", safe)
            raise StorageRequestError("range read", resp.status_code, safe)
        return resp.content

    async def write(self, path_type: PathType, filename: str, data: bytes) -> str:
        safe = self._validate_filename(filename)
        headers = await self._headers()
        headers["Content-Type"] = "application/octet-stream"
        resp = await self._client.put(self._file_url(path_type, filename), headers=headers, content=data)
        _check_response(resp, "write", safe)
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
        if resp.status_code == 404:
            # The Files API creates parent directories on the first PUT, so a
            # user who has not written anything yet has no directory at all —
            # for them "nothing here" is the truthful answer, not a failure.
            # A list of an empty directory is a 200 with no contents; only the
            # directory's own absence is a 404. (Observed on a fresh per-user
            # prefix right after the 0.11 slug change: every list_* tool failed
            # with "Storage list failed (HTTP 404)" until the first upload.)
            return []
        _check_response(resp, "list")

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
        safe = self._validate_filename(filename)
        headers = await self._headers()
        resp = await self._client.head(self._file_url(path_type, filename), headers=headers)
        if resp.status_code == 404:
            raise FileNotFoundError(f"File not found: {filename}")
        _check_response(resp, "stat", safe)
        return FileInfo(
            name=safe,
            size_bytes=int(resp.headers.get("Content-Length", 0)),
            modified_timestamp=0.0,  # HEAD doesn't return mtime
        )

    async def exists(self, path_type: PathType, filename: str) -> bool:
        """Answer False for a bad name or a failed request, never for a refused identity.

        ``UserContextRequiredError`` (a ``RuntimeError``) from ``_user_prefix``
        is deliberately NOT swallowed: "this deployment will not resolve a
        namespace for you" is a different answer from "no such file", and
        collapsing the two would make a resume believe its checkpoints were
        gone and pay to record them again.
        """
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
        tmp.close()
        try:
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(data)
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
        base = self._volume_base()
        return f"/Volumes/{base}/{subdir}/{safe}"
