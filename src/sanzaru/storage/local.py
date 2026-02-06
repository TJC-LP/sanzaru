# SPDX-License-Identifier: MIT
"""Local filesystem storage backend.

Wraps the existing ``security.py`` and ``config.py`` helpers so that switching
to ``LocalStorageBackend`` introduces zero behaviour change for users who keep
the default ``STORAGE_BACKEND=local`` setting.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiofiles

from ..config import get_path
from ..security import check_not_symlink, validate_safe_path
from .protocol import FileInfo, PathType

logger = logging.getLogger("sanzaru")


class LocalStorageBackend:
    """Local-disk storage using the paths from ``VIDEO_PATH`` / ``IMAGE_PATH`` / ``AUDIO_PATH``.

    Args:
        path_overrides: Optional mapping of path_type → Path used in tests to
            redirect I/O into ``tmp_path`` fixtures without touching env vars.
    """

    def __init__(self, path_overrides: dict[str, pathlib.Path] | None = None) -> None:
        self._overrides = path_overrides or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base(self, path_type: PathType) -> pathlib.Path:
        if path_type in self._overrides:
            return self._overrides[path_type]
        return get_path(path_type)

    def _safe(self, path_type: PathType, filename: str, *, allow_create: bool = False) -> pathlib.Path:
        return validate_safe_path(self._base(path_type), filename, allow_create=allow_create)

    # ------------------------------------------------------------------
    # Byte-level I/O
    # ------------------------------------------------------------------

    def _check_symlink(self, path_type: PathType, filename: str) -> None:
        """Check for symlinks on the unresolved path (before validate_safe_path resolves it)."""
        check_not_symlink(self._base(path_type) / filename, f"{path_type} file")

    async def read(self, path_type: PathType, filename: str) -> bytes:
        self._check_symlink(path_type, filename)
        file_path = self._safe(path_type, filename)
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def write(self, path_type: PathType, filename: str, data: bytes) -> str:
        file_path = self._safe(path_type, filename, allow_create=True)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)
        return str(file_path)

    async def write_stream(self, path_type: PathType, filename: str, chunks: AsyncIterator[bytes]) -> str:
        file_path = self._safe(path_type, filename, allow_create=True)
        async with aiofiles.open(file_path, "wb") as f:
            async for chunk in chunks:
                await f.write(chunk)
        return str(file_path)

    # ------------------------------------------------------------------
    # Metadata / listing
    # ------------------------------------------------------------------

    async def list_files(
        self,
        path_type: PathType,
        pattern: str = "*",
        extensions: set[str] | None = None,
    ) -> list[FileInfo]:
        base = self._base(path_type)
        results: list[FileInfo] = []
        for file_path in base.glob(pattern):
            if not file_path.is_file():
                continue
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            # Security: stay within base
            try:
                file_path.resolve().relative_to(base)
            except ValueError:
                logger.debug("Skipping file outside base path: %s", file_path)
                continue
            st = file_path.stat()
            results.append(FileInfo(name=file_path.name, size_bytes=st.st_size, modified_timestamp=st.st_mtime))
        return results

    async def stat(self, path_type: PathType, filename: str) -> FileInfo:
        file_path = self._safe(path_type, filename)
        try:
            st = file_path.stat()
        except OSError as e:
            raise FileNotFoundError(f"Cannot stat file: {e}") from e
        return FileInfo(name=file_path.name, size_bytes=st.st_size, modified_timestamp=st.st_mtime)

    async def exists(self, path_type: PathType, filename: str) -> bool:
        try:
            self._check_symlink(path_type, filename)
            base = self._base(path_type)
            file_path = (base / filename).resolve()
            file_path.relative_to(base)
            return file_path.exists()
        except (ValueError, OSError):
            return False

    # ------------------------------------------------------------------
    # Local-path helpers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def local_path(self, path_type: PathType, filename: str):
        """Yield the real filesystem path (no temp file needed)."""
        self._check_symlink(path_type, filename)
        file_path = self._safe(path_type, filename)
        yield file_path

    @asynccontextmanager
    async def local_tempfile(self, path_type: PathType, filename: str):
        """Yield the actual destination path (no temp file needed)."""
        file_path = self._safe(path_type, filename, allow_create=True)
        yield file_path

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def resolve_display_path(self, path_type: PathType, filename: str) -> str:
        return str(self._safe(path_type, filename, allow_create=True))
