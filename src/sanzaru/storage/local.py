# SPDX-License-Identifier: MIT
"""Local filesystem storage backend.

Wraps the existing ``security.py`` and ``config.py`` helpers so that switching
to ``LocalStorageBackend`` introduces zero behaviour change for users who keep
the default ``STORAGE_BACKEND=local`` setting.
"""

from __future__ import annotations

import errno
import logging
import os
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiofiles

from ..config import get_path
from ..security import check_not_symlink, validate_safe_path
from .protocol import FileInfo, PathType

logger = logging.getLogger("sanzaru")


# POSIX-only; Windows has no symlink-following open to refuse, so the flag
# degrades to a no-op there and the pre-open `check_not_symlink` is what remains.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _no_follow(path: str, flags: int) -> int:
    """`open()` opener that refuses to traverse a final-component symlink.

    Containment is checked with `validate_safe_path()` and the file is opened
    afterwards *by name*, so the two steps are not atomic: an attacker with
    write access inside the media directory can rename a symlink onto the
    validated name in between and have the server read or truncate whatever it
    points at (CWE-367). The pre-open `check_not_symlink` only ever saw the
    pre-swap state; `O_NOFOLLOW` moves the decision into the kernel, at the
    moment of the open, where the race has nowhere left to run.

    Only the final component is protected. Directory components are the
    configured media path itself, which is trusted — if that is attacker-
    controlled, confinement was never meaningful.
    """
    try:
        return os.open(path, flags | _NOFOLLOW)
    except OSError as exc:
        # FreeBSD answers EMLINK where Linux and macOS say ELOOP.
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError(f"Refusing to follow a symbolic link: {os.path.basename(path)}") from exc
        raise


class LocalStorageBackend:
    """Local-disk storage using paths from ``SANZARU_MEDIA_PATH`` (or legacy individual vars).

    Args:
        path_overrides: Optional mapping of path_type → Path used in tests to
            redirect I/O into ``tmp_path`` fixtures without touching env vars.
        file_overrides: Optional mapping of (path_type, basename) → the directory
            that one file lives in, overriding ``path_overrides`` for it alone.
            One CLI invocation installs a single backend and then reads its
            inputs concurrently, so a batch spanning several directories cannot
            swap backends per file — the per-file knowledge has to live inside
            one instance (#38). Directory-wide operations (``list_files``) still
            use the path-type override; only named-file lookups consult this.
    """

    def __init__(
        self,
        path_overrides: dict[str, pathlib.Path] | None = None,
        file_overrides: dict[tuple[str, str], pathlib.Path] | None = None,
    ) -> None:
        self._overrides = path_overrides or {}
        self._file_overrides = file_overrides or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base(self, path_type: PathType, filename: str | None = None) -> pathlib.Path:
        if filename is not None:
            per_file = self._file_overrides.get((path_type, filename))
            if per_file is not None:
                return per_file
        if path_type in self._overrides:
            return self._overrides[path_type]
        return get_path(path_type)

    def _safe(self, path_type: PathType, filename: str, *, allow_create: bool = False) -> pathlib.Path:
        return validate_safe_path(self._base(path_type, filename), filename, allow_create=allow_create)

    # ------------------------------------------------------------------
    # Byte-level I/O
    # ------------------------------------------------------------------

    def _check_symlink(self, path_type: PathType, filename: str) -> None:
        """Check for symlinks on the unresolved path (before validate_safe_path resolves it)."""
        check_not_symlink(self._base(path_type, filename) / filename, f"{path_type} file")

    async def read(self, path_type: PathType, filename: str) -> bytes:
        self._check_symlink(path_type, filename)
        file_path = self._safe(path_type, filename)
        async with aiofiles.open(file_path, "rb", opener=_no_follow) as f:
            return await f.read()

    async def read_range(self, path_type: PathType, filename: str, offset: int, length: int) -> bytes:
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        self._check_symlink(path_type, filename)
        file_path = self._safe(path_type, filename)
        async with aiofiles.open(file_path, "rb", opener=_no_follow) as f:
            await f.seek(offset)
            return await f.read(length)

    async def write(self, path_type: PathType, filename: str, data: bytes) -> str:
        file_path = self._safe(path_type, filename, allow_create=True)
        async with aiofiles.open(file_path, "wb", opener=_no_follow) as f:
            await f.write(data)
        return str(file_path)

    async def write_stream(self, path_type: PathType, filename: str, chunks: AsyncIterator[bytes]) -> str:
        file_path = self._safe(path_type, filename, allow_create=True)
        async with aiofiles.open(file_path, "wb", opener=_no_follow) as f:
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
            base = self._base(path_type, filename)
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
        """Yield the actual destination path (no temp file needed).

        The symlink check that `local_path` has is needed here more, not less:
        this is the *write* side, and the caller (pydub, `shutil.copyfile`, PIL)
        opens the path itself, so `_no_follow` never gets a say. Without it a
        link planted at the destination was written straight through.

        This closes the pre-planted case only. A link introduced between here
        and the caller's open is still a race this cannot win — the opener is
        third-party code. `validate_safe_path` bounds the damage in that window
        to somewhere inside the media directory.
        """
        self._check_symlink(path_type, filename)
        file_path = self._safe(path_type, filename, allow_create=True)
        yield file_path

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def resolve_display_path(self, path_type: PathType, filename: str) -> str:
        return str(self._safe(path_type, filename, allow_create=True))
