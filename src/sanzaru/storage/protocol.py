# SPDX-License-Identifier: MIT
"""Storage backend protocol and shared types.

Defines the interface that all storage backends must implement.
"""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

PathType = Literal["video", "reference", "audio"]
"""Identifies which base path / directory a file operation targets."""


@dataclass(frozen=True)
class FileInfo:
    """Metadata about a stored file."""

    name: str
    size_bytes: int
    modified_timestamp: float


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable file storage operations.

    All filenames are relative to the backend's configured base path for the
    given *path_type*.  Implementations handle path-traversal prevention
    internally.
    """

    # ------------------------------------------------------------------
    # Byte-level I/O
    # ------------------------------------------------------------------

    async def read(self, path_type: PathType, filename: str) -> bytes:
        """Read entire file contents.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If path traversal detected.
        """
        ...

    async def write(self, path_type: PathType, filename: str, data: bytes) -> str:
        """Write entire file contents.

        Returns:
            A display-friendly path or URI for the written file.
        """
        ...

    async def write_stream(self, path_type: PathType, filename: str, chunks: AsyncIterator[bytes]) -> str:
        """Write file from an async byte-chunk stream (e.g. video download).

        Returns:
            A display-friendly path or URI for the written file.
        """
        ...

    # ------------------------------------------------------------------
    # Metadata / listing
    # ------------------------------------------------------------------

    async def list_files(
        self,
        path_type: PathType,
        pattern: str = "*",
        extensions: set[str] | None = None,
    ) -> list[FileInfo]:
        """List files matching *pattern* and optional *extensions* filter."""
        ...

    async def stat(self, path_type: PathType, filename: str) -> FileInfo:
        """Get file metadata.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        ...

    async def exists(self, path_type: PathType, filename: str) -> bool:
        """Check whether a file exists."""
        ...

    # ------------------------------------------------------------------
    # Local-path helpers (for PIL and other path-based libraries)
    # ------------------------------------------------------------------

    def local_path(self, path_type: PathType, filename: str) -> AbstractAsyncContextManager[pathlib.Path]:
        """Return a context manager yielding a local :class:`pathlib.Path`.

        For the local backend this yields the real path.  Remote backends
        download to a temp file, yield it, then clean up.
        """
        ...

    def local_tempfile(self, path_type: PathType, filename: str) -> AbstractAsyncContextManager[pathlib.Path]:
        """Return a context manager yielding a local temp path for *writing*.

        For the local backend this yields the actual destination.  Remote
        backends yield a temp path and upload on context exit.
        """
        ...

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def resolve_display_path(self, path_type: PathType, filename: str) -> str:
        """Human-readable path or URI for inclusion in tool results."""
        ...
