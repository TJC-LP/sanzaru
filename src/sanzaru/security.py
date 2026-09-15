# SPDX-License-Identifier: MIT
"""Security utilities for file path validation and safe file operations.

This module provides reusable functions to prevent common security issues:
- Path traversal attacks
- Symlink exploitation, both pre-planted (`check_not_symlink`) and swapped in
  between validation and open (`open_nofollow`)
- Standardized error handling for file I/O
"""

import errno
import os
import pathlib
import stat
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import IO, Any, Literal

import aiofiles

# POSIX-only. Windows has no symlink-following `open()` flag to set, so the
# opener degrades to a plain open there and the pre-open `check_not_symlink`
# is the only symlink defence — which is why every writer calls it too, not
# just the readers.
O_NOFOLLOW: int = getattr(os, "O_NOFOLLOW", 0)


def open_nofollow(path: str, flags: int) -> int:
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

    Pass as ``opener=`` to ``open()`` / ``aiofiles.open()``. CPython finalises
    the flags (``O_CLOEXEC``, and ``O_BINARY`` on Windows) before calling the
    opener, so nothing has to be added here beyond ``O_NOFOLLOW``.
    """
    try:
        return os.open(path, flags | O_NOFOLLOW)
    except OSError as exc:
        # FreeBSD answers EMLINK where Linux and macOS say ELOOP. Either can
        # also mean a genuine link loop higher up the path, so the message
        # names the open rather than asserting the final component is a link.
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError(f"Refusing to follow a symbolic link while opening {os.path.basename(path)}") from exc
        raise


def validate_safe_path(base_path: pathlib.Path, filename: str, *, allow_create: bool = False) -> pathlib.Path:
    """Validate a filename is safe and construct the full path within base_path.

    Security measures:
    - Prevents path traversal (e.g., "../../../etc/passwd")
    - Validates filename resolves to a path within base_path
    - Optionally checks file exists (when allow_create=False)

    Args:
        base_path: Base directory that must contain the file
        filename: User-provided filename (not full path)
        allow_create: If True, don't require file to exist (for write operations)

    Returns:
        Validated absolute path within base_path

    Raises:
        ValueError: If path traversal detected or file doesn't exist
    """
    # Construct and resolve path
    file_path = base_path / filename
    try:
        file_path = file_path.resolve()
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid filename '{filename}': {e}") from e

    # Security: prevent path traversal - ensure resolved path is within base_path
    try:
        file_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Invalid filename: path traversal detected in '{filename}'") from None

    # Validate existence unless creating new file
    if not allow_create and not file_path.exists():
        raise ValueError(f"File not found: {filename}")

    return file_path


def check_not_symlink(path: pathlib.Path, error_context: str) -> None:
    """Verify a path is not a symbolic link.

    Security: Prevents symlink-based attacks where an attacker could redirect
    file operations to arbitrary locations.

    Args:
        path: Path to check
        error_context: Context for error message (e.g., "Reference image", "Output file")

    Raises:
        ValueError: If path is a symlink
        RuntimeError: If permission denied checking path
    """
    # lstat, alone, and first: `exists()` follows the link, so the old
    # `exists() and is_symlink()` answered False for a *dangling* symlink and
    # skipped the test entirely — the guard let through exactly the links a
    # later `open(..., "wb")` would follow and create the target of. lstat does
    # not follow, so a broken link is still a link here.
    #
    # `os.lstat` rather than `Path.is_symlink()`: the latter is `os.path.islink`,
    # which swallows every OSError and answers False, so a path that cannot be
    # lstat'ed read as "a regular file" and the PermissionError branch below
    # could never fire. Only a missing path is a non-answer here.
    try:
        mode = os.lstat(path).st_mode
    except PermissionError as e:
        raise RuntimeError(f"Cannot validate {error_context}: permission denied for {path}") from e
    except OSError:
        # Missing path, or a file where a directory was expected: nothing to
        # be a link yet. `validate_safe_path` and the open itself report those.
        return
    if stat.S_ISLNK(mode):
        raise ValueError(f"{error_context} cannot be a symbolic link: {path.name}")


@contextmanager
def safe_open_file(
    path: pathlib.Path, mode: str, error_context: str, *, check_symlink: bool = True
) -> Iterator[IO[Any]]:
    """Context manager for safe file operations with standardized error handling.

    Provides consistent error messages across the codebase and handles common failure modes.

    Args:
        path: File path to open
        mode: File mode ("rb" for reading, "wb" for writing)
        error_context: Context for error messages (e.g., "reference image", "video file")
        check_symlink: If True, verify file is not a symlink before opening.
            False skips only that pre-check (for a path this call is about to
            create); the open itself still refuses to follow a symlink.

    Yields:
        Open file handle

    Raises:
        ValueError: If file operation fails (file not found, permission denied, symlink detected, etc.)
    """
    # Security check: prevent symlink exploitation
    if check_symlink:
        check_not_symlink(path, error_context)

    # `opener=`: the pre-open check above only sees the pre-swap state, so a
    # symlink renamed onto the name between the check and this open would be
    # followed. O_NOFOLLOW makes the kernel refuse at the moment of the open.
    try:
        with open(path, mode, opener=open_nofollow) as f:
            yield f
    except FileNotFoundError as e:
        raise ValueError(f"{error_context} not found: {path.name}") from e
    except PermissionError as e:
        action = "reading" if "r" in mode else "writing"
        raise ValueError(f"Permission denied {action} {error_context}: {path.name}") from e
    except OSError as e:
        action = "reading" if "r" in mode else "writing"
        raise ValueError(f"Error {action} {error_context}: {e}") from e


@asynccontextmanager
async def async_safe_open_file(
    path: pathlib.Path, mode: Literal["rb", "wb"], error_context: str, *, check_symlink: bool = True
) -> AsyncIterator[Any]:
    """Async context manager for safe file operations with standardized error handling.

    Provides consistent error messages across the codebase and handles common failure modes.
    Uses aiofiles for non-blocking I/O operations.

    Args:
        path: File path to open
        mode: File mode ("rb" for reading, "wb" for writing)
        error_context: Context for error messages (e.g., "reference image", "video file")
        check_symlink: If True, verify file is not a symlink before opening.
            False skips only that pre-check; the open itself still refuses to
            follow a symlink.

    Yields:
        Async file handle from aiofiles

    Raises:
        ValueError: If file operation fails (file not found, permission denied, symlink detected, etc.)
    """
    # Security check: prevent symlink exploitation
    if check_symlink:
        check_not_symlink(path, error_context)

    # Same opener as `safe_open_file`, for the same race.
    try:
        async with aiofiles.open(path, mode, opener=open_nofollow) as f:
            yield f
    except FileNotFoundError as e:
        raise ValueError(f"{error_context} not found: {path.name}") from e
    except PermissionError as e:
        action = "reading" if "r" in mode else "writing"
        raise ValueError(f"Permission denied {action} {error_context}: {path.name}") from e
    except OSError as e:
        action = "reading" if "r" in mode else "writing"
        raise ValueError(f"Error {action} {error_context}: {e}") from e
