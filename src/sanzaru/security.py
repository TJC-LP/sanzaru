# SPDX-License-Identifier: MIT
"""Security utilities for file path validation and safe file operations.

This module provides reusable functions to prevent common security issues:
- Path traversal attacks
- Symlink exploitation
- Standardized error handling for file I/O
"""

import pathlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import IO, Any, Literal

import aiofiles


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
    try:
        if path.exists() and path.is_symlink():
            raise ValueError(f"{error_context} cannot be a symbolic link: {path.name}")
    except PermissionError as e:
        raise RuntimeError(f"Cannot validate {error_context}: permission denied for {path}") from e


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
        check_symlink: If True, verify file is not a symlink before opening

    Yields:
        Open file handle

    Raises:
        ValueError: If file operation fails (file not found, permission denied, symlink detected, etc.)
    """
    # Security check: prevent symlink exploitation
    if check_symlink:
        check_not_symlink(path, error_context)

    try:
        with open(path, mode) as f:
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
        check_symlink: If True, verify file is not a symlink before opening

    Yields:
        Async file handle from aiofiles

    Raises:
        ValueError: If file operation fails (file not found, permission denied, symlink detected, etc.)
    """
    # Security check: prevent symlink exploitation
    if check_symlink:
        check_not_symlink(path, error_context)

    try:
        async with aiofiles.open(path, mode) as f:
            yield f
    except FileNotFoundError as e:
        raise ValueError(f"{error_context} not found: {path.name}") from e
    except PermissionError as e:
        action = "reading" if "r" in mode else "writing"
        raise ValueError(f"Permission denied {action} {error_context}: {path.name}") from e
    except OSError as e:
        action = "reading" if "r" in mode else "writing"
        raise ValueError(f"Error {action} {error_context}: {e}") from e
