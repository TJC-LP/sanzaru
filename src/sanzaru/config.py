# SPDX-License-Identifier: MIT
"""Configuration management for sanzaru MCP server.

This module handles:
- OpenAI client initialization
- Environment variable validation
- Path configuration with security checks
- Logging setup
"""

import logging
import os
import pathlib
import sys
from functools import lru_cache
from typing import Literal

from openai import AsyncOpenAI

# ---------- Logging configuration ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # Log to stderr to avoid interfering with stdio MCP transport
)
logger = logging.getLogger("sanzaru")


# ---------- OpenAI client (stateless) ----------
def get_client() -> AsyncOpenAI:
    """Get an OpenAI async client instance.

    Returns:
        Configured AsyncOpenAI client

    Raises:
        RuntimeError: If OPENAI_API_KEY environment variable is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)


# ---------- Path configuration (runtime) ----------
@lru_cache(maxsize=3)
def get_path(path_type: Literal["video", "reference", "audio"]) -> pathlib.Path:
    """Get and validate a configured path from environment.

    Requires explicit environment variable configuration - no defaults.
    Creates paths lazily at runtime, so this works with both `uv run` and `mcp run`.

    Security: Rejects symlinks in environment variable paths to prevent directory traversal.

    Args:
        path_type: Either "video" for VIDEO_PATH, "reference" for IMAGE_PATH, or "audio" for AUDIO_PATH

    Returns:
        Validated absolute path

    Raises:
        RuntimeError: If environment variable not set, malformed, path doesn't exist, isn't a directory, or is a symlink
    """
    if path_type == "video":
        path_str = os.getenv("VIDEO_PATH")
        env_var = "VIDEO_PATH"
        error_name = "Video download directory"
    elif path_type == "reference":
        path_str = os.getenv("IMAGE_PATH")
        env_var = "IMAGE_PATH"
        error_name = "Image directory"
    else:  # audio
        path_str = os.getenv("AUDIO_PATH")
        env_var = "AUDIO_PATH"
        error_name = "Audio files directory"

    # Validate env var is set and not empty/whitespace
    if not path_str or not path_str.strip():
        raise RuntimeError(f"{env_var} environment variable is not set or is empty")

    # Strip whitespace and resolve path with error handling
    try:
        path = pathlib.Path(path_str.strip()).resolve()
    except (ValueError, OSError) as e:
        raise RuntimeError(f"Invalid {error_name} path '{path_str}': {e}") from e

    # Security: Reject symlinks in configured paths (env vars only, not user filenames)
    # Check the original path before resolution to catch symlinks
    original_path = pathlib.Path(path_str.strip())
    try:
        if original_path.exists() and original_path.is_symlink():
            raise RuntimeError(f"{error_name} cannot be a symbolic link: {path_str}")
    except PermissionError as e:
        raise RuntimeError(f"Cannot validate {error_name}: permission denied for {path_str}") from e

    # Validate path exists and is a directory
    if not path.exists():
        raise RuntimeError(f"{env_var}: {error_name} does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"{env_var}: {error_name} is not a directory: {path}")

    return path
