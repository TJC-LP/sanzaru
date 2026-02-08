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

# Mapping from path_type to (individual env var, subdirectory under SANZARU_MEDIA_PATH)
_MEDIA_SUBDIRS: dict[str, tuple[str, str]] = {
    "video": ("VIDEO_PATH", "videos"),
    "reference": ("IMAGE_PATH", "images"),
    "audio": ("AUDIO_PATH", "audio"),
}

_ERROR_NAMES: dict[str, str] = {
    "video": "Video download directory",
    "reference": "Image directory",
    "audio": "Audio files directory",
}


def _resolve_media_path(path_type: Literal["video", "reference", "audio"]) -> tuple[str | None, str, bool]:
    """Resolve path string from individual env var or SANZARU_MEDIA_PATH.

    Priority: individual env var > SANZARU_MEDIA_PATH/{subdir} > None.

    Args:
        path_type: Path type to resolve

    Returns:
        (path_str, env_var_name_for_errors, using_unified) tuple
    """
    env_var, subdir = _MEDIA_SUBDIRS[path_type]

    individual = os.getenv(env_var)
    if individual and individual.strip():
        return individual.strip(), env_var, False

    unified = os.getenv("SANZARU_MEDIA_PATH")
    if unified and unified.strip():
        return os.path.join(unified.strip(), subdir), "SANZARU_MEDIA_PATH", True

    return None, env_var, False


@lru_cache(maxsize=3)
def get_path(path_type: Literal["video", "reference", "audio"]) -> pathlib.Path:
    """Get and validate a configured path from environment.

    Supports two configuration modes:
    1. Individual env vars: VIDEO_PATH, IMAGE_PATH, AUDIO_PATH (take precedence)
    2. Unified root: SANZARU_MEDIA_PATH (auto-creates videos/, images/, audio/ subdirs)

    Creates paths lazily at runtime, so this works with both `uv run` and `mcp run`.

    Security: Rejects symlinks in environment variable paths to prevent directory traversal.

    Args:
        path_type: Either "video" for VIDEO_PATH, "reference" for IMAGE_PATH, or "audio" for AUDIO_PATH

    Returns:
        Validated absolute path

    Raises:
        RuntimeError: If environment variable not set, malformed, path doesn't exist, isn't a directory, or is a symlink
    """
    error_name = _ERROR_NAMES[path_type]
    path_str, env_var, using_unified = _resolve_media_path(path_type)

    # Validate env var is set and not empty/whitespace
    if not path_str:
        individual_var = _MEDIA_SUBDIRS[path_type][0]
        raise RuntimeError(f"{error_name} not configured. Set {individual_var} or SANZARU_MEDIA_PATH")

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

    # Auto-create subdirectories when using unified SANZARU_MEDIA_PATH
    if using_unified and not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Auto-created directory: %s", path)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to auto-create {error_name} at {path}: {e}") from e

    # Validate path exists and is a directory
    if not path.exists():
        raise RuntimeError(f"{env_var}: {error_name} does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"{env_var}: {error_name} is not a directory: {path}")

    return path
