# SPDX-License-Identifier: MIT
"""Audio feature configuration for sanzaru.

Manages AUDIO_FILES_PATH validation and audio-specific settings.

NOTE: This module exists separately from root config.py because:
- Uses pydantic-settings for auto env loading (different pattern from root config's explicit get_path())
- Migrated from mcp-server-whisper which used this pattern
- Provides AudioConfig class for structured validation vs root config's function-based approach
- Eventually, this could be consolidated into root config.py if unified configuration is desired

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from ..exceptions import ConfigurationError


class AudioConfig(BaseSettings):
    """Configuration for audio feature.

    Loads AUDIO_FILES_PATH from environment with validation.
    """

    audio_files_path: Path = Field(
        ...,
        description="Path to the directory containing audio files",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "arbitrary_types_allowed": True,
    }

    @field_validator("audio_files_path")
    @classmethod
    def validate_audio_path(cls, v: Path) -> Path:
        """Validate that the audio path exists and is a directory."""
        # Strip whitespace and resolve path
        if isinstance(v, str):
            v = Path(v.strip())

        resolved_path = v.resolve()

        # Check for symlinks (security)
        if v.exists() and v.is_symlink():
            raise ConfigurationError(f"Audio path cannot be a symbolic link: {v}")

        # Validate existence and type
        if not resolved_path.exists():
            raise ConfigurationError(f"Audio path does not exist: {resolved_path}")
        if not resolved_path.is_dir():
            raise ConfigurationError(f"Audio path is not a directory: {resolved_path}")

        return resolved_path


@lru_cache
def get_audio_config() -> AudioConfig:
    """Get the audio configuration (cached singleton).

    Returns:
        AudioConfig: The validated configuration object.

    Raises:
        ConfigurationError: If configuration is invalid or missing.
    """
    try:
        return AudioConfig()  # type: ignore
    except Exception as e:
        raise ConfigurationError(f"Failed to load audio configuration: {e}") from e


def get_audio_path() -> Path:
    """Get the validated audio files path.

    This is a convenience function that extracts just the path from config.

    Returns:
        Path: The validated audio files path.

    Raises:
        ConfigurationError: If AUDIO_FILES_PATH is not set or invalid.
    """
    config = get_audio_config()
    return config.audio_files_path
