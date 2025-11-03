"""Feature detection for optional sanzaru capabilities.

Detects which optional features are available based on:
1. Path configuration (environment variables)
2. Installed dependencies

If a path is not configured, the feature is disabled regardless of dependencies.
"""

import logging
import os

logger = logging.getLogger("sanzaru")


def check_video_available() -> bool:
    """Check if video feature is enabled.

    Video feature requires VIDEO_PATH environment variable to be set.
    No extra dependencies required beyond base OpenAI client.

    Returns:
        True if VIDEO_PATH is configured, False otherwise
    """
    if os.getenv("VIDEO_PATH") is None:
        logger.info("VIDEO_PATH not set - video tools disabled")
        return False
    return True


def check_audio_available() -> bool:
    """Check if audio feature is enabled.

    Requires:
    1. AUDIO_PATH environment variable set
    2. Dependencies: pydub, ffmpeg-python

    Returns:
        True if AUDIO_PATH configured and dependencies installed, False otherwise
    """
    if os.getenv("AUDIO_PATH") is None:
        logger.info("AUDIO_PATH not set - audio tools disabled")
        return False

    try:
        import ffmpeg  # noqa: F401 # type: ignore[import-untyped]
        import pydub  # noqa: F401 # type: ignore[import-untyped]

        logger.info("AUDIO_PATH configured and dependencies detected - audio tools available")
        return True
    except ImportError as e:
        logger.warning(f"AUDIO_PATH set but dependencies not available - audio tools disabled: {e}")
        return False


def check_image_available() -> bool:
    """Check if image feature is enabled.

    Requires:
    1. IMAGE_PATH environment variable set
    2. Dependencies: pillow

    Returns:
        True if IMAGE_PATH configured and dependencies installed, False otherwise
    """
    if os.getenv("IMAGE_PATH") is None:
        logger.info("IMAGE_PATH not set - image tools disabled")
        return False

    try:
        import PIL  # noqa: F401

        logger.info("IMAGE_PATH configured and dependencies detected - image tools available")
        return True
    except ImportError as e:
        logger.warning(f"IMAGE_PATH set but dependencies not available - image tools disabled: {e}")
        return False


def get_available_features() -> dict[str, bool]:
    """Get a dictionary of available features.

    Returns:
        Dict mapping feature name to availability status
    """
    return {
        "video": check_video_available(),
        "audio": check_audio_available(),
        "image": check_image_available(),
    }
