"""Feature detection for optional sanzaru capabilities.

Detects which optional features are available based on:
1. Path configuration (environment variables)
2. Installed dependencies

If a path is not configured, the feature is disabled regardless of dependencies.
"""

import logging
import os

logger = logging.getLogger("sanzaru")


def _is_path_configured(env_var: str) -> bool:
    """Check if a media path is configured (individual or unified).

    Args:
        env_var: The individual environment variable name (e.g. "VIDEO_PATH")

    Returns:
        True if the individual env var or SANZARU_MEDIA_PATH is set
    """
    if os.getenv(env_var):
        return True
    return bool(os.getenv("SANZARU_MEDIA_PATH"))


def check_video_available() -> bool:
    """Check if video feature is enabled.

    Video feature requires VIDEO_PATH or SANZARU_MEDIA_PATH to be set.
    No extra dependencies required beyond base OpenAI client.

    Returns:
        True if video path is configured, False otherwise
    """
    if not _is_path_configured("VIDEO_PATH"):
        logger.info("Video path not configured - video tools disabled")
        return False
    return True


def check_audio_available() -> bool:
    """Check if audio feature is enabled.

    Requires:
    1. AUDIO_PATH or SANZARU_MEDIA_PATH set
    2. Dependencies: pydub, ffmpeg-python

    Returns:
        True if audio path configured and dependencies installed, False otherwise
    """
    if not _is_path_configured("AUDIO_PATH"):
        logger.info("Audio path not configured - audio tools disabled")
        return False

    try:
        import ffmpeg  # noqa: F401 # type: ignore[import-untyped]
        import pydub  # noqa: F401 # type: ignore[import-untyped]

        logger.info("Audio path configured and dependencies detected - audio tools available")
        return True
    except ImportError as e:
        logger.warning(f"Audio path set but dependencies not available - audio tools disabled: {e}")
        return False


def check_image_available() -> bool:
    """Check if image feature is enabled.

    Requires:
    1. IMAGE_PATH or SANZARU_MEDIA_PATH set
    2. Dependencies: pillow

    Returns:
        True if image path configured and dependencies installed, False otherwise
    """
    if not _is_path_configured("IMAGE_PATH"):
        logger.info("Image path not configured - image tools disabled")
        return False

    try:
        import PIL  # noqa: F401

        logger.info("Image path configured and dependencies detected - image tools available")
        return True
    except ImportError as e:
        logger.warning(f"Image path set but dependencies not available - image tools disabled: {e}")
        return False


def check_databricks_storage() -> bool:
    """Check if Databricks storage backend is configured.

    Requires:
    1. STORAGE_BACKEND environment variable set to "databricks"
    2. All Databricks credentials configured
    3. Volume path via DATABRICKS_VOLUME_PATH or SANZARU_MEDIA_PATH

    Returns:
        True if all required Databricks env vars are present, False otherwise
    """
    if os.getenv("STORAGE_BACKEND", "local").lower() != "databricks":
        return False

    required = ["DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"]
    missing = [v for v in required if not os.getenv(v)]

    # Volume path: DATABRICKS_VOLUME_PATH > SANZARU_MEDIA_PATH
    if not (os.getenv("DATABRICKS_VOLUME_PATH") or os.getenv("SANZARU_MEDIA_PATH")):
        missing.append("DATABRICKS_VOLUME_PATH")

    if missing:
        logger.warning("STORAGE_BACKEND=databricks but missing env vars: %s", missing)
        return False

    logger.info("Databricks storage backend configured")
    return True


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
