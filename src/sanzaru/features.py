"""Feature detection for optional sanzaru capabilities.

Detects which optional features are available based on installed dependencies.
"""

import logging

logger = logging.getLogger("sanzaru")


def check_video_available() -> bool:
    """Check if video feature dependencies are available.

    Video feature has no extra dependencies beyond base OpenAI client.
    Always available.

    Returns:
        True (video always available with base install)
    """
    return True


def check_audio_available() -> bool:
    """Check if audio feature dependencies are available.

    Requires: pydub, ffmpeg-python

    Returns:
        True if audio dependencies installed, False otherwise
    """
    try:
        import ffmpeg  # noqa: F401 # type: ignore[import-untyped]
        import pydub  # noqa: F401 # type: ignore[import-untyped]

        logger.info("Audio dependencies detected - audio tools available")
        return True
    except ImportError as e:
        logger.info(f"Audio dependencies not available - audio tools disabled: {e}")
        return False


def check_image_available() -> bool:
    """Check if image feature dependencies are available.

    Requires: pillow

    Returns:
        True if image dependencies installed, False otherwise
    """
    try:
        import PIL  # noqa: F401

        logger.info("Image dependencies detected - image tools available")
        return True
    except ImportError as e:
        logger.info(f"Image dependencies not available - image tools disabled: {e}")
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
