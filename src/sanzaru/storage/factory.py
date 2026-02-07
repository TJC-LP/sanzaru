# SPDX-License-Identifier: MIT
"""Storage backend factory.

Reads ``STORAGE_BACKEND`` env var (default ``"local"``) and returns the
appropriate singleton backend instance.
"""

from __future__ import annotations

import atexit
import logging
import os
from functools import lru_cache

from .local import LocalStorageBackend
from .protocol import StorageBackend

logger = logging.getLogger("sanzaru")


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    """Return the configured :class:`StorageBackend` (cached singleton).

    The httpx client used by remote backends (e.g. Databricks) is closed
    automatically at process exit via :func:`atexit`.

    Configuration
    -------------
    ``STORAGE_BACKEND``
        ``"local"`` (default) – uses ``VIDEO_PATH`` / ``IMAGE_PATH`` / ``AUDIO_PATH``.
        ``"databricks"`` – uses Databricks Unity Catalog Volumes via the Files API.
            Requires ``DATABRICKS_HOST``, ``DATABRICKS_CLIENT_ID``,
            ``DATABRICKS_CLIENT_SECRET``, and ``DATABRICKS_VOLUME_PATH``.
    """
    backend_type = os.getenv("STORAGE_BACKEND", "local").lower()

    if backend_type == "local":
        return LocalStorageBackend()

    if backend_type == "databricks":
        try:
            from .databricks import DatabricksVolumesBackend
        except ImportError as exc:
            raise RuntimeError(
                "Databricks storage backend requires extra dependencies. "
                "Install with: pip install 'sanzaru[databricks]'"
            ) from exc
        backend = DatabricksVolumesBackend()
        _register_cleanup(backend)
        return backend

    raise RuntimeError(f"Unknown STORAGE_BACKEND: {backend_type!r}. Use 'local' or 'databricks'.")


def _register_cleanup(backend: StorageBackend) -> None:
    """Register an atexit handler to close the backend's httpx client."""

    def _cleanup() -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(backend.aclose())  # type: ignore[attr-defined]
        except RuntimeError:
            # No running loop — run synchronously
            asyncio.run(backend.aclose())  # type: ignore[attr-defined]
        logger.debug("Storage backend httpx client closed")

    atexit.register(_cleanup)
