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


_backend_override: StorageBackend | None = None


def set_storage_backend(backend: StorageBackend | None) -> None:
    """Install a process-wide backend override; None restores default resolution.

    Used by the CLI to redirect file I/O to caller-chosen directories via
    ``LocalStorageBackend(path_overrides=...)``. The MCP server never sets this.
    """
    global _backend_override
    _backend_override = backend


def get_storage() -> StorageBackend:
    """Return the configured :class:`StorageBackend`.

    Returns the installed override (see :func:`set_storage_backend`) when one
    is active; otherwise the env-configured cached singleton.
    """
    if _backend_override is not None:
        return _backend_override
    return _default_storage()


def get_default_storage() -> StorageBackend:
    """Return the env-configured backend, ignoring any installed override.

    The override exists so ``-o`` can repoint a whole path type for one CLI
    invocation. Anything addressed by *identity* rather than by output path —
    a simulated podcast's run manifest and act checkpoints, which a later
    ``--resume RUN_ID`` has to find with no ``-o`` in hand — must resolve
    against the media library instead, which is what this returns.
    """
    return _default_storage()


@lru_cache(maxsize=1)
def _default_storage() -> StorageBackend:
    """Build the env-configured backend (cached singleton).

    The httpx client used by remote backends (e.g. Databricks) is closed
    automatically at process exit via :func:`atexit`.

    Configuration
    -------------
    ``STORAGE_BACKEND``
        ``"local"`` (default) – uses ``SANZARU_MEDIA_PATH`` (or individual path vars).
        ``"databricks"`` – uses Databricks Unity Catalog Volumes via the Files API.
            Requires ``DATABRICKS_HOST``, ``DATABRICKS_CLIENT_ID``,
            ``DATABRICKS_CLIENT_SECRET``, and ``DATABRICKS_VOLUME_PATH``
            (or ``SANZARU_MEDIA_PATH``).
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
