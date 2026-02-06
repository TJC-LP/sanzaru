# SPDX-License-Identifier: MIT
"""Pluggable storage backend for Sanzaru.

The storage layer abstracts all file I/O so Sanzaru can write to local disk
(default) or remote backends like Databricks Unity Catalog Volumes.

Usage::

    from sanzaru.storage import get_storage

    storage = get_storage()
    data = await storage.read("reference", "hero.png")
    await storage.write("video", "output.mp4", video_bytes)
"""

from .factory import get_storage
from .protocol import FileInfo, PathType, StorageBackend

__all__ = ["FileInfo", "PathType", "StorageBackend", "get_storage"]
