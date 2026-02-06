# SPDX-License-Identifier: MIT
"""Caching utilities for audio file metadata.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from collections.abc import Awaitable, Callable
from functools import _CacheInfo

from async_lru import alru_cache

from ..audio.models import FilePathSupportParams


# Global cached function with maxsize=32
@alru_cache(maxsize=32)
async def get_cached_audio_file_support(
    filename: str,
    mtime: float,
    get_support_func: Callable[[str], Awaitable[FilePathSupportParams]],
) -> FilePathSupportParams:
    """Get cached audio file support using async LRU cache.

    Uses filename and modification time as cache key to ensure
    cache invalidation when files are modified.

    Args:
        filename: Name of the audio file.
        mtime: File modification time (Unix timestamp) - used as cache key.
        get_support_func: Async function to get file support info.

    Returns:
        FilePathSupportParams: Cached or freshly computed file support info.
    """
    return await get_support_func(filename)


def clear_global_cache() -> None:
    """Clear the global audio file cache."""
    get_cached_audio_file_support.cache_clear()


def get_global_cache_info() -> _CacheInfo:
    """Get statistics for the global cache.

    Returns:
        _CacheInfo: Named tuple with cache statistics (hits, misses, maxsize, currsize).
    """
    return get_cached_audio_file_support.cache_info()
