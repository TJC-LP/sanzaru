# SPDX-License-Identifier: MIT
"""Input media for Higgsfield jobs: what may be uploaded, and how long a source clip runs.

The source clip's duration is part of the price of an edit or extend (input
seconds are billed), so a cap needs it *before* the upload. ffprobe is optional
in sanzaru (it comes with the audio tooling, not the core install), so the
duration is read straight from the MP4's `moov/mvhd` atom — the bytes are
already in hand for the upload — with ffprobe only as a fallback.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import PurePosixPath
from typing import Literal

from ..storage.protocol import PathType

IMAGE_CONTENT_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
VIDEO_CONTENT_TYPES: dict[str, str] = {".mp4": "video/mp4"}
INPUT_CONTENT_TYPES: dict[str, str] = {**IMAGE_CONTENT_TYPES, **VIDEO_CONTENT_TYPES}


def content_type_for(filename: str, kind: Literal["image", "video"]) -> str:
    """The upload content type for `filename`, or ValueError when Higgsfield would refuse it."""
    ext = PurePosixPath(filename).suffix.lower()
    table = IMAGE_CONTENT_TYPES if kind == "image" else VIDEO_CONTENT_TYPES
    if ext not in table:
        if kind == "video":
            raise ValueError(f"{filename!r:.80}: Higgsfield accepts MP4 source videos only")
        raise ValueError(f"{filename!r:.80}: use a JPEG, PNG, WebP or GIF image (got {ext or 'no extension'})")
    return table[ext]


_CONTAINERS = frozenset({b"moov", b"trak", b"mdia"})


def _atoms(data: bytes, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """(type, payload_start, payload_end) for each atom in data[start:end]; stops at the first bad one."""
    out: list[tuple[bytes, int, int]] = []
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                break
            (size,) = struct.unpack_from(">Q", data, pos + 8)
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            break
        out.append((kind, pos + header, pos + size))
        pos += size
    return out


def mp4_duration_seconds(data: bytes) -> float | None:
    """Duration from the movie header (`moov/mvhd`), or None if it cannot be read."""
    for kind, start, end in _atoms(data, 0, len(data)):
        if kind != b"moov":
            continue
        for inner, istart, iend in _atoms(data, start, end):
            if inner != b"mvhd" or iend - istart < 4:
                continue
            version = data[istart]
            try:
                if version == 1:
                    timescale, duration = struct.unpack_from(">IQ", data, istart + 4 + 16)
                else:
                    timescale, duration = struct.unpack_from(">II", data, istart + 4 + 8)
            except struct.error:
                return None
            if timescale == 0 or duration == 0 or duration == 0xFFFFFFFF:
                return None
            return duration / timescale
    return None


async def probe_duration(path_type: PathType, filename: str, data: bytes) -> float | None:
    """Source duration: the mvhd atom first, ffprobe (if installed) second, else None."""
    seconds = mp4_duration_seconds(data)
    if seconds is not None:
        return seconds
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    from ..storage import get_storage
    from ..tools.inspect import _probe_duration, safe_video_demuxer

    demuxer = safe_video_demuxer(filename)
    async with get_storage().local_path(path_type, filename) as path:
        try:
            return await _probe_duration(ffprobe, str(path), demuxer)
        except ValueError:
            return None
