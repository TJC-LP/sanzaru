# SPDX-License-Identifier: MIT
"""Let the model look at the media it just made.

Every other tool here answers with metadata: a filename, a size, a status. That
is enough to *drive* a workflow and not enough to *check* one, so the model has
been refining images it cannot see (`previous_response_id` chains), editing
pictures it cannot compare, cropping references without confirming the subject
survived, and rendering Sora clips with no way to tell whether the motion it
asked for happened. Audio has had an answer since the QC pass landed — transcribe
the render and judge it. These two tools are the visual equivalent.

`view_media` is not this. That opens a player for the *person*; these return
image content to the *model*. The names are deliberately unalike.

Three facts about Claude's vision input shape the design, and none of them are
negotiable from here:

- **A per-image ceiling of 10 MB base64**, so roughly 7.5 MB of bytes. Several
  real gpt-image renders in a working media directory are already past it, which
  means handing over the file as-is is not "expensive", it is *rejected*.
- **Cost is counted in 28-pixel patches**: an image costs
  ``ceil(w / 28) * ceil(h / 28)`` visual tokens. A 4K render is ~4784 tokens
  before it tells you anything; the same picture at a 1568 px long edge is well
  under half that and answers most questions just as well.
- **Anything past a 2576 px long edge is downscaled by the API anyway**, so
  shipping more than that spends transfer on pixels nobody will see.

So the rule is: crop first if asked, then fit the long edge, then encode in the
most faithful format that fits the byte ceiling. Every departure from the
original is stated in the text block, because a model reasoning about a picture
needs to know it is looking at a resized crop and not the artifact on disk.
"""

from __future__ import annotations

import io
import json
import math
import shutil
from typing import Literal

import anyio
from mcp.server.mcpserver.utilities.types import Image
from PIL import Image as PILImage

from ..config import logger
from ..storage import get_storage

#: Long edge we fit to by default. The standard resolution tier's own ceiling —
#: below it nothing is downscaled twice, and it is ample for judging composition,
#: colour, framing and most rendered text.
DEFAULT_LONG_EDGE = 1568

#: The high-resolution tier's ceiling. Asking for more cannot buy detail, because
#: the API scales anything larger back down to this before the model sees it.
MAX_LONG_EDGE = 2576

#: Side of the square patch one visual token covers.
VISUAL_TOKEN_PATCH = 28

#: Ceiling on the bytes we will hand back per image, comfortably inside the
#: 10 MB base64 (~7.5 MB raw) per-image cap with room for several frames in one
#: reply. Exceeding it triggers the format ladder below rather than a rejection
#: at the client.
MAX_IMAGE_BYTES = 3 * 1024 * 1024

#: Tried in order until one fits `MAX_IMAGE_BYTES`. PNG is first because the
#: thing most worth checking about a generated image is whether its *text*
#: rendered correctly, and that is exactly what lossy compression destroys. WebP
#: precedes JPEG so transparency survives one step longer — a flattened
#: background would otherwise read as a deliberate white one.
_FORMAT_LADDER: tuple[str, ...] = ("png", "webp", "jpeg")

ImageFormat = Literal["png", "webp", "jpeg"]

#: ffmpeg demuxer per video extension, and the membership test for "may be
#: handed to ffmpeg at all". The mapping exists for the same reason
#: `AUDIO_DEMUXER_BY_EXTENSION` does: the demuxer must never be chosen by
#: letting ffmpeg probe the file's *contents*, because the playlist demuxers
#: (`concat`, `hls`, `dash`) read other local files named inside the data. We
#: pass `-f <demuxer>` explicitly, so content cannot select one.
VIDEO_DEMUXER_BY_EXTENSION: dict[str, str] = {
    "mp4": "mp4",
    "m4v": "mp4",
    "mov": "mov",
    "webm": "matroska",
    "mkv": "matroska",
}

DECODABLE_VIDEO_EXTENSIONS: frozenset[str] = frozenset(VIDEO_DEMUXER_BY_EXTENSION)

#: Upper bound on frames per call. Each one costs visual tokens, and the point
#: is to sample a clip, not to transcode it into the context window.
MAX_FRAMES = 8

DEFAULT_FRAMES = 3

#: Frames default smaller than stills: three of them at this long edge cost
#: about what one still does, and motion questions ("does the camera pan?",
#: "is the subject still there at the end?") do not need full resolution.
DEFAULT_FRAME_LONG_EDGE = 768

_FFMPEG_TIMEOUT = 60


def safe_video_demuxer(name_or_suffix: str) -> str:
    """Return the ffmpeg demuxer for an allowlisted video extension.

    Mirrors `audio.constants.safe_audio_format`, and exists for the same
    reason: the returned name goes to ffmpeg as ``-f <demuxer>``, which stops
    the file's own contents from selecting a playlist demuxer that would open
    unrelated local files. Accepts a filename or a bare/dotted suffix.

    Raises:
        ValueError: If the extension is outside `DECODABLE_VIDEO_EXTENSIONS`.
    """
    ext = name_or_suffix.rsplit(".", 1)[-1].lower() if "." in name_or_suffix else name_or_suffix.lower()
    if ext not in DECODABLE_VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video format {ext!r}; allowed: {', '.join(sorted(DECODABLE_VIDEO_EXTENSIONS))}")
    return VIDEO_DEMUXER_BY_EXTENSION[ext]


def visual_tokens(width: int, height: int) -> int:
    """What this image will cost the model to look at, in visual tokens."""
    return math.ceil(width / VISUAL_TOKEN_PATCH) * math.ceil(height / VISUAL_TOKEN_PATCH)


def _check_long_edge(long_edge: int) -> int:
    if not 1 <= long_edge <= MAX_LONG_EDGE:
        raise ValueError(
            f"max_dimension must be between 1 and {MAX_LONG_EDGE} (got {long_edge}); "
            f"the API downscales anything larger before the model sees it"
        )
    return long_edge


def _crop(img: PILImage.Image, region: list[int]) -> PILImage.Image:
    """Crop to ``[left, top, right, bottom]``, validated against the real image."""
    if len(region) != 4:
        raise ValueError(f"region must be [left, top, right, bottom] (got {len(region)} values)")
    left, top, right, bottom = region
    if right <= left or bottom <= top:
        raise ValueError(f"region must have right > left and bottom > top (got {region})")
    width, height = img.size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"region {region} falls outside the {width}x{height} image")
    return img.crop((left, top, right, bottom))


def _fit(img: PILImage.Image, long_edge: int) -> PILImage.Image:
    """Scale down so the long edge is at most `long_edge`. Never scales up."""
    if max(img.size) <= long_edge:
        return img
    scale = long_edge / max(img.size)
    # At least 1px per side: a sliver of a crop must still produce a valid image.
    target = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(target, PILImage.Resampling.LANCZOS)


def _encode(img: PILImage.Image, fmt: str) -> bytes:
    """Encode to `fmt`, converting the mode only as far as the format demands."""
    buffer = io.BytesIO()
    if fmt == "jpeg":
        # JPEG has no alpha. Composite onto white rather than dropping the
        # channel, so a transparent PNG does not come back with black fringes.
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            flattened = PILImage.new("RGB", rgba.size, (255, 255, 255))
            flattened.paste(rgba, mask=rgba.split()[-1])
            img = flattened
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=85, optimize=True)
    elif fmt == "webp":
        img.save(buffer, format="WEBP", quality=90, method=4)
    else:
        if img.mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _render(img: PILImage.Image, long_edge: int, preferred: str) -> tuple[bytes, str, tuple[int, int], bool]:
    """Fit, then encode into the most faithful format that stays under the cap.

    Walks `_FORMAT_LADDER` from the requested format onward; if even the last
    one is too big, halves the long edge and starts again. Returns the bytes,
    the format used, the delivered dimensions, and whether the byte ceiling
    forced a change the caller did not ask for.
    """
    start = _FORMAT_LADDER.index(preferred)
    edge = long_edge
    downgraded = False

    for _ in range(4):
        fitted = _fit(img, edge)
        for fmt in _FORMAT_LADDER[start:]:
            data = _encode(fitted, fmt)
            if len(data) <= MAX_IMAGE_BYTES:
                return data, fmt, fitted.size, downgraded or fmt != preferred or edge != long_edge
            downgraded = True
        edge = max(1, edge // 2)

    # Four halvings from <=2576 leaves a thumbnail; whatever the last ladder
    # produced is what the model gets, oversized or not, because refusing to
    # answer is worse than answering with a large picture.
    fitted = _fit(img, edge)
    data = _encode(fitted, _FORMAT_LADDER[-1])
    logger.warning("inspect: could not fit image under %d bytes; returning %d", MAX_IMAGE_BYTES, len(data))
    return data, _FORMAT_LADDER[-1], fitted.size, True


def _note(
    label: str,
    source_size: tuple[int, int],
    source_format: str | None,
    delivered_size: tuple[int, int],
    fmt: str,
    data: bytes,
    *,
    region: list[int] | None,
    forced: bool,
) -> str:
    """The text block that travels with the image.

    Without it the model is looking at pixels of unknown provenance: it cannot
    tell a downscaled crop from the artifact on disk, and would happily report
    "the image is 768px wide" about a file that is 4K.
    """
    parts = [f"{label}: {source_size[0]}x{source_size[1]}"]
    if source_format:
        parts.append(source_format.upper())
    if region is not None:
        parts.append(f"cropped to {region}")
    if delivered_size != source_size:
        parts.append(f"shown at {delivered_size[0]}x{delivered_size[1]}")
    else:
        parts.append("shown at full size")
    parts.append(f"{fmt.upper()}, {len(data) / 1024:.0f} KB")
    parts.append(f"~{visual_tokens(*delivered_size)} visual tokens")
    note = " · ".join(parts)
    if forced:
        note += " (re-encoded to stay within the per-image size limit)"
    return note


async def inspect_image(
    filename: str,
    max_dimension: int = DEFAULT_LONG_EDGE,
    region: list[int] | None = None,
    image_format: ImageFormat = "png",
) -> list[Image | str]:
    """Return a generated or reference image as visual content the model can read.

    Args:
        filename: Image filename (not a path) in the configured image directory.
        max_dimension: Long-edge ceiling for what is returned, 1..`MAX_LONG_EDGE`.
        region: Optional ``[left, top, right, bottom]`` crop in *source* pixels,
            applied before scaling — the way to read small rendered text.
        image_format: Preferred encoding. Downgraded along PNG -> WebP -> JPEG
            only if needed to stay under the per-image byte ceiling.

    Returns:
        The image followed by a one-line note describing what was done to it.

    Raises:
        ValueError: Unreadable file, bad region, or out-of-range max_dimension.
        RuntimeError: If the image path is not configured.
    """
    long_edge = _check_long_edge(max_dimension)
    storage = get_storage()

    async with storage.local_path("reference", filename) as path:

        def _work() -> tuple[bytes, str, tuple[int, int], bool, tuple[int, int], str | None]:
            try:
                with PILImage.open(path) as opened:
                    opened.load()
                    source_format = opened.format
                    source_size = opened.size
                    img = _crop(opened, region) if region is not None else opened
                    data, fmt, delivered, forced = _render(img, long_edge, image_format)
            except FileNotFoundError as exc:
                raise ValueError(f"Image not found: {filename}") from exc
            except PILImage.UnidentifiedImageError as exc:
                raise ValueError(f"Not a readable image: {filename}") from exc
            except OSError as exc:
                raise ValueError(f"Error reading image {filename}: {exc}") from exc
            return data, fmt, delivered, forced, source_size, source_format

        data, fmt, delivered, forced, source_size, source_format = await anyio.to_thread.run_sync(_work)

    logger.info(
        "inspect_image: %s %dx%d -> %dx%d %s (%d bytes)",
        filename,
        source_size[0],
        source_size[1],
        delivered[0],
        delivered[1],
        fmt,
        len(data),
    )
    note = _note(filename, source_size, source_format, delivered, fmt, data, region=region, forced=forced)
    return [Image(data=data, format=fmt), note]


def _require_ffmpeg() -> tuple[str, str]:
    """Locate ffmpeg and ffprobe, or explain how to get them.

    sanzaru only ever reaches ffmpeg through pydub today, so an install that
    never touched audio can be missing it entirely. A RuntimeError puts this in
    the same class as a missing API key, which the CLI already maps to a
    configuration exit code rather than a crash.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError(
            "inspect_video_frame needs ffmpeg and ffprobe on PATH "
            "(brew install ffmpeg, apt install ffmpeg, or pip install static-ffmpeg)"
        )
    return ffmpeg, ffprobe


async def _run(command: list[str], *, what: str) -> bytes:
    # Bounded for the same reason every Realtime turn is: nothing in a decoder
    # promises to terminate, and this runs inside a blocking tool call, so a
    # wedged ffmpeg would hold the request open indefinitely. Leaving the cancel
    # scope tears the process down.
    try:
        with anyio.fail_after(_FFMPEG_TIMEOUT):
            # input=b"" closes stdin, so a malformed file can never leave ffmpeg
            # waiting on a terminal that is not there. `-nostdin` covers the same
            # ground; both, because the failure is a hang rather than an error.
            result = await anyio.run_process(command, check=False, input=b"")
    except TimeoutError as exc:
        raise ValueError(f"{what} timed out after {_FFMPEG_TIMEOUT}s") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ValueError(f"{what} failed: {detail[-1] if detail else f'exit {result.returncode}'}")
    return result.stdout


async def _probe_duration(ffprobe: str, path: str, demuxer: str) -> float | None:
    """Clip duration in seconds, or None when the container does not report one."""
    raw = await _run(
        [
            ffprobe,
            "-v",
            "error",
            "-f",
            demuxer,
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ],
        what="ffprobe",
    )
    try:
        value = json.loads(raw)["format"]["duration"]
        duration = float(value)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return duration if duration > 0 else None


def _frame_timestamps(duration: float | None, frames: int, requested: list[float] | None) -> list[float]:
    """Explicit timestamps if given, else `frames` spread across the clip.

    Sampling avoids both ends: the first and last frames of a generated clip are
    the least informative (fades, and the frame a thumbnail already shows), so
    the points sit at the midpoints of `frames` equal slices.
    """
    if requested is not None:
        if not requested:
            raise ValueError("timestamps must contain at least one value")
        if len(requested) > MAX_FRAMES:
            raise ValueError(f"at most {MAX_FRAMES} timestamps per call (got {len(requested)})")
        if any(t < 0 for t in requested):
            raise ValueError("timestamps must not be negative")
        if duration is not None and any(t >= duration for t in requested):
            raise ValueError(f"timestamps must be under the clip duration ({duration:.2f}s)")
        return sorted(requested)

    if not 1 <= frames <= MAX_FRAMES:
        raise ValueError(f"frames must be between 1 and {MAX_FRAMES} (got {frames})")
    if duration is None:
        # No duration to divide: take the opening frame and say so upstream.
        return [0.0]
    return [duration * (index + 0.5) / frames for index in range(frames)]


async def inspect_video_frame(
    filename: str,
    frames: int = DEFAULT_FRAMES,
    timestamps: list[float] | None = None,
    max_dimension: int = DEFAULT_FRAME_LONG_EDGE,
) -> list[Image | str]:
    """Return still frames from a downloaded video as content the model can read.

    The only way to check what Sora actually rendered. Frames are decoded with
    an explicitly chosen demuxer, never one probed from the file's contents.

    Args:
        filename: Video filename (not a path) in the configured video directory.
        frames: How many evenly spaced frames to sample, 1..`MAX_FRAMES`. Ignored
            when `timestamps` is given.
        timestamps: Optional explicit offsets in seconds.
        max_dimension: Long-edge ceiling per frame, 1..`MAX_LONG_EDGE`.

    Returns:
        Each frame followed by its own note, in ascending time order.

    Raises:
        ValueError: Unsupported container, unreadable file, or bad arguments.
        RuntimeError: If ffmpeg/ffprobe are missing or the video path is unset.
    """
    long_edge = _check_long_edge(max_dimension)
    demuxer = safe_video_demuxer(filename)
    ffmpeg, ffprobe = _require_ffmpeg()
    storage = get_storage()

    blocks: list[Image | str] = []

    async with storage.local_path("video", filename) as path:
        location = str(path)
        duration = await _probe_duration(ffprobe, location, demuxer)
        offsets = _frame_timestamps(duration, frames, timestamps)

        for offset in offsets:
            raw = await _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-ss",
                    f"{offset:.3f}",
                    # Before -i, so the container's own content cannot pick the demuxer.
                    "-f",
                    demuxer,
                    "-i",
                    location,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                what="ffmpeg",
            )
            if not raw:
                raise ValueError(f"No frame at {offset:.2f}s in {filename} (past the end of the clip?)")

            def _work(encoded: bytes = raw) -> tuple[bytes, str, tuple[int, int], bool, tuple[int, int]]:
                with PILImage.open(io.BytesIO(encoded)) as frame:
                    frame.load()
                    source_size = frame.size
                    data, fmt, delivered, forced = _render(frame, long_edge, "png")
                return data, fmt, delivered, forced, source_size

            data, fmt, delivered, forced, source_size = await anyio.to_thread.run_sync(_work)
            label = f"{filename} @ {offset:.2f}s"
            blocks.append(Image(data=data, format=fmt))
            blocks.append(_note(label, source_size, "PNG frame", delivered, fmt, data, region=None, forced=forced))

    logger.info("inspect_video_frame: %s -> %d frame(s) at %s", filename, len(offsets), offsets)
    if duration is None:
        blocks.append(f"{filename}: container reported no duration, so only the opening frame was sampled.")
    return blocks
