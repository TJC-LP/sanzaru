# SPDX-License-Identifier: MIT
"""Turning recorded PCM into deliverable audio.

Realtime hands back raw PCM16/24kHz, which maps straight onto a pydub
`AudioSegment` constructor — no ffmpeg round-trip between the models' mouths and
the finished episode. That is why the stitch path takes a `decode` callable
rather than assuming mp3.

Everything here is CPU-bound and expects to be called via
`anyio.to_thread.run_sync`. It imports pydub at module scope, so it must only be
imported from code already gated on the audio extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from pydub import AudioSegment  # type: ignore[import-untyped]

from .types import REALTIME_CHANNELS, REALTIME_SAMPLE_RATE, REALTIME_SAMPLE_WIDTH

# A stretch of the timeline: (speaker_id or None for a gap, PCM16 bytes).
TimelineItem = tuple[str | None, bytes]


def pcm_to_segment(pcm: bytes) -> AudioSegment:
    """Wrap raw realtime PCM as an AudioSegment without re-encoding."""
    return AudioSegment(
        data=pcm,
        sample_width=REALTIME_SAMPLE_WIDTH,
        frame_rate=REALTIME_SAMPLE_RATE,
        channels=REALTIME_CHANNELS,
    )


def decode_to_pcm(data: bytes, fmt: str = "mp3") -> bytes:
    """Decode a checkpointed act back to the raw PCM the mixer works in.

    Only the resume path needs this, and it costs one mp3 generation: an act
    read back from a checkpoint has been encoded once more than one recorded in
    the same run.
    """
    segment = AudioSegment.from_file(BytesIO(data), format=fmt)
    segment = (
        segment.set_frame_rate(REALTIME_SAMPLE_RATE)
        .set_channels(REALTIME_CHANNELS)
        .set_sample_width(REALTIME_SAMPLE_WIDTH)
    )
    return bytes(segment.raw_data)


def encode_pcm(pcm: bytes, output_format: str = "mp3", bitrate: str = "192k") -> bytes:
    """Encode raw PCM to a deliverable file."""
    segment = pcm_to_segment(pcm)
    buffer = BytesIO()
    if output_format == "mp3":
        segment.export(buffer, format="mp3", bitrate=bitrate)
    else:
        segment.export(buffer, format=output_format)
    return buffer.getvalue()


def render_stem(timeline: Sequence[TimelineItem], speaker_id: str, output_format: str, bitrate: str) -> bytes:
    """One speaker's isolated, time-aligned track.

    Everything the speaker did not say becomes silence of exactly the same
    length, so every stem lines up with the master sample-for-sample and can be
    dropped straight onto an editor timeline.
    """
    parts = [pcm if owner == speaker_id else b"\x00" * len(pcm) for owner, pcm in timeline]
    return encode_pcm(b"".join(parts), output_format, bitrate)


def slice_pcm_by_durations(pcm: bytes, seconds: Sequence[float]) -> list[bytes]:
    """Split one act's audio back into per-turn buffers.

    Used only when resuming: the checkpoint manifest knows each turn's duration
    but the decoded mp3 is a few frames longer or shorter than the original, so
    boundaries are scaled proportionally rather than computed from the recorded
    durations directly. Sub-frame drift lands inside the gap between turns,
    where the other speaker is silent anyway.
    """
    total = sum(seconds)
    if total <= 0 or not seconds:
        return [pcm]
    frame = REALTIME_SAMPLE_WIDTH * REALTIME_CHANNELS
    frames = len(pcm) // frame
    parts: list[bytes] = []
    cumulative = 0.0
    start_frame = 0
    for index, duration in enumerate(seconds):
        cumulative += duration
        end_frame = frames if index == len(seconds) - 1 else round(frames * cumulative / total)
        parts.append(pcm[start_frame * frame : end_frame * frame])
        start_frame = end_frame
    return parts
