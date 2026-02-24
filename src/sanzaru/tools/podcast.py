# SPDX-License-Identifier: MIT
"""Multi-voice podcast generation tool.

Generates a multi-voice podcast from a structured script by:
1. Validating the PodcastScript schema
2. Generating each segment via the OpenAI TTS API (sequentially)
3. Stitching segments with configurable silence gaps using pydub
4. Optionally peak-normalizing each segment for consistent loudness
5. Writing the final audio to the audio storage backend
"""

import time
from io import BytesIO
from typing import Any

import anyio
from pydantic import BaseModel
from pydub import AudioSegment  # type: ignore[import-untyped]
from pydub.effects import normalize as pydub_normalize  # type: ignore[import-untyped]

from ..audio.processor import AudioProcessor
from ..config import get_client, logger
from ..infrastructure import FileSystemRepository, split_text_for_tts

# ==================== RESULT MODEL ====================


class PodcastResult(BaseModel):
    """Result from generate_podcast."""

    output_file: str
    title: str
    segment_count: int
    estimated_duration_seconds: float
    speakers: list[str]
    transcript: str


# ==================== VALIDATION ====================


def _validate_script(script: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate PodcastScript structure and return its components.

    Args:
        script: Raw dict representing a PodcastScript.

    Returns:
        Tuple of (title, speakers, segments, config).

    Raises:
        ValueError: If the script is invalid.
    """
    for key in ("title", "speakers", "segments", "config"):
        if key not in script:
            raise ValueError(f"PodcastScript missing required field: '{key}'")

    title: str = script["title"]
    if not title or not title.strip():
        raise ValueError("PodcastScript 'title' must not be empty")

    speakers: list[dict[str, Any]] = script["speakers"]
    if not speakers:
        raise ValueError("PodcastScript must have at least 1 speaker")
    if len(speakers) > 4:
        raise ValueError("PodcastScript supports at most 4 speakers")

    speaker_ids: set[str] = set()
    for i, speaker in enumerate(speakers):
        for field in ("id", "name", "voice", "speed", "instructions"):
            if field not in speaker:
                raise ValueError(f"Speaker {i} missing required field: '{field}'")
        speaker_ids.add(speaker["id"])

    segments: list[dict[str, Any]] = script["segments"]
    if not segments:
        raise ValueError("PodcastScript must have at least 1 segment")

    for i, segment in enumerate(segments):
        if "speaker" not in segment:
            raise ValueError(f"Segment {i} missing required field: 'speaker'")
        if "text" not in segment:
            raise ValueError(f"Segment {i} missing required field: 'text'")
        if segment["speaker"] not in speaker_ids:
            raise ValueError(f"Segment {i} references unknown speaker id: '{segment['speaker']}'")
        if len(segment["text"]) > 40000:
            raise ValueError(f"Segment {i} text exceeds 40000 characters")

    config: dict[str, Any] = script["config"]
    for key in ("default_pause_ms", "section_pause_ms", "normalize_loudness", "output_format"):
        if key not in config:
            raise ValueError(f"PodcastConfig missing required field: '{key}'")

    if config["output_format"] not in ("mp3", "wav"):
        raise ValueError("PodcastConfig 'output_format' must be 'mp3' or 'wav'")

    return title, speakers, segments, config


# ==================== DURATION ESTIMATION ====================


def _estimate_duration(
    segments: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    config: dict[str, Any],
) -> float:
    """Estimate total podcast duration in seconds.

    Uses ~150 words/minute at 1.0x speed as the baseline.

    Args:
        segments: List of segment dicts.
        speakers: List of speaker dicts.
        config: PodcastConfig dict.

    Returns:
        Estimated duration in seconds.
    """
    speaker_speeds = {s["id"]: float(s.get("speed", 1.0)) for s in speakers}

    speech_seconds = 0.0
    total_pause_ms = 0

    for segment in segments:
        word_count = len(segment["text"].split())
        speed = float(segment.get("speed_override") or speaker_speeds.get(segment["speaker"], 1.0))
        speech_seconds += word_count * 60.0 / (150.0 * speed)
        total_pause_ms += int(segment.get("pause_after", config["default_pause_ms"]))

    intro_ms = int(config.get("intro_silence_ms") or 0)
    outro_ms = int(config.get("outro_silence_ms") or 0)

    return speech_seconds + (total_pause_ms + intro_ms + outro_ms) / 1000.0


# ==================== TTS GENERATION ====================


async def _generate_tts_bytes(
    text: str,
    voice: str,
    speed: float,
) -> bytes:
    """Generate TTS audio bytes for a single text block.

    Handles texts longer than the 4096-character API limit by splitting
    into chunks and concatenating. Chunk-level generation is parallelised
    for long segments.

    Args:
        text: Text to synthesise.
        voice: TTS voice identifier.
        speed: Speech speed multiplier (0.25–4.0).

    Returns:
        Raw MP3 bytes from the TTS API.
    """
    from aioresult import ResultCapture  # type: ignore[import-untyped]

    client = get_client()
    text_chunks = split_text_for_tts(text)

    if len(text_chunks) == 1:
        response = await client.audio.speech.create(
            input=text_chunks[0],
            model="gpt-4o-mini-tts",
            voice=voice,  # type: ignore[arg-type]
            speed=speed,
        )
        return response.content

    # Multiple chunks — parallel generation, then concatenate
    logger.debug(f"Segment split into {len(text_chunks)} TTS chunks")

    async def _gen_chunk(chunk_text: str) -> bytes:
        r = await client.audio.speech.create(
            input=chunk_text,
            model="gpt-4o-mini-tts",
            voice=voice,  # type: ignore[arg-type]
            speed=speed,
        )
        return r.content

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _gen_chunk, chunk) for chunk in text_chunks]

    audio_chunks = [c.result() for c in captures]
    processor = AudioProcessor()
    return await processor.concatenate_audio_segments(audio_chunks)


# ==================== STITCHING ====================


def _stitch_audio(
    segment_bytes_list: list[bytes],
    pause_ms_list: list[int],
    intro_ms: int,
    outro_ms: int,
    normalize_loudness: bool,
    output_format: str,
    output_bitrate: str,
) -> bytes:
    """Stitch audio segments with silence gaps using pydub.

    This is CPU-bound work that runs in a thread pool.

    Args:
        segment_bytes_list: Raw MP3 bytes for each segment.
        pause_ms_list: Silence duration in ms after each segment (same length as segment_bytes_list).
        intro_ms: Silence in ms before the first segment.
        outro_ms: Silence in ms after the last segment.
        normalize_loudness: Whether to peak-normalize each segment.
        output_format: Output format ("mp3" or "wav").
        output_bitrate: MP3 bitrate string (e.g., "192k"). Ignored for WAV.

    Returns:
        Final concatenated audio as bytes.
    """
    combined = AudioSegment.silent(duration=intro_ms) if intro_ms > 0 else AudioSegment.empty()

    for raw_bytes, pause_ms in zip(segment_bytes_list, pause_ms_list, strict=True):
        seg = AudioSegment.from_mp3(BytesIO(raw_bytes))
        if normalize_loudness:
            seg = pydub_normalize(seg)
        combined += seg
        if pause_ms > 0:
            combined += AudioSegment.silent(duration=pause_ms)

    if outro_ms > 0:
        combined += AudioSegment.silent(duration=outro_ms)

    output = BytesIO()
    if output_format == "mp3":
        combined.export(output, format="mp3", bitrate=output_bitrate)
    else:
        combined.export(output, format="wav")

    return output.getvalue()


# ==================== TRANSCRIPT BUILDER ====================


def _build_transcript(
    segments: list[dict[str, Any]],
    speaker_map: dict[str, dict[str, Any]],
) -> str:
    """Build a plain-text transcript from the podcast script.

    Args:
        segments: List of segment dicts.
        speaker_map: Mapping from speaker id to speaker dict.

    Returns:
        Formatted transcript string.
    """
    lines = []
    for segment in segments:
        speaker = speaker_map[segment["speaker"]]
        lines.append(f"**{speaker['name']}:** {segment['text']}")
    return "\n\n".join(lines)


# ==================== SAFE FILENAME ====================


def _safe_title(title: str) -> str:
    """Convert a podcast title to a filesystem-safe slug."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_") or "podcast"


# ==================== MAIN TOOL ====================


async def generate_podcast(script: dict[str, Any]) -> PodcastResult:
    """Generate a multi-voice podcast from a structured PodcastScript.

    Args:
        script: A PodcastScript object (see schema in tool description).

    Returns:
        PodcastResult with output filename, duration estimate, and transcript.

    Raises:
        ValueError: If the script fails validation.
    """
    # 1. Validate
    title, speakers, segments, config = _validate_script(script)
    speaker_map = {s["id"]: s for s in speakers}

    # 2. Estimate duration (informational)
    estimated_duration = _estimate_duration(segments, speakers, config)
    logger.info(
        f"Podcast '{title}': {len(segments)} segments, {len(speakers)} speakers, ~{estimated_duration:.0f}s estimated"
    )

    # 3. Generate segments sequentially
    segment_bytes_list: list[bytes] = []
    for i, segment in enumerate(segments):
        speaker = speaker_map[segment["speaker"]]
        voice = str(speaker["voice"])
        speed = float(segment.get("speed_override") or speaker.get("speed", 1.0))

        logger.info(f"Generating segment {i + 1}/{len(segments)} [{speaker['name']} / {voice}]")
        audio_bytes = await _generate_tts_bytes(text=segment["text"], voice=voice, speed=speed)
        segment_bytes_list.append(audio_bytes)

    # 4. Build silence schedule (no trailing pause on last segment — outro covers it)
    default_pause_ms = int(config.get("default_pause_ms", 600))
    pause_ms_list: list[int] = []
    for i, segment in enumerate(segments):
        if i == len(segments) - 1:
            pause_ms_list.append(0)
        else:
            pause_ms_list.append(int(segment.get("pause_after", default_pause_ms)))

    intro_ms = int(config.get("intro_silence_ms") or 0)
    outro_ms = int(config.get("outro_silence_ms") or 0)
    normalize_loudness = bool(config.get("normalize_loudness", True))
    output_format = str(config.get("output_format", "mp3"))
    output_bitrate = str(config.get("output_bitrate", "192k"))

    # 5. Stitch (CPU-bound — run in thread pool)
    logger.info("Stitching podcast audio...")
    final_audio = await anyio.to_thread.run_sync(
        lambda: _stitch_audio(
            segment_bytes_list=segment_bytes_list,
            pause_ms_list=pause_ms_list,
            intro_ms=intro_ms,
            outro_ms=outro_ms,
            normalize_loudness=normalize_loudness,
            output_format=output_format,
            output_bitrate=output_bitrate,
        )
    )

    # 6. Write output to audio storage
    timestamp = int(time.time())
    output_filename = f"{_safe_title(title)}_{timestamp}.{output_format}"
    file_repo = FileSystemRepository()
    await file_repo.write_audio_file(output_filename, final_audio)
    logger.info(f"Podcast written: {output_filename} ({len(final_audio):,} bytes)")

    # 7. Build transcript
    transcript = _build_transcript(segments, speaker_map)

    return PodcastResult(
        output_file=output_filename,
        title=title,
        segment_count=len(segments),
        estimated_duration_seconds=round(estimated_duration, 1),
        speakers=[s["name"] for s in speakers],
        transcript=transcript,
    )
