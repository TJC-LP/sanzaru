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
from typing import Literal, NotRequired, TypedDict

import anyio
from aioresult import ResultCapture  # type: ignore[import-untyped]
from openai._types import Omit, omit
from openai.types.audio.speech_model import SpeechModel
from pydantic import BaseModel
from pydub import AudioSegment  # type: ignore[import-untyped]
from pydub.effects import normalize as pydub_normalize  # type: ignore[import-untyped]

from ..audio.processor import AudioProcessor
from ..config import get_client, logger
from ..infrastructure import FileSystemRepository, split_text_for_tts


class Speaker(TypedDict):
    id: str
    name: str
    voice: str
    speed: float
    instructions: str
    role: NotRequired[str]


class Segment(TypedDict):
    speaker: str
    text: str
    pause_after: NotRequired[int]
    speed_override: NotRequired[float]
    instruction_override: NotRequired[str]


class PodcastConfig(TypedDict):
    default_pause_ms: int
    intro_silence_ms: NotRequired[int]
    outro_silence_ms: NotRequired[int]
    normalize_loudness: bool
    output_format: Literal["mp3", "wav"]
    output_bitrate: NotRequired[str]


class PodcastScript(TypedDict):
    title: str
    description: NotRequired[str]
    speakers: list[Speaker]
    segments: list[Segment]
    config: PodcastConfig


class PodcastResult(BaseModel):
    """Result from generate_podcast."""

    output_file: str
    title: str
    segment_count: int
    estimated_duration_seconds: float
    speakers: list[str]
    transcript: str


def _validate_script(script: PodcastScript) -> tuple[str, list[Speaker], list[Segment], PodcastConfig]:
    """Validate PodcastScript structure and return its components.

    Raises ValueError if the script is invalid.
    """
    for key in ("title", "speakers", "segments", "config"):
        if key not in script:
            raise ValueError(f"PodcastScript missing required field: '{key}'")

    title = script["title"]
    if not title or not title.strip():
        raise ValueError("PodcastScript 'title' must not be empty")

    speakers = script["speakers"]
    if not speakers:
        raise ValueError("PodcastScript must have at least 1 speaker")
    if len(speakers) > 4:
        raise ValueError("PodcastScript supports at most 4 speakers")

    speaker_ids: set[str] = set()
    for i, speaker in enumerate(speakers):
        for field in ("id", "name", "voice", "speed", "instructions"):
            if field not in speaker:
                raise ValueError(f"Speaker {i} missing required field: '{field}'")
        if not (0.25 <= speaker["speed"] <= 4.0):
            raise ValueError(f"Speaker {i} speed must be between 0.25 and 4.0, got {speaker['speed']}")
        speaker_ids.add(speaker["id"])

    segments = script["segments"]
    if not segments:
        raise ValueError("PodcastScript must have at least 1 segment")

    for i, segment in enumerate(segments):
        if "speaker" not in segment:
            raise ValueError(f"Segment {i} missing required field: 'speaker'")
        if "text" not in segment:
            raise ValueError(f"Segment {i} missing required field: 'text'")
        if segment["speaker"] not in speaker_ids:
            raise ValueError(f"Segment {i} references unknown speaker id: '{segment['speaker']}'")
        if not segment["text"].strip():
            raise ValueError(f"Segment {i} text must not be empty")
        if len(segment["text"]) > 40000:
            raise ValueError(f"Segment {i} text exceeds 40000 characters")
        if "speed_override" in segment and not (0.25 <= segment["speed_override"] <= 4.0):
            raise ValueError(
                f"Segment {i} speed_override must be between 0.25 and 4.0, got {segment['speed_override']}"
            )

    config = script["config"]
    for key in ("default_pause_ms", "normalize_loudness", "output_format"):
        if key not in config:
            raise ValueError(f"PodcastConfig missing required field: '{key}'")

    if config["output_format"] not in ("mp3", "wav"):
        raise ValueError("PodcastConfig 'output_format' must be 'mp3' or 'wav'")

    return title, speakers, segments, config


def _estimate_duration(segments: list[Segment], speakers: list[Speaker], config: PodcastConfig) -> float:
    """Estimate total podcast duration in seconds (~150 wpm)."""
    speaker_speeds = {s["id"]: float(s["speed"]) for s in speakers}

    speech_seconds = 0.0
    total_pause_ms = 0

    for segment in segments:
        word_count = len(segment["text"].split())
        speed = float(segment["speed_override"]) if "speed_override" in segment else speaker_speeds[segment["speaker"]]
        speech_seconds += word_count * 60.0 / (150.0 * speed)
        total_pause_ms += int(segment.get("pause_after", config["default_pause_ms"]))

    intro_ms = int(config.get("intro_silence_ms") or 0)
    outro_ms = int(config.get("outro_silence_ms") or 0)

    return speech_seconds + (total_pause_ms + intro_ms + outro_ms) / 1000.0


async def _generate_tts_bytes(
    text: str,
    voice: str,
    speed: float,
    model: SpeechModel = "gpt-4o-mini-tts",
    instructions: str | None = None,
) -> bytes:
    """Generate TTS audio bytes for a single text block.

    Handles texts longer than the 4096-character API limit by splitting
    into chunks and concatenating. Chunk-level generation is parallelised
    for long segments.
    """
    client = get_client()
    text_chunks = split_text_for_tts(text)
    instr_param: str | Omit = omit if instructions is None else instructions

    if len(text_chunks) == 1:
        response = await client.audio.speech.create(
            input=text_chunks[0],
            model=model,
            voice=voice,  # type: ignore[arg-type]
            speed=speed,
            instructions=instr_param,
            response_format="mp3",
        )
        return response.content

    # Multiple chunks — parallel generation, then concatenate
    logger.debug(f"Segment split into {len(text_chunks)} TTS chunks")

    async def _gen_chunk(chunk_text: str) -> bytes:
        r = await client.audio.speech.create(
            input=chunk_text,
            model=model,
            voice=voice,  # type: ignore[arg-type]
            speed=speed,
            instructions=instr_param,
            response_format="mp3",
        )
        return r.content

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _gen_chunk, chunk) for chunk in text_chunks]

    audio_chunks = [c.result() for c in captures]
    processor = AudioProcessor()
    return await processor.concatenate_audio_segments(audio_chunks)


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
        # TTS calls explicitly request response_format="mp3" — keep in sync
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


def _safe_title(title: str) -> str:
    """Convert a podcast title to a filesystem-safe slug."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_") or "podcast"


async def generate_podcast(script: PodcastScript, model: SpeechModel = "gpt-4o-mini-tts") -> PodcastResult:
    """Generate a multi-voice podcast from a structured PodcastScript.

    Raises ValueError if the script fails validation.
    """
    title, speakers, segments, config = _validate_script(script)
    speaker_map: dict[str, Speaker] = {s["id"]: s for s in speakers}

    estimated_duration = _estimate_duration(segments, speakers, config)
    logger.info(
        f"Podcast '{title}': {len(segments)} segments, {len(speakers)} speakers, ~{estimated_duration:.0f}s estimated"
    )

    # Generate all TTS segments in parallel
    async def _gen_segment(i: int, segment: Segment) -> bytes:
        speaker = speaker_map[segment["speaker"]]
        voice = speaker["voice"]
        speed = segment["speed_override"] if "speed_override" in segment else speaker["speed"]
        instructions = segment.get("instruction_override") or speaker["instructions"]
        logger.info(f"Generating segment {i + 1}/{len(segments)} [{speaker['name']} / {voice}]")
        return await _generate_tts_bytes(
            text=segment["text"], voice=voice, speed=speed, model=model, instructions=instructions
        )

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _gen_segment, i, seg) for i, seg in enumerate(segments)]

    segment_bytes_list = [c.result() for c in captures]

    default_pause_ms = config.get("default_pause_ms", 600)
    pause_ms_list: list[int] = []
    for i, segment in enumerate(segments):
        if i == len(segments) - 1:
            pause_ms_list.append(0)
        else:
            pause_ms_list.append(segment.get("pause_after", default_pause_ms))

    intro_ms = config.get("intro_silence_ms") or 0
    outro_ms = config.get("outro_silence_ms") or 0
    normalize_loudness = config.get("normalize_loudness", True)
    output_format = config.get("output_format", "mp3")
    output_bitrate = config.get("output_bitrate", "192k")

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

    timestamp = int(time.time())
    output_filename = f"{_safe_title(title)}_{timestamp}.{output_format}"
    file_repo = FileSystemRepository()
    await file_repo.write_audio_file(output_filename, final_audio)
    logger.info(f"Podcast written: {output_filename} ({len(final_audio):,} bytes)")

    transcript = "\n\n".join(f"**{speaker_map[s['speaker']]['name']}:** {s['text']}" for s in segments)

    return PodcastResult(
        output_file=output_filename,
        title=title,
        segment_count=len(segments),
        estimated_duration_seconds=round(estimated_duration, 1),
        speakers=[s["name"] for s in speakers],
        transcript=transcript,
    )
