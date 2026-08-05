# SPDX-License-Identifier: MIT
"""Multi-voice podcast generation tool.

Generates a multi-voice podcast from a structured script by:
1. Validating the PodcastScript schema
2. Generating every segment in parallel via the configured TTS provider(s),
   bounded by a per-provider concurrency limiter
3. Stitching segments with configurable silence gaps using pydub
4. Optionally peak-normalizing each segment for consistent loudness
5. Writing the final audio to the audio storage backend

Speakers choose their provider independently, so a single episode can mix
OpenAI and ElevenLabs voices — the stitch path is mp3-in, mp3-out.
"""

import time
from io import BytesIO
from typing import Literal, NotRequired, TypedDict

import anyio
from aioresult import ResultCapture  # type: ignore[import-untyped]
from openai.types.audio.speech_model import SpeechModel
from pydantic import BaseModel
from pydub import AudioSegment  # type: ignore[import-untyped]
from pydub.effects import normalize as pydub_normalize  # type: ignore[import-untyped]

from ..audio.constants import (
    ELEVENLABS_SPEED_RANGE,
    OPENAI_SPEED_RANGE,
    PODCAST_TARGET_FRAME_RATE,
    ElevenLabsModel,
    TTSProviderName,
)
from ..audio.providers import (
    VOICE_SETTINGS_BOOL_KEYS,
    VOICE_SETTINGS_FLOAT_KEYS,
    VOICE_SETTINGS_KEYS,
    SpeechRequest,
    TTSProvider,
    VoiceSettingsDict,
    get_provider,
    synthesize_speech,
    validate_provider_name,
)
from ..config import logger
from ..infrastructure import FileSystemRepository


class Speaker(TypedDict):
    id: str
    name: str
    voice: str
    speed: float
    instructions: str
    role: NotRequired[str]
    provider: NotRequired[TTSProviderName]
    model: NotRequired[str]
    voice_settings: NotRequired[VoiceSettingsDict]


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
    provider: NotRequired[TTSProviderName]
    max_concurrency: NotRequired[int]


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


def _resolve_provider_name(
    speaker: Speaker,
    config: PodcastConfig,
    default: TTSProviderName,
) -> TTSProviderName:
    """Provider for a speaker: speaker override > episode default > tool default."""
    return speaker.get("provider") or config.get("provider") or default


def _resolve_model(speaker: Speaker, provider: TTSProvider, openai_model: str) -> str:
    """Model for a speaker.

    A per-speaker `model` always wins. Otherwise the tool's `model` argument
    applies to OpenAI speakers (it names an OpenAI model), and ElevenLabs
    speakers fall back to the provider default.
    """
    if "model" in speaker:
        return provider.resolve_model(speaker["model"])
    return provider.resolve_model(openai_model if provider.name == "openai" else None)


def _speed_range(provider_name: TTSProviderName) -> tuple[float, float]:
    return ELEVENLABS_SPEED_RANGE if provider_name == "elevenlabs" else OPENAI_SPEED_RANGE


def _validate_voice_settings(settings: VoiceSettingsDict, context: str) -> None:
    """Validate a per-speaker voice_settings mapping."""
    if not isinstance(settings, dict):
        raise ValueError(f"{context} voice_settings must be an object")
    for key in settings:
        if key not in VOICE_SETTINGS_KEYS:
            raise ValueError(
                f"{context} voice_settings has unknown key '{key}'; expected: {', '.join(VOICE_SETTINGS_KEYS)}"
            )
    for key in VOICE_SETTINGS_FLOAT_KEYS:
        if key in settings and not isinstance(settings[key], int | float):  # type: ignore[literal-required]
            raise ValueError(f"{context} voice_settings['{key}'] must be a number")
    for key in VOICE_SETTINGS_BOOL_KEYS:
        if key in settings and not isinstance(settings[key], bool):  # type: ignore[literal-required]
            raise ValueError(f"{context} voice_settings['{key}'] must be a boolean")


def _validate_script(
    script: PodcastScript,
    default_provider: TTSProviderName = "openai",
) -> tuple[str, list[Speaker], list[Segment], PodcastConfig]:
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

    config = script["config"]
    for key in ("default_pause_ms", "normalize_loudness", "output_format"):
        if key not in config:
            raise ValueError(f"PodcastConfig missing required field: '{key}'")

    if config["output_format"] not in ("mp3", "wav"):
        raise ValueError("PodcastConfig 'output_format' must be 'mp3' or 'wav'")

    if "provider" in config:
        validate_provider_name(config["provider"], "PodcastConfig 'provider'")
    if "max_concurrency" in config and (
        not isinstance(config["max_concurrency"], int) or config["max_concurrency"] < 1
    ):
        raise ValueError(f"PodcastConfig 'max_concurrency' must be a positive integer, got {config['max_concurrency']}")

    speaker_ids: set[str] = set()
    # provider name per speaker id, so segment-level checks below know which
    # speed range applies.
    speaker_providers: dict[str, TTSProviderName] = {}

    for i, speaker in enumerate(speakers):
        for field in ("id", "name", "voice", "speed", "instructions"):
            if field not in speaker:
                raise ValueError(f"Speaker {i} missing required field: '{field}'")

        if "provider" in speaker:
            validate_provider_name(speaker["provider"], f"Speaker {i} 'provider'")
        provider_name = _resolve_provider_name(speaker, config, default_provider)

        low, high = _speed_range(provider_name)
        if not low <= speaker["speed"] <= high:
            raise ValueError(
                f"Speaker {i} speed must be between {low} and {high} for provider='{provider_name}', "
                f"got {speaker['speed']}"
            )

        if "voice_settings" in speaker:
            _validate_voice_settings(speaker["voice_settings"], f"Speaker {i}")

        # Fail before spending a single API call: this resolves the model,
        # rejects cross-provider model names, and applies provider-specific
        # rules (eleven_v3 has no speed, ElevenLabs needs a voice id).
        provider = get_provider(provider_name)
        model = _resolve_model(speaker, provider, "gpt-4o-mini-tts")
        provider.validate(
            SpeechRequest(
                text="validation probe",
                voice=provider.resolve_voice(speaker["voice"]),
                model=model,
                speed=speaker["speed"],
                instructions=speaker["instructions"],
                voice_settings=speaker.get("voice_settings"),
            )
        )

        speaker_ids.add(speaker["id"])
        speaker_providers[speaker["id"]] = provider_name

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
        if "speed_override" in segment:
            # Checked against the owning speaker's provider range, which is why
            # this runs after the speaker pass.
            low, high = _speed_range(speaker_providers[segment["speaker"]])
            if not low <= segment["speed_override"] <= high:
                raise ValueError(
                    f"Segment {i} speed_override must be between {low} and {high} for "
                    f"provider='{speaker_providers[segment['speaker']]}', got {segment['speed_override']}"
                )

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
        # Every provider is asked for mp3 — keep in sync with the TTS providers.
        seg = AudioSegment.from_mp3(BytesIO(raw_bytes))
        # Providers differ in native rate (OpenAI 24kHz, ElevenLabs 44.1kHz).
        # pydub would resample on concatenation anyway, but pinning it here makes
        # a mixed-provider episode deterministic regardless of segment order.
        seg = seg.set_frame_rate(PODCAST_TARGET_FRAME_RATE).set_channels(1)
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


async def generate_podcast(
    script: PodcastScript,
    model: SpeechModel | ElevenLabsModel = "gpt-4o-mini-tts",
    provider: TTSProviderName = "openai",
) -> PodcastResult:
    """Generate a multi-voice podcast from a structured PodcastScript.

    Args:
        script: The podcast script. Speakers may each pick their own provider,
            so one episode can mix OpenAI and ElevenLabs voices.
        model: Default model for OpenAI speakers. ElevenLabs speakers fall back to
            that provider's default unless they set their own `model`.
        provider: Episode-wide default provider, itself overridden by
            `config.provider` and then by each speaker's `provider`.

    Raises ValueError if the script fails validation.
    """
    title, speakers, segments, config = _validate_script(script, default_provider=provider)
    speaker_map: dict[str, Speaker] = {s["id"]: s for s in speakers}

    estimated_duration = _estimate_duration(segments, speakers, config)
    logger.info(
        f"Podcast '{title}': {len(segments)} segments, {len(speakers)} speakers, ~{estimated_duration:.0f}s estimated"
    )

    # Resolve each speaker's provider and model once, up front.
    providers: dict[str, TTSProvider] = {}
    models: dict[str, str] = {}
    for speaker in speakers:
        speaker_provider = get_provider(_resolve_provider_name(speaker, config, provider))
        providers[speaker["id"]] = speaker_provider
        models[speaker["id"]] = _resolve_model(speaker, speaker_provider, model)

    # One limiter per provider, built here because CapacityLimiter binds to the
    # running event loop. The same limiter is passed down into synthesize_speech
    # so segment-level and chunk-level parallelism share one budget — which is
    # what ElevenLabs' concurrency cap actually counts. OpenAI's limit is 0
    # (unbounded) by default, preserving the historical behavior exactly.
    limiters: dict[str, anyio.CapacityLimiter | None] = {}
    for speaker_id, speaker_provider in providers.items():
        if speaker_provider.name in limiters:
            continue
        limit = config.get("max_concurrency") or speaker_provider.max_concurrency(models[speaker_id])
        limiters[speaker_provider.name] = anyio.CapacityLimiter(limit) if limit else None
        if limit:
            logger.info("Limiting %s to %d concurrent TTS requests", speaker_provider.name, limit)

    async def _gen_segment(i: int, segment: Segment) -> bytes:
        """Render one segment. Independent of the others, so #35's verify pass
        can re-invoke this for just the segments that failed QC."""
        speaker = speaker_map[segment["speaker"]]
        speaker_provider = providers[speaker["id"]]
        speed = segment["speed_override"] if "speed_override" in segment else speaker["speed"]
        # `in`-check rather than `or`: an intentional empty-string override must
        # not silently fall back to the speaker's instructions.
        instructions = segment["instruction_override"] if "instruction_override" in segment else speaker["instructions"]
        # "Queued", not "Generating": the limiter is acquired downstream in
        # synthesize_speech, so this line fires before the request goes out.
        logger.info(
            f"Queued segment {i + 1}/{len(segments)} [{speaker['name']} / {speaker['voice']} / {speaker_provider.name}]"
        )
        request = SpeechRequest(
            text=segment["text"],
            voice=speaker_provider.resolve_voice(speaker["voice"]),
            model=models[speaker["id"]],
            speed=speed,
            # instructions is OpenAI-only; _validate_script already warned once
            # per ElevenLabs speaker that carries one.
            instructions=instructions if speaker_provider.name == "openai" else None,
            voice_settings=speaker.get("voice_settings"),
        )
        return await synthesize_speech(speaker_provider, request, limiter=limiters[speaker_provider.name])

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _gen_segment, i, seg) for i, seg in enumerate(segments)]

    # Read by index, not completion order — the limiter only delays task entry.
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
