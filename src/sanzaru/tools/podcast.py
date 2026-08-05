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
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Literal, NotRequired, TypedDict

import anyio
from aioresult import ResultCapture  # type: ignore[import-untyped]
from openai.types.audio.speech_model import SpeechModel
from pydantic import BaseModel
from pydub import AudioSegment  # type: ignore[import-untyped]
from pydub.effects import normalize as pydub_normalize  # type: ignore[import-untyped]

from ..audio.constants import (
    DEFAULT_RENDER_MODE,
    ELEVENLABS_MODELS,
    ELEVENLABS_SPEED_RANGE,
    MIN_DIALOGUE_TURNS,
    OPENAI_SPEED_RANGE,
    PODCAST_TARGET_FRAME_RATE,
    RENDER_MODES,
    ElevenLabsModel,
    PodcastRenderMode,
    TTSProviderName,
)
from ..audio.providers import (
    DialogueTurn,
    SpeechRequest,
    TTSProvider,
    VoiceSettingsDict,
    as_dialogue_provider,
    check_voice_settings_types,
    get_provider,
    synthesize_speech,
    validate_provider_name,
)
from ..config import logger
from ..infrastructure import FileSystemRepository

_ELEVENLABS_MODEL_NAMES = frozenset(ELEVENLABS_MODELS)


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
    render_mode: NotRequired[PodcastRenderMode]
    dialogue_stability: NotRequired[float]


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


def _resolve_model(speaker: Speaker, provider: TTSProvider, requested_model: str | None) -> str:
    """Model for a speaker.

    A per-speaker `model` always wins, and a wrong one there is an error the
    caller asked for. Otherwise the tool's `model` argument applies to whichever
    provider it names; on a mixed-provider episode the speakers of the *other*
    provider fall back to their own default rather than failing the render.
    """
    if "model" in speaker:
        return provider.resolve_model(speaker["model"])
    try:
        resolved = provider.resolve_model(requested_model)
    except ValueError:
        # ElevenLabs rejects a foreign model name outright.
        return provider.resolve_model(None)
    # OpenAI's resolve_model accepts unknown names on purpose (new speech models
    # ship between our releases), so only the allowlist catches the reverse case.
    if provider.name == "openai" and resolved in _ELEVENLABS_MODEL_NAMES:
        return provider.resolve_model(None)
    return resolved


def _speed_range(provider_name: TTSProviderName) -> tuple[float, float]:
    return ELEVENLABS_SPEED_RANGE if provider_name == "elevenlabs" else OPENAI_SPEED_RANGE


def _resolve_segment_speech(segment: Segment, speaker: Speaker) -> tuple[float, VoiceSettingsDict | None]:
    """Speed and voice_settings for one segment, with script-level precedence applied.

    A segment's `speed_override` is the most specific value in the script, so it
    beats both `speaker["speed"]` and the speaker's `voice_settings["speed"]`.
    The second half matters because ElevenLabs' native speed knob lives *inside*
    voice_settings and wins there over the neutral `SpeechRequest.speed`: without
    materializing the override into the copy below, a speaker that sets
    `voice_settings["speed"]` would silently kill every override in its segments.

    Validation and rendering both go through here, so they can never disagree
    about which speed the episode will actually be rendered at.
    """
    settings = speaker.get("voice_settings")
    if "speed_override" not in segment:
        return speaker["speed"], settings
    speed = segment["speed_override"]
    if settings is not None and "speed" in settings:
        settings = settings.copy()
        settings["speed"] = speed
    return speed, settings


def _validate_script(
    script: PodcastScript,
    default_provider: TTSProviderName = "openai",
    default_model: str | None = None,
) -> tuple[str, list[Speaker], list[Segment], PodcastConfig]:
    """Validate PodcastScript structure and return its components.

    `default_model` must be the same value the render path will use, or
    validation checks a model the episode never runs on.

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

    render_mode: PodcastRenderMode = config.get("render_mode", DEFAULT_RENDER_MODE)
    if render_mode not in RENDER_MODES:
        raise ValueError(f"PodcastConfig 'render_mode' must be one of: {', '.join(RENDER_MODES)}, got {render_mode!r}")
    if "dialogue_stability" in config:
        stability = config["dialogue_stability"]
        if not isinstance(stability, int | float) or not 0.0 <= stability <= 1.0:
            raise ValueError(f"PodcastConfig 'dialogue_stability' must be between 0.0 and 1.0, got {stability}")
        if render_mode != "dialogue":
            logger.warning("PodcastConfig 'dialogue_stability' is ignored unless render_mode is 'dialogue'")

    speaker_ids: set[str] = set()
    # Per speaker id, so the segment pass below can re-probe with the same
    # provider, model, and voice the render will use.
    speaker_providers: dict[str, TTSProviderName] = {}
    speaker_probes: dict[str, tuple[Speaker, TTSProvider, SpeechRequest]] = {}

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
            check_voice_settings_types(speaker["voice_settings"], f"Speaker {i} ")
            if render_mode == "dialogue" and provider_name == "elevenlabs":
                # The dialogue endpoint takes one `stability` for the whole
                # request, not per-voice settings. Warn rather than reject:
                # a speaker may still fall back to segment rendering if it does
                # not end up inside a dialogue run.
                logger.warning(
                    "Speaker %d ('%s') sets voice_settings, which the dialogue endpoint cannot apply "
                    "per speaker - use config.dialogue_stability, or render_mode='segments' to keep them",
                    i,
                    speaker["id"],
                )

        # Fail before spending a single API call: this resolves the model,
        # rejects cross-provider model names, and applies provider-specific
        # rules (eleven_v3 has no speed, ElevenLabs needs a voice id).
        provider = get_provider(provider_name)
        model = _resolve_model(speaker, provider, default_model)
        probe = SpeechRequest(
            text="validation probe",
            voice=provider.resolve_voice(speaker["voice"]),
            model=model,
            speed=speaker["speed"],
            instructions=speaker["instructions"],
            voice_settings=speaker.get("voice_settings"),
        )
        provider.validate(probe)

        speaker_ids.add(speaker["id"])
        speaker_providers[speaker["id"]] = provider_name
        speaker_probes[speaker["id"]] = (speaker, provider, probe)

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
            # The range above is provider-wide; per-model rules (eleven_v3 has no
            # speed at all) live in provider.validate, which the speaker pass only
            # ever ran against speaker["speed"]. Re-probe each override or an
            # in-range-but-unsupported value fails inside the task group, after
            # sibling segments have already spent API calls.
            owner, owner_provider, probe = speaker_probes[segment["speaker"]]
            speed, settings = _resolve_segment_speech(segment, owner)
            try:
                # instructions=None: the speaker pass already warned once about
                # ElevenLabs ignoring them, and this must not repeat it per segment.
                owner_provider.validate(replace(probe, speed=speed, instructions=None, voice_settings=settings))
            except ValueError as exc:
                raise ValueError(f"Segment {i} speed_override: {exc}") from exc

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


def _derive_concurrency_limits(providers: dict[str, TTSProvider], models: dict[str, str]) -> dict[str, int]:
    """Tightest concurrency cap per provider, across every model the episode uses.

    The cap is per-model, but the limiter is per-provider, so the smallest model
    cap has to win: an eleven_flash_v2_5 host (4) beside an eleven_v3 guest (2)
    must still run 2-wide or ElevenLabs answers HTTP 429. Taking the first
    speaker's cap instead made the answer depend on speaker order.

    Both dicts are keyed by speaker id. 0 means unbounded, so it can never win a
    min against a real cap.
    """
    limits: dict[str, int] = {}
    for speaker_id, provider in providers.items():
        limit = provider.max_concurrency(models[speaker_id])
        current = limits.get(provider.name)
        if current is None or current == 0:
            limits[provider.name] = limit
        elif limit != 0:
            limits[provider.name] = min(current, limit)
    return limits


def _safe_title(title: str) -> str:
    """Convert a podcast title to a filesystem-safe slug."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_") or "podcast"


@dataclass(frozen=True, slots=True)
class RenderUnit:
    """One TTS request's worth of the script.

    Either a single segment (`indices` of length 1) or a consecutive run of
    turns rendered together as one dialogue. Stitching works in units, so the
    pause after a unit comes from its *last* segment.
    """

    indices: tuple[int, ...]
    speaker_id: str
    """Speaker of the first segment; for dialogue units, only its provider and
    model matter — the turns carry their own voices."""
    is_dialogue: bool


def _plan_render_units(
    segments: list[Segment],
    speaker_map: dict[str, Speaker],
    providers: dict[str, TTSProvider],
    models: dict[str, str],
    render_mode: PodcastRenderMode,
) -> list[RenderUnit]:
    """Split the script into TTS requests.

    In "segments" mode every segment is its own unit — the historical behavior.

    In "dialogue" mode, maximal runs of consecutive segments that share a
    dialogue-capable provider *and* model are batched into one request, so the
    model paces the exchange itself. Runs are further split to stay under the
    provider's per-request character budget, always at a turn boundary. Anything
    that cannot participate — an OpenAI speaker, a non-dialogue model, a
    lone turn — falls back to its own segment unit, which is what makes
    dialogue mode compose with mixed-provider episodes.
    """
    if render_mode == "segments":
        return [RenderUnit((i,), seg["speaker"], False) for i, seg in enumerate(segments)]

    units: list[RenderUnit] = []
    run: list[int] = []
    run_key: tuple[str, str] | None = None
    run_chars = 0

    def flush() -> None:
        nonlocal run, run_key, run_chars
        if not run:
            return
        # A single turn gains nothing from the dialogue endpoint and would lose
        # its per-speaker voice_settings, so render it normally.
        is_dialogue = len(run) >= MIN_DIALOGUE_TURNS
        if is_dialogue:
            units.append(RenderUnit(tuple(run), segments[run[0]]["speaker"], True))
        else:
            units.extend(RenderUnit((i,), segments[i]["speaker"], False) for i in run)
        run, run_key, run_chars = [], None, 0

    for i, segment in enumerate(segments):
        speaker = speaker_map[segment["speaker"]]
        provider = providers[speaker["id"]]
        model = models[speaker["id"]]
        dialogue = as_dialogue_provider(provider)

        if dialogue is None or not dialogue.supports_dialogue_model(model):
            flush()
            units.append(RenderUnit((i,), segment["speaker"], False))
            continue

        key = (provider.name, model)
        length = len(segment["text"])
        budget = dialogue.max_dialogue_chars(model)

        if run_key is not None and (key != run_key or run_chars + length > budget):
            flush()
        run_key = key
        run.append(i)
        run_chars += length

    flush()
    return units


async def generate_podcast(
    script: PodcastScript,
    model: SpeechModel | ElevenLabsModel = "gpt-4o-mini-tts",
    provider: TTSProviderName = "openai",
) -> PodcastResult:
    """Generate a multi-voice podcast from a structured PodcastScript.

    Args:
        script: The podcast script. Speakers may each pick their own provider,
            so one episode can mix OpenAI and ElevenLabs voices.
        model: Default model for the speakers whose provider it belongs to.
            Speakers on the other provider fall back to that provider's default
            unless they set their own `model`.
        provider: Episode-wide default provider, itself overridden by
            `config.provider` and then by each speaker's `provider`.

    Raises ValueError if the script fails validation.
    """
    title, speakers, segments, config = _validate_script(script, default_provider=provider, default_model=model)
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
    override = config.get("max_concurrency")
    limiters: dict[str, anyio.CapacityLimiter | None] = {}
    for provider_name, derived in _derive_concurrency_limits(providers, models).items():
        limit = override or derived
        limiters[provider_name] = anyio.CapacityLimiter(limit) if limit else None
        if limit:
            logger.info("Limiting %s to %d concurrent TTS requests", provider_name, limit)

    render_mode: PodcastRenderMode = config.get("render_mode", DEFAULT_RENDER_MODE)
    units = _plan_render_units(segments, speaker_map, providers, models, render_mode)
    if render_mode == "dialogue":
        dialogue_units = [u for u in units if u.is_dialogue]
        if dialogue_units:
            logger.info(
                "Dialogue mode: %d/%d segments batched into %d conversation request(s); "
                "the model paces those turns, so their pause_after is not applied",
                sum(len(u.indices) for u in dialogue_units),
                len(segments),
                len(dialogue_units),
            )
        else:
            logger.warning(
                "render_mode='dialogue' but no run of 2+ consecutive turns shares a "
                "dialogue-capable provider and model (eleven_v3) - rendering per segment"
            )

    async def _gen_segment(i: int, segment: Segment) -> bytes:
        """Render one segment. Independent of the others, so #35's verify pass
        can re-invoke this for just the segments that failed QC."""
        speaker = speaker_map[segment["speaker"]]
        speaker_provider = providers[speaker["id"]]
        speed, voice_settings = _resolve_segment_speech(segment, speaker)
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
            voice_settings=voice_settings,
        )
        return await synthesize_speech(speaker_provider, request, limiter=limiters[speaker_provider.name])

    async def _gen_dialogue(unit: RenderUnit) -> bytes:
        """Render a run of consecutive turns as one conversation."""
        speaker_provider = providers[unit.speaker_id]
        dialogue = as_dialogue_provider(speaker_provider)
        if dialogue is None:  # pragma: no cover - _plan_render_units guarantees this
            raise RuntimeError(f"provider {speaker_provider.name!r} cannot render dialogue")

        turns = [
            DialogueTurn(
                text=segments[i]["text"],
                voice=speaker_provider.resolve_voice(speaker_map[segments[i]["speaker"]]["voice"]),
            )
            for i in unit.indices
        ]
        names = ", ".join(dict.fromkeys(speaker_map[segments[i]["speaker"]]["name"] for i in unit.indices))
        logger.info(
            f"Queued dialogue segments {unit.indices[0] + 1}-{unit.indices[-1] + 1}/{len(segments)} "
            f"[{names} / {speaker_provider.name}]"
        )
        limiter = limiters[speaker_provider.name]
        if limiter is None:
            return await dialogue.synthesize_dialogue(turns, models[unit.speaker_id], config.get("dialogue_stability"))
        async with limiter:
            return await dialogue.synthesize_dialogue(turns, models[unit.speaker_id], config.get("dialogue_stability"))

    async def _gen_unit(unit: RenderUnit) -> bytes:
        if unit.is_dialogue:
            return await _gen_dialogue(unit)
        index = unit.indices[0]
        return await _gen_segment(index, segments[index])

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _gen_unit, unit) for unit in units]

    # Read by index, not completion order — the limiter only delays task entry.
    segment_bytes_list = [c.result() for c in captures]

    # Pauses are per unit: a dialogue unit's internal gaps belong to the model,
    # so only the pause after its final turn is applied here.
    default_pause_ms = config.get("default_pause_ms", 600)
    pause_ms_list: list[int] = []
    for unit_index, unit in enumerate(units):
        if unit_index == len(units) - 1:
            pause_ms_list.append(0)
        else:
            last = segments[unit.indices[-1]]
            pause_ms_list.append(last.get("pause_after", default_pause_ms))

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
