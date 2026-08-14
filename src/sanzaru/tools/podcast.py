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

import pathlib
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Literal, NoReturn, NotRequired, TypedDict, cast

import anyio
from aioresult import ResultCapture  # type: ignore[import-untyped]
from openai.types.audio.speech_model import SpeechModel
from pydantic import BaseModel, Field
from pydub import AudioSegment  # type: ignore[import-untyped]
from pydub.effects import normalize as pydub_normalize  # type: ignore[import-untyped]

from ..audio.constants import (
    DEFAULT_RENDER_MODE,
    ELEVENLABS_MODELS,
    ELEVENLABS_SPEED_RANGE,
    MIN_DIALOGUE_SPEAKERS,
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
from ..audio.verification import similarity, transcribe_bytes, words
from ..config import logger
from ..infrastructure import FileSystemRepository

_ELEVENLABS_MODEL_NAMES = frozenset(ELEVENLABS_MODELS)

# Defaults the render path already applied via `config.get(...)` while
# `_validate_script` rejected scripts that omitted the same keys — the validator
# was strictly stricter than the renderer, so the requirement bought nothing but
# an edit cycle (#36). Named once here so the two can never disagree.
DEFAULT_PAUSE_MS = 600
DEFAULT_NORMALIZE_LOUDNESS = True
DEFAULT_OUTPUT_FORMAT: Literal["mp3", "wav"] = "mp3"

#: Neutral speed. ElevenLabs' no-speed models require exactly this, so it is
#: also the only value that is safe to supply on a speaker's behalf.
DEFAULT_SPEAKER_SPEED = 1.0

#: Stand-in title. No timestamp: the generated filename appends its own, and
#: `_safe_title` falls back to this same value, so naming it once keeps the two
#: from drifting the way the config defaults above would have.
DEFAULT_TITLE = "podcast"


class Speaker(TypedDict):
    """One voice in an episode, as authored.

    `name` and `voice` are the only fields a caller must supply: everything else
    has a defensible default, and demanding it just costs an edit cycle (#36).
    `_validate_script` returns speakers with `id` and `speed` filled in, so the
    render path can index them unconditionally.
    """

    name: str
    voice: str
    id: NotRequired[str]
    speed: NotRequired[float]
    instructions: NotRequired[str]
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
    """Episode-wide render settings. Every field is optional — the render path
    already defaulted these three, so requiring them only cost a round trip."""

    default_pause_ms: NotRequired[int]
    intro_silence_ms: NotRequired[int]
    outro_silence_ms: NotRequired[int]
    normalize_loudness: NotRequired[bool]
    output_format: NotRequired[Literal["mp3", "wav"]]
    output_bitrate: NotRequired[str]
    provider: NotRequired[TTSProviderName]
    max_concurrency: NotRequired[int]
    render_mode: NotRequired[PodcastRenderMode]
    dialogue_stability: NotRequired[float]


class PodcastScript(TypedDict):
    """A whole episode. Only `speakers` and `segments` are required."""

    speakers: list[Speaker]
    segments: list[Segment]
    title: NotRequired[str]
    description: NotRequired[str]
    config: NotRequired[PodcastConfig]


class SegmentVerdict(BaseModel):
    """What verification found for one segment of the script (#35)."""

    index: int
    speaker: str
    ok: bool
    reason: str = ""
    """Empty when ok; otherwise `tail_missing`, `segment_missing`, `diverged`,
    or `not_transcribed` (verification itself failed for this unit)."""
    similarity: float = 1.0
    """Word overlap between the script text and what the audio actually says."""
    retried: bool = False
    """True when this segment's audio was re-rendered before the final stitch."""


class PodcastResult(BaseModel):
    """Result from generate_podcast."""

    output_file: str
    title: str
    segment_count: int
    estimated_duration_seconds: float
    speakers: list[str]
    transcript: str
    verified: bool | None = None
    """None when `verify` was not requested. True when every segment was
    confirmed present in the rendered audio, False when some were not — the
    episode is still written either way."""
    verify_retries: int = 0
    """Segments re-rendered because verification flagged them."""
    segment_verdicts: list[SegmentVerdict] = Field(default_factory=list)
    """Per-segment findings; empty unless `verify` was requested."""


VERIFY_TAIL_WORDS = 8
"""How much of a segment's ending to check.

The documented failure is a *tail* drop — the last words of a segment simply
absent from the audio — so the check is aimed there rather than at the segment
as a whole, where a few missing words barely move an overall score."""

VERIFY_TAIL_THRESHOLD = 0.75
"""Fuzzy-match floor for that tail. Below it, the words are not there.

Loose on purpose: ASR reorders currency, spells numbers differently, and wobbles
on proper nouns. This is looking for absent speech, not for a perfect reading."""

VERIFY_SHORT_SEGMENT_WORDS = 4
"""At or under this, a segment is checked for presence anywhere in the audio
instead of by its tail — a four-word segment has no tail to speak of, and whole
short segments are the other documented failure (one three-word segment
vanished entirely)."""

VERIFY_SEGMENT_THRESHOLD = 0.60
"""Overall floor for a whole segment. A tail can pass while the middle is gone;
this catches the gross case without flagging normal ASR disagreement."""


def _tail_of(text: str, count: int = VERIFY_TAIL_WORDS) -> str:
    return " ".join(words(text)[-count:])


def _best_window_similarity(needle: str, haystack: str) -> float:
    """Best match for `needle` anywhere in `haystack`, at word level.

    A segment's words sit somewhere inside the rendered text, not at a known
    offset, so comparing the two whole strings would score a short segment
    against a long transcript as almost total mismatch. Sliding a
    needle-sized window is what makes "is this present at all?" answerable.
    """
    needle_words = words(needle)
    hay_words = words(haystack)
    if not needle_words:
        return 1.0
    if not hay_words:
        return 0.0
    span = len(needle_words)
    if len(hay_words) <= span:
        return similarity(needle, haystack)
    return max(
        similarity(needle, " ".join(hay_words[start : start + span])) for start in range(len(hay_words) - span + 1)
    )


def _verdict_for(index: int, speaker: str, intended: str, rendered: str) -> SegmentVerdict:
    """Judge one segment against the audio the unit containing it produced."""
    overall = _best_window_similarity(intended, rendered)
    if len(words(intended)) <= VERIFY_SHORT_SEGMENT_WORDS:
        # No meaningful tail; the question is only whether it is there at all.
        ok = overall >= VERIFY_TAIL_THRESHOLD
        return SegmentVerdict(
            index=index,
            speaker=speaker,
            ok=ok,
            reason="" if ok else "segment_missing",
            similarity=round(overall, 3),
        )

    tail_score = _best_window_similarity(_tail_of(intended), rendered)
    if tail_score < VERIFY_TAIL_THRESHOLD:
        reason = "tail_missing"
    elif overall < VERIFY_SEGMENT_THRESHOLD:
        reason = "diverged"
    else:
        reason = ""
    return SegmentVerdict(
        index=index,
        speaker=speaker,
        ok=not reason,
        reason=reason,
        similarity=round(min(overall, tail_score), 3),
    )


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


def _speaker_label(index: int, speaker: object) -> str:
    """`Speaker 0`, plus its id or name when it has one.

    An author fixing a four-speaker script should not have to count array
    positions to find the one being complained about.
    """
    if isinstance(speaker, dict):
        for key in ("id", "name"):
            value = speaker.get(key)
            if isinstance(value, str) and value.strip():
                return f"Speaker {index} ({value.strip()!r})"
    return f"Speaker {index}"


def _raise_validation_errors(errors: list[str]) -> NoReturn:
    """Report every problem found in this pass, not just the first.

    One error keeps the historical single-line message. Several are listed
    together: the caller is usually an agent that has to re-read the whole
    schema on each round trip, so N errors must not cost N renders (#53).
    """
    if len(errors) == 1:
        raise ValueError(errors[0])
    listed = "\n".join(f"  - {error}" for error in errors)
    raise ValueError(f"PodcastScript has {len(errors)} problems:\n{listed}")


def _normalize_speaker(speaker: Speaker) -> Speaker:
    """Apply the speaker defaults from #36 / #53.

    `instructions` is deliberately left absent rather than defaulted to `""`:
    the render path distinguishes "no direction" from an explicitly empty
    `instruction_override`, and ElevenLabs warns on any non-empty value.
    """
    normalized = dict(speaker)
    # Strip an explicit id as well as a derived one: otherwise {"id": " Alex "}
    # demands {"speaker": " Alex "} back — the same trap stripping the derived
    # one avoids — and " Alex " vs "Alex" are two ids that print identically in
    # every log line and error message.
    explicit_id = normalized.get("id")
    if isinstance(explicit_id, str):
        normalized["id"] = explicit_id.strip()
    name = normalized.get("name")
    if not normalized.get("id") and isinstance(name, str) and name.strip():
        # Segments may then reference the speaker by name, which is what an
        # author writes first anyway.
        normalized["id"] = name.strip()
    normalized.setdefault("speed", DEFAULT_SPEAKER_SPEED)
    return cast(Speaker, normalized)


#: Same shape `SimulationBrief.filename` enforces via pydantic: a bare name, no
#: separators of either flavour. Kept in sync deliberately — the two podcast
#: tools should not disagree about what a filename is.
_FILENAME_MAX_LEN = 200


def _check_output_filename(filename: str | None, output_format: str) -> str | None:
    """Validate a caller-supplied episode name, cheaply and early.

    Returns the name to use, or None to fall back to the generated slug. Empty
    and whitespace-only are treated as absent rather than rejected, so the
    caller and the naming below agree on what "no name given" means.

    Separators are refused rather than sanitized. The local backend would let
    `sub/ep.mp3` through `validate_safe_path` and then fail on a missing
    directory, and the Databricks backend silently *strips* the directory —
    writing `ep.mp3` while `PodcastResult.output_file` still said `sub/ep.mp3`,
    which is precisely the two-answers problem this parameter exists to remove.
    """
    if filename is None or not filename.strip():
        return None
    if "/" in filename or "\\" in filename:
        raise ValueError(
            f"output filename {filename!r} must be a bare filename, not a path — it names a file in the audio directory"
        )
    if filename in (".", "..") or len(filename) > _FILENAME_MAX_LEN:
        raise ValueError(f"output filename {filename!r} is not a usable filename")
    if pathlib.PurePath(filename).suffix.lstrip(".").lower() != output_format:
        # Warn, don't raise: the caller may well want a different extension on
        # purpose. Here rather than at write time so it is not learned after
        # several minutes of synthesis.
        logger.warning(
            "Podcast filename %r does not end in .%s but the file will contain %s audio — "
            "rename it to .%s, or set config.output_format to match",
            filename,
            output_format,
            output_format,
            output_format,
        )
    return filename


def _validate_script(
    script: PodcastScript,
    default_provider: TTSProviderName = "openai",
    default_model: str | None = None,
) -> tuple[str, list[Speaker], list[Segment], PodcastConfig]:
    """Validate a PodcastScript, apply defaults, and return its components.

    `default_model` must be the same value the render path will use, or
    validation checks a model the episode never runs on.

    Errors are collected per phase rather than raised on the first one, because
    a later phase cannot be trusted once an earlier one failed: segment checks
    need resolved speaker ids, and speaker checks need a usable provider. Within
    a phase every problem is reported at once.

    Collection is across items, not within one: a speaker with both a bad
    `speed` and bad `voice_settings` reports only the speed, because each item's
    later checks depend on its earlier ones holding. So "one edit cycle" is a
    promise per item, not an absolute — deliberately, not an oversight.

    Returns speakers with `id` and `speed` filled in, so the render path can
    index them unconditionally.

    Raises ValueError if the script is invalid.
    """
    errors: list[str] = []

    # ---- phase 1: shape. Nothing else is checkable until these exist. ----
    for key in ("speakers", "segments"):
        if key not in script:
            errors.append(f"PodcastScript missing required field: '{key}'")
    if errors:
        _raise_validation_errors(errors)

    # Container types first. The script arrives from a raw `json.loads` with no
    # pydantic in the way, so anything below that is structurally used — len(),
    # enumerate(), dict() — would otherwise raise TypeError from deep inside and
    # land as an `internal` envelope at exit 1 rather than the usage error it is.
    # `{"speakers": {"host": {...}}}` is the one worth naming: an object keyed by
    # id instead of an array is a very plausible mistake, and it used to surface
    # as "dictionary update sequence element #0 has length 4; 2 is required".
    # Literal subscripts, not a loop variable: mypy cannot narrow the latter to
    # a TypedDict key.
    for key, value in (("speakers", script["speakers"]), ("segments", script["segments"])):
        if not isinstance(value, list):
            errors.append(f"PodcastScript '{key}' must be an array, got {type(value).__name__}")
    raw_config = script.get("config")
    if raw_config is not None and not isinstance(raw_config, dict):
        errors.append(f"PodcastScript 'config' must be an object, got {type(raw_config).__name__}")
    if errors:
        _raise_validation_errors(errors)

    speakers_raw = script["speakers"]
    if not speakers_raw:
        errors.append("PodcastScript must have at least 1 speaker")
    elif len(speakers_raw) > 4:
        errors.append("PodcastScript supports at most 4 speakers")
    if not script["segments"]:
        errors.append("PodcastScript must have at least 1 segment")

    # Speaker elements only. Segment elements are checked in phase 3, where the
    # rest of the segment rules live: a malformed segment does not stop speaker
    # validation, so raising on it here would suppress the very speaker errors
    # #53 is about — the caller would fix the segment, re-run, and only then
    # meet the speakers.
    for i, entry in enumerate(speakers_raw):
        if not isinstance(entry, dict):
            errors.append(f"Speaker {i} must be an object, got {type(entry).__name__}")
    if errors:
        _raise_validation_errors(errors)

    # `title` is presentational — it names the default output file and rides in
    # the result. A plain default is fine; refusing to render without one was
    # not (#36). No timestamp: the generated filename appends its own.
    raw_title = script.get("title")
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else DEFAULT_TITLE

    config: PodcastConfig = raw_config or cast(PodcastConfig, {})

    # ---- phase 2: config and speakers, collected together ----
    output_format = config.get("output_format", DEFAULT_OUTPUT_FORMAT)
    if output_format not in ("mp3", "wav"):
        errors.append(f"PodcastConfig 'output_format' must be 'mp3' or 'wav', got {output_format!r}")

    config_provider_ok = True
    if "provider" in config:
        try:
            validate_provider_name(config["provider"], "PodcastConfig 'provider'")
        except ValueError as exc:
            errors.append(str(exc))
            # Every speaker inherits this, so probing them would just restate it.
            config_provider_ok = False

    if "max_concurrency" in config and (
        not isinstance(config["max_concurrency"], int) or config["max_concurrency"] < 1
    ):
        errors.append(f"PodcastConfig 'max_concurrency' must be a positive integer, got {config['max_concurrency']}")

    # The silence knobs, whose segment-level counterpart (`pause_after`) is
    # already checked. A string here builds a pause list of strings, and
    # `sum(pause_ms_list)` in the duration estimate raises TypeError — exit 1.
    for ms_key, ms_value in (
        ("default_pause_ms", config.get("default_pause_ms")),
        ("intro_silence_ms", config.get("intro_silence_ms")),
        ("outro_silence_ms", config.get("outro_silence_ms")),
    ):
        if ms_value is not None and (isinstance(ms_value, bool) or not isinstance(ms_value, int)):
            errors.append(f"PodcastConfig {ms_key!r} must be an integer, got {type(ms_value).__name__}")

    render_mode: PodcastRenderMode = config.get("render_mode", DEFAULT_RENDER_MODE)
    if render_mode not in RENDER_MODES:
        errors.append(f"PodcastConfig 'render_mode' must be one of: {', '.join(RENDER_MODES)}, got {render_mode!r}")
    if "dialogue_stability" in config:
        stability = config["dialogue_stability"]
        if not isinstance(stability, int | float) or not 0.0 <= stability <= 1.0:
            errors.append(f"PodcastConfig 'dialogue_stability' must be between 0.0 and 1.0, got {stability}")
        elif render_mode != "dialogue":
            logger.warning("PodcastConfig 'dialogue_stability' is ignored unless render_mode is 'dialogue'")

    speakers = [_normalize_speaker(s) for s in speakers_raw]

    # Two sets, and they are equal by the time phase 3 runs — any speaker that
    # bails out early also appends to `errors`, which raises before then. They
    # are kept apart because they answer different questions at different times,
    # and collapsing them would couple this to that reasoning staying true.
    #
    # `speaker_ids` is the fully-validated set, populated at the bottom of a
    # successful iteration in lockstep with `speaker_providers`/`speaker_probes`
    # — so phase 3 can index those safely for any id it contains.
    speaker_ids: set[str] = set()
    # `seen_ids` is every id whose *shape* was valid, whether or not the rest of
    # that speaker checked out. Duplicate detection has to read this one: a
    # speaker that bails out early never reaches the bottom of the loop, so
    # testing against the validated set would hide a collision behind an
    # unrelated error on the first of the pair — the extra round trip #53 exists
    # to remove.
    seen_ids: set[str] = set()
    # Per speaker id, so the segment pass below can re-probe with the same
    # provider, model, and voice the render will use.
    speaker_providers: dict[str, TTSProviderName] = {}
    speaker_probes: dict[str, tuple[Speaker, TTSProvider, SpeechRequest]] = {}

    for i, speaker in enumerate(speakers):
        label = _speaker_label(i, speaker)

        # Id bookkeeping first. A missing `voice` does not stop us knowing the
        # id, so registering it before the required-field check is what lets a
        # duplicate be reported in the *same* pass as the missing field rather
        # than on the next run — the round trip #53 exists to remove.
        #
        # Read through `.get`: `id` is only synthesized from a non-blank string
        # name, so a name of "   " (or a non-string) leaves it absent — and a
        # presence check cannot see that, because the key *is* there.
        # `name` and `voice` are the only fields with no defensible default.
        missing = [field for field in ("name", "voice") if field not in speaker]

        speaker_id = speaker.get("id")
        id_ok = isinstance(speaker_id, str) and bool(speaker_id.strip())
        if not id_ok:
            # Silent when `name` is what is missing: that error is already
            # reported below and is the actual diagnosis, so saying "needs an
            # id, or a name to derive one from" too would be one defect
            # described twice.
            if "name" not in missing:
                errors.append(f"{label} needs a non-empty string 'id', or a non-empty 'name' to derive one from")
        elif speaker_id in seen_ids:
            # Newly reachable now that `id` defaults to `name`: two speakers
            # sharing one would silently collapse in the render path's
            # speaker_map, and only one of them would ever be heard.
            errors.append(f"{label} duplicates the speaker id {speaker_id!r}; ids must be unique")
            id_ok = False
        else:
            seen_ids.add(cast(str, speaker_id))

        if missing:
            errors.append(f"{label} missing required field(s): {', '.join(repr(f) for f in missing)}")
        # Present but wrong type. This is the sharpest case in the family: with
        # an explicit `id`, a non-string `name` passes every other check, the
        # episode renders and is *written*, and only then does
        # `PodcastResult(speakers=[s["name"] ...])` fail pydantic — exit 1, with
        # paid-for audio orphaned on disk under a name the caller never learns.
        wrong_type = [
            (field, value)
            for field, value in (("name", speaker.get("name")), ("voice", speaker.get("voice")))
            if field not in missing and not isinstance(value, str)
        ]
        if wrong_type:
            errors.append(
                f"{label} field(s) must be strings: "
                + ", ".join(f"{field!r} is {type(value).__name__}" for field, value in wrong_type)
            )
        if "speed" in speaker and (isinstance(speaker["speed"], bool) or not isinstance(speaker["speed"], int | float)):
            # `"speed": "1.0"` is a classic hand-authored-JSON slip, and #36
            # making the field optional means the authors most likely to type it
            # are the ones writing it by hand. Left unchecked it reached a
            # float/str comparison — a TypeError, i.e. exit 1, blaming the tool.
            errors.append(f"{label} 'speed' must be a number, got {type(speaker['speed']).__name__}")
            continue
        if missing or wrong_type or not id_ok:
            continue

        speaker_ok = True
        if "provider" in speaker:
            try:
                validate_provider_name(speaker["provider"], f"{label} 'provider'")
            except ValueError as exc:
                errors.append(str(exc))
                speaker_ok = False
        if not speaker_ok or not config_provider_ok:
            continue

        provider_name = _resolve_provider_name(speaker, config, default_provider)

        low, high = _speed_range(provider_name)
        if not low <= speaker["speed"] <= high:
            errors.append(
                f"{label} speed must be between {low} and {high} for provider='{provider_name}', got {speaker['speed']}"
            )
            continue

        if "voice_settings" in speaker:
            try:
                check_voice_settings_types(speaker["voice_settings"], f"{label} ")
            except ValueError as exc:
                errors.append(str(exc))
                continue

        # Fail before spending a single API call: this resolves the model,
        # rejects cross-provider model names, and applies provider-specific
        # rules (eleven_v3 has no speed, ElevenLabs needs a voice id).
        provider = get_provider(provider_name)
        try:
            model = _resolve_model(speaker, provider, default_model)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue

        if render_mode == "dialogue" and "voice_settings" in speaker:
            # The dialogue endpoint takes one `stability` for the whole request,
            # not per-voice settings. Gate on the *model*, not just the provider:
            # only eleven_v3 can batch, so an eleven_multilingual_v2 speaker is
            # guaranteed to keep its voice_settings and must not be warned.
            # Still only a warning for the ones that can — a qualifying speaker
            # may yet fall back to segment rendering if no run forms around it.
            dialogue_capable = as_dialogue_provider(provider)
            if dialogue_capable is not None and dialogue_capable.supports_dialogue_model(model):
                logger.warning(
                    "Speaker %d ('%s') sets voice_settings, which the dialogue endpoint cannot apply "
                    "per speaker - use config.dialogue_stability, or render_mode='segments' to keep them",
                    i,
                    speaker["id"],
                )

        try:
            probe = SpeechRequest(
                text="validation probe",
                voice=provider.resolve_voice(speaker["voice"]),
                model=model,
                speed=speaker["speed"],
                instructions=speaker.get("instructions"),
                voice_settings=speaker.get("voice_settings"),
            )
            provider.validate(probe)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue

        speaker_ids.add(speaker["id"])
        speaker_providers[speaker["id"]] = provider_name
        speaker_probes[speaker["id"]] = (speaker, provider, probe)

    if errors:
        _raise_validation_errors(errors)

    # ---- phase 3: segments, which need the resolved speaker ids above ----
    segments = script["segments"]

    for i, segment in enumerate(segments):
        # `["Alex: hello"]` — a flat list of strings — is the single most
        # plausible thing an agent hands over without reading the schema, and it
        # did not even crash: `"speaker" in "Alex: hello"` is a *substring*
        # test, so it reported missing fields on something that is not an object.
        if not isinstance(segment, dict):
            errors.append(f"Segment {i} must be an object, got {type(segment).__name__}")
            continue
        missing = [field for field in ("speaker", "text") if field not in segment]
        if missing:
            errors.append(f"Segment {i} missing required field(s): {', '.join(repr(f) for f in missing)}")
            continue
        # Leaf types before anything uses them structurally: a non-string
        # `speaker` is unhashable against the id set and a non-string `text`
        # has no `.strip()` — both TypeError/AttributeError, i.e. exit 1.
        if not isinstance(segment["speaker"], str):
            errors.append(f"Segment {i} 'speaker' must be a string, got {type(segment['speaker']).__name__}")
            continue
        if not isinstance(segment["text"], str):
            errors.append(f"Segment {i} 'text' must be a string, got {type(segment['text']).__name__}")
            continue
        if segment["speaker"] not in speaker_ids:
            known = ", ".join(sorted(speaker_ids))
            errors.append(f"Segment {i} references unknown speaker id: '{segment['speaker']}' — known ids: {known}")
            continue
        if not segment["text"].strip():
            errors.append(f"Segment {i} text must not be empty")
            continue
        if len(segment["text"]) > 40000:
            errors.append(f"Segment {i} text exceeds 40000 characters")
            continue
        if "pause_after" in segment and (
            isinstance(segment["pause_after"], bool) or not isinstance(segment["pause_after"], int)
        ):
            # Detonates in pydub rather than here if it gets through.
            errors.append(f"Segment {i} 'pause_after' must be an integer, got {type(segment['pause_after']).__name__}")
            continue
        if "speed_override" in segment and (
            isinstance(segment["speed_override"], bool) or not isinstance(segment["speed_override"], int | float)
        ):
            errors.append(
                f"Segment {i} 'speed_override' must be a number, got {type(segment['speed_override']).__name__}"
            )
            continue
        if "speed_override" in segment:
            # Checked against the owning speaker's provider range, which is why
            # this runs after the speaker pass.
            low, high = _speed_range(speaker_providers[segment["speaker"]])
            if not low <= segment["speed_override"] <= high:
                errors.append(
                    f"Segment {i} speed_override must be between {low} and {high} for "
                    f"provider='{speaker_providers[segment['speaker']]}', got {segment['speed_override']}"
                )
                continue
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
                errors.append(f"Segment {i} speed_override: {exc}")

    if errors:
        _raise_validation_errors(errors)

    return title, speakers, segments, config


def _estimate_duration(
    segments: list[Segment],
    speakers: list[Speaker],
    pause_ms_list: list[int],
    config: PodcastConfig,
) -> float:
    """Estimate total podcast duration in seconds (~150 wpm).

    `pause_ms_list` must be the very list the stitch step inserts — one entry per
    render *unit*, not per segment — or the estimate reports silence the output
    never contains. Deriving it here from `pause_after` instead over-reported
    twice over: it counted a trailing pause after the final segment, and in
    dialogue mode it counted the intra-run gaps the model paces itself.
    """
    speaker_speeds = {s["id"]: float(s["speed"]) for s in speakers}

    speech_seconds = 0.0
    for segment in segments:
        word_count = len(segment["text"].split())
        speed = float(segment["speed_override"]) if "speed_override" in segment else speaker_speeds[segment["speaker"]]
        speech_seconds += word_count * 60.0 / (150.0 * speed)

    intro_ms = int(config.get("intro_silence_ms") or 0)
    outro_ms = int(config.get("outro_silence_ms") or 0)

    return speech_seconds + (sum(pause_ms_list) + intro_ms + outro_ms) / 1000.0


def _decode_mp3(raw_bytes: bytes) -> AudioSegment:
    """Default segment decoder: every TTS provider is asked for mp3."""
    return AudioSegment.from_mp3(BytesIO(raw_bytes))


def _stitch_audio(
    segment_bytes_list: list[bytes],
    pause_ms_list: list[int],
    intro_ms: int,
    outro_ms: int,
    normalize_loudness: bool,
    output_format: str,
    output_bitrate: str,
    decode: Callable[[bytes], AudioSegment] = _decode_mp3,
) -> bytes:
    """Stitch audio segments with silence gaps using pydub.

    This is CPU-bound work that runs in a thread pool.

    Args:
        segment_bytes_list: Encoded audio for each segment, in `decode`'s format.
        pause_ms_list: Silence duration in ms after each segment (same length as segment_bytes_list).
        intro_ms: Silence in ms before the first segment.
        outro_ms: Silence in ms after the last segment.
        normalize_loudness: Whether to peak-normalize each segment.
        output_format: Output format ("mp3" or "wav").
        output_bitrate: MP3 bitrate string (e.g., "192k"). Ignored for WAV.
        decode: Bytes → AudioSegment. Defaults to mp3, which is the TTS contract;
            simulated podcasts pass a raw-PCM decoder instead, since realtime
            audio never needs to become mp3 on the way here.

    Returns:
        Final concatenated audio as bytes.
    """
    combined = AudioSegment.silent(duration=intro_ms) if intro_ms > 0 else AudioSegment.empty()

    for raw_bytes, pause_ms in zip(segment_bytes_list, pause_ms_list, strict=True):
        seg = decode(raw_bytes)
        # Sources differ in native rate (OpenAI TTS and realtime 24kHz,
        # ElevenLabs mp3_44100_128 44.1kHz). pydub would resample on
        # concatenation anyway, but pinning it here makes a mixed episode
        # deterministic regardless of segment order.
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
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_") or DEFAULT_TITLE


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
    model paces the exchange itself. Runs are further split to stay within the
    provider's per-request character budget, always at a turn boundary. Anything
    that cannot participate — an OpenAI speaker, a non-dialogue model, a turn
    that alone fills the budget, a lone turn, a run with only one voice in it —
    falls back to its own segment unit, which is what makes dialogue mode
    compose with mixed-provider episodes.
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
        # A run in one voice has no turn-taking left for the model to pace, so
        # it would cost a dialogue request and drop every pause_after between
        # those paragraph beats for nothing. Distinct *voices*, not speaker ids:
        # two speaker entries can point at the same ElevenLabs voice, and the
        # endpoint only hears the voice. At MIN_DIALOGUE_SPEAKERS=2 this also
        # covers the lone turn, which would lose its per-speaker voice_settings.
        voices_in_run = {
            providers[segments[i]["speaker"]].resolve_voice(speaker_map[segments[i]["speaker"]]["voice"]) for i in run
        }
        if len(voices_in_run) >= MIN_DIALOGUE_SPEAKERS:
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

        # The budget is the only length rule: it is below every provider's
        # per-chunk budget, so a turn too long to share a dialogue request is
        # also one segments mode would happily send whole. Such a turn opens a
        # run of its own, which the next turn (or the final flush) closes into a
        # single-voice run — i.e. a segment unit, chunked normally. That is also
        # why a batched run can never exceed the budget: only the first turn of
        # a run skips the check below.
        if run_key is not None and (key != run_key or run_chars + length > budget):
            flush()
        run_key = key
        run.append(i)
        run_chars += length

    flush()
    return units


def _build_pause_list(units: list[RenderUnit], segments: list[Segment], default_pause_ms: int) -> list[int]:
    """Silence to insert after each render unit.

    Pauses are per unit, not per segment: a dialogue unit's internal gaps belong
    to the model, so only the pause after its *last* turn survives. The final
    unit gets no trailing pause — `outro_silence_ms` is the knob for that.

    Both the stitch step and the duration estimate read this list, which is the
    only way the estimate can stay honest about silence that is actually added.
    """
    return [
        0 if unit_index == len(units) - 1 else segments[unit.indices[-1]].get("pause_after", default_pause_ms)
        for unit_index, unit in enumerate(units)
    ]


async def generate_podcast(
    script: PodcastScript,
    model: SpeechModel | ElevenLabsModel = "gpt-4o-mini-tts",
    provider: TTSProviderName = "openai",
    filename: str | None = None,
    verify: bool = False,
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
        filename: Name to write the episode under, mirroring
            `SimulationBrief.filename`. Defaults to a title-and-timestamp slug.
            `PodcastResult.output_file` always reports the name actually
            written, so a caller never has to guess which of two names is real.
        verify: Transcribe each rendered unit and check it actually says what
            the script said, re-rendering the ones that do not, once (#35). TTS
            drops segment tails and whole short segments at random, with no
            error — and `transcript` is only an echo of the input, so it is no
            evidence. Costs one transcription per unit and is off by default.

    Raises ValueError if the script fails validation.
    """
    title, speakers, segments, config = _validate_script(script, default_provider=provider, default_model=model)
    # Before a single TTS request goes out. The storage layer would catch a
    # traversal attempt too, but only at the final write — after the whole
    # episode has been synthesized and billed, and with the audio then dropped.
    filename = _check_output_filename(filename, config.get("output_format", DEFAULT_OUTPUT_FORMAT))
    speaker_map: dict[str, Speaker] = {s["id"]: s for s in speakers}

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
                "render_mode='dialogue' but no run of 2+ consecutive turns in 2+ distinct voices shares a "
                "dialogue-capable provider and model (eleven_v3) - rendering per segment"
            )

    pause_ms_list = _build_pause_list(units, segments, config.get("default_pause_ms", DEFAULT_PAUSE_MS))
    estimated_duration = _estimate_duration(segments, speakers, pause_ms_list, config)
    logger.info(
        f"Podcast '{title}': {len(segments)} segments, {len(speakers)} speakers, ~{estimated_duration:.0f}s estimated"
    )

    async def _gen_segment(i: int, segment: Segment) -> bytes:
        """Render one segment. Independent of the others, so #35's verify pass
        can re-invoke this for just the segments that failed QC."""
        speaker = speaker_map[segment["speaker"]]
        speaker_provider = providers[speaker["id"]]
        speed, voice_settings = _resolve_segment_speech(segment, speaker)
        # `in`-check rather than `or`: an intentional empty-string override must
        # not silently fall back to the speaker's instructions.
        instructions = (
            segment["instruction_override"] if "instruction_override" in segment else speaker.get("instructions")
        )
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

    intro_ms = config.get("intro_silence_ms") or 0
    outro_ms = config.get("outro_silence_ms") or 0
    normalize_loudness = config.get("normalize_loudness", DEFAULT_NORMALIZE_LOUDNESS)
    output_format = config.get("output_format", DEFAULT_OUTPUT_FORMAT)
    output_bitrate = config.get("output_bitrate", "192k")

    async def _verify_units(targets: list[int]) -> dict[int, list[SegmentVerdict]]:
        """Transcribe the given units' audio and judge their segments.

        Verifying *before* stitching is what makes attribution exact: each
        unit's bytes are still separate and index-aligned to `units`, every one
        is far under the transcription limits, and no offset arithmetic against
        the finished file is needed. It does not prove the stitch — but the
        documented failure is TTS dropping speech, and stitching is a
        deterministic concatenation.
        """
        verify_limiter = anyio.CapacityLimiter(4)

        async def _one(unit_index: int) -> tuple[int, list[SegmentVerdict]]:
            unit = units[unit_index]
            try:
                async with verify_limiter:
                    rendered = await transcribe_bytes(
                        segment_bytes_list[unit_index], f"unit{unit_index}.{output_format}"
                    )
            except Exception as exc:  # noqa: BLE001 - verification must never lose the episode
                logger.warning("Verification could not transcribe unit %d: %s", unit_index, exc)
                return unit_index, [
                    SegmentVerdict(
                        index=i,
                        speaker=speaker_map[segments[i]["speaker"]]["name"],
                        ok=True,
                        reason="not_transcribed",
                        similarity=0.0,
                    )
                    for i in unit.indices
                ]
            return unit_index, [
                _verdict_for(i, speaker_map[segments[i]["speaker"]]["name"], segments[i]["text"], rendered)
                for i in unit.indices
            ]

        async with anyio.create_task_group() as verify_tg:
            verify_captures = [ResultCapture.start_soon(verify_tg, _one, i) for i in targets]
        return dict(capture.result() for capture in verify_captures)

    verdicts_by_unit: dict[int, list[SegmentVerdict]] = {}
    retried_count = 0
    if verify:
        logger.info("Verifying %d rendered unit(s) against the script", len(units))
        verdicts_by_unit = await _verify_units(list(range(len(units))))

        # A dialogue unit is one request or nothing — the model paced those turns
        # together — so a failure anywhere in a run re-renders the whole run.
        failed_units = [i for i, verdicts in verdicts_by_unit.items() if any(not v.ok for v in verdicts)]
        if failed_units:
            flagged = [f"{v.index + 1} ({v.reason})" for i in failed_units for v in verdicts_by_unit[i] if not v.ok]
            logger.warning("Verification flagged segment(s) %s - re-rendering once", ", ".join(flagged))
            async with anyio.create_task_group() as retry_tg:
                retry_captures = [(i, ResultCapture.start_soon(retry_tg, _gen_unit, units[i])) for i in failed_units]
            for unit_index, capture in retry_captures:
                segment_bytes_list[unit_index] = capture.result()
                retried_count += len(units[unit_index].indices)

            # Re-judge only what was re-rendered; a passing unit is untouched.
            for unit_index, verdicts in (await _verify_units(failed_units)).items():
                verdicts_by_unit[unit_index] = [v.model_copy(update={"retried": True}) for v in verdicts]

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

    # Already validated up front, before any synthesis was paid for.
    written_name = filename or f"{_safe_title(title)}_{int(time.time())}.{output_format}"
    file_repo = FileSystemRepository()
    await file_repo.write_audio_file(written_name, final_audio)
    logger.info(f"Podcast written: {written_name} ({len(final_audio):,} bytes)")

    transcript = "\n\n".join(f"**{speaker_map[s['speaker']]['name']}:** {s['text']}" for s in segments)

    verdicts = sorted(
        (v for unit_verdicts in verdicts_by_unit.values() for v in unit_verdicts),
        key=lambda v: v.index,
    )
    if verify:
        unresolved = [v for v in verdicts if not v.ok]
        if unresolved:
            logger.warning(
                "Verification: %d segment(s) still not found in the audio after one retry: %s. "
                "A segment that fails twice is unlikely to be fixed by a third render - "
                "rewrite the tail so it is grammatically part of a longer sentence",
                len(unresolved),
                ", ".join(f"{v.index + 1} ({v.reason})" for v in unresolved),
            )
        else:
            logger.info("Verification: every segment confirmed present in the rendered audio")

    return PodcastResult(
        output_file=written_name,
        title=title,
        segment_count=len(segments),
        estimated_duration_seconds=round(estimated_duration, 1),
        speakers=[s["name"] for s in speakers],
        transcript=transcript,
        verified=None if not verify else all(v.ok for v in verdicts),
        verify_retries=retried_count,
        segment_verdicts=verdicts,
    )
