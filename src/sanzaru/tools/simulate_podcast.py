# SPDX-License-Identifier: MIT
"""Simulated podcasts: N realtime agents actually converse.

`generate_podcast` speaks a script you wrote. This records a conversation that
did not exist until the models had it. Each host is a `gpt-realtime` session
with a persona; the producer gives one of them the floor, then plays that audio
into every other host's ears, so they respond to delivery and timing rather than
to a transcript. See `audio/realtime/` for the machinery.

Three structural decisions worth knowing before reading the code:

**Acts, not one long session.** The Realtime API closes a WebSocket at 60
minutes, so a long episode cannot be one session at any price. Chunking into
acts also keeps per-session context flat — a 30-minute episode costs ~192k input
tokens across 6 acts versus ~992k as one — and lets acts record in parallel,
which is what makes a blocking tool viable: ~1.1 minutes of wall clock for 30
minutes of audio.

**Acts are recorded blind to each other.** An act cannot hear another act. All
coherence comes from the rundown — `prior_context` (what came before),
`upcoming` (what later acts own, so this one stays out of it), and `handoff`
(where to leave the conversation). That is a real constraint, and the QC judge
exists to catch the place it fails: two acts covering the same ground.

**Every act is checkpointed the moment it finishes.** Realtime is the most
expensive thing sanzaru does, so a crash, a timeout, or a cost abort must never
throw away audio that was already paid for. `resume` re-reads the checkpoints and
records only what is missing.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import anyio
from aioresult import ResultCapture  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator

from ..audio.realtime import (
    ActBrief,
    ActResult,
    ActSummary,
    CostBudget,
    HostSpec,
    RealtimeUsage,
    Rundown,
    RundownRequest,
    SimulationSettings,
    Turn,
    TurnAudio,
    plan_rundown,
    run_act,
)
from ..audio.realtime.mixdown import (
    TimelineItem,
    decode_to_pcm,
    encode_pcm,
    pcm_to_segment,
    render_stem,
    slice_pcm_by_durations,
)
from ..audio.realtime.pricing import prices_for, project_usage, usage_cost
from ..audio.realtime.producer import DEFAULT_REALTIME_MODEL, DEFAULT_TURN_SECONDS, DEFAULT_TURN_TOKENS
from ..audio.realtime.qc import DEFAULT_JUDGE_MODEL, DEFAULT_TRANSCRIBE_MODEL, QCReport, run_qc
from ..audio.realtime.rundown import DEFAULT_PLANNER_MODEL
from ..audio.realtime.types import (
    MAX_ACTS,
    MAX_EPISODE_MINUTES,
    MAX_TURN_TOKENS,
    Bitrate,
    Filename,
    RunId,
)
from ..config import logger
from ..infrastructure import FileSystemRepository
from ..storage import StorageBackend, get_default_storage, get_storage
from .podcast import _safe_title, _stitch_audio

DEFAULT_MAX_SESSIONS = 6
"""Concurrent Realtime sessions allowed across all acts. Each act opens one per
host, so this caps parallelism at `sessions // hosts` acts. Deliberately
conservative: the account-level ceiling is not published, and exceeding it fails
a session mid-act. Raise with SANZARU_REALTIME_MAX_SESSIONS once you know yours."""

DEFAULT_ACT_GAP_MS = 700
"""A beat between acts. Turns inside an act are butted together — the models pace
themselves, and inserting silence between their turns makes it sound scripted."""


# ---------- request / result ----------


class SimulationBrief(BaseModel):
    """Everything one simulated episode needs.

    Two ways in: hand over a full `rundown` (from `sanzaru podcast rundown`, then
    edited), or give a `premise` and let pre-production plan the acts.
    """

    premise: str = Field(default="", max_length=20_000)
    rundown: Rundown | None = None
    title: str | None = Field(default=None, max_length=200)
    style: str | None = None
    hosts: list[HostSpec] = Field(default_factory=list, max_length=8)
    acts: int = Field(default=4, ge=1, le=MAX_ACTS)
    target_minutes: float = Field(default=12.0, gt=0, le=MAX_EPISODE_MINUTES)

    model: str = DEFAULT_REALTIME_MODEL
    planner_model: str = DEFAULT_PLANNER_MODEL
    turn_seconds: float = Field(default=DEFAULT_TURN_SECONDS, ge=2.0, le=120.0)
    turn_tokens: int = Field(default=DEFAULT_TURN_TOKENS, ge=0, le=MAX_TURN_TOKENS)
    """0 → derived from `turn_seconds`; see `producer.turn_token_cap`."""
    turn_timeout_s: float = Field(default=0.0, ge=0, le=3600.0)
    """Wall clock a single turn may take before the act is declared stalled.
    0 → SANZARU_REALTIME_TURN_TIMEOUT, else derived from `turn_seconds`."""

    max_concurrent_sessions: int = Field(default=0, ge=0, le=64)
    """0 → SANZARU_REALTIME_MAX_SESSIONS, else DEFAULT_MAX_SESSIONS."""
    max_cost_usd: float | None = Field(default=None, gt=0)

    run_id: RunId | None = None
    resume: bool = False
    stems: bool = False
    dry_run: bool = False

    qc: bool = True
    qc_retry: bool = False
    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL

    output_format: Literal["mp3", "wav"] = "mp3"
    output_bitrate: Bitrate = "192k"
    filename: Filename | None = None
    act_gap_ms: int = Field(default=DEFAULT_ACT_GAP_MS, ge=0, le=60_000)
    intro_silence_ms: int = Field(default=0, ge=0, le=60_000)
    outro_silence_ms: int = Field(default=0, ge=0, le=60_000)
    normalize_loudness: bool = False
    """Off by default, unlike scripted podcasts. Peak-normalizing each act
    separately can *introduce* level jumps between acts recorded by the same
    models at consistent levels."""

    @model_validator(mode="after")
    def _check_there_is_something_to_record(self) -> SimulationBrief:
        """Reject a brief with no episode in it before anything else happens."""
        if not (self.premise.strip() or self.rundown is not None or self.resume):
            raise ValueError("nothing to record: give a premise, a rundown, or resume=true with a run_id")
        if self.resume and not self.run_id and self.rundown is None:
            raise ValueError("resume=true needs a run_id (or a rundown to record from scratch)")
        return self

    @model_validator(mode="after")
    def _warn_on_unpriced_model(self) -> SimulationBrief:
        """A ceiling that cannot price the model cannot enforce anything.

        Not an error — a new model should still be usable the day it ships — but
        silently accepting `max_cost_usd` against a model we cannot price would
        be the worst of both worlds.
        """
        if prices_for(self.model) is not None:
            return self
        env_name = "SANZARU_REALTIME_PRICE_" + self.model.upper().replace("-", "_").replace(".", "_")
        if self.max_cost_usd is not None:
            logger.warning(
                "max_cost_usd=%.2f is set, but no price is known for model %r, so the ceiling "
                "will never fire - set %s to enable it",
                self.max_cost_usd,
                self.model,
                env_name,
            )
        else:
            logger.info("No price known for model %r; spend will be reported as unknown (set %s)", self.model, env_name)
        return self

    @model_validator(mode="after")
    def _check_explicit_session_cap_can_fit_an_act(self) -> SimulationBrief:
        """An act is indivisible: it needs one session per host, all at once.

        Only checked when the cap is set explicitly. The default is resolved
        against SANZARU_REALTIME_MAX_SESSIONS at run time, and a schema
        validator that second-guessed the environment would reject episodes
        that are perfectly runnable.
        """
        if self.rundown is None or self.max_concurrent_sessions <= 0:
            return self
        hosts = len(self.rundown.hosts)
        if hosts > self.max_concurrent_sessions:
            raise ValueError(
                f"max_concurrent_sessions={self.max_concurrent_sessions} cannot fit one act: "
                f"{hosts} hosts need {hosts} sessions at once"
            )
        return self


class CostReport(BaseModel):
    """What the run cost, or is projected to cost."""

    usd: float | None = None
    """None when the model has no known price — see audio/realtime/pricing.py."""
    usage: RealtimeUsage = Field(default_factory=RealtimeUsage)
    limit_usd: float | None = None
    unpriced_models: list[str] = Field(default_factory=list)
    estimated: bool = False


class SimulatedPodcastResult(BaseModel):
    """Result from simulate_podcast."""

    run_id: str
    title: str
    output_file: str = ""
    duration_seconds: float = 0.0
    turn_count: int = 0
    hosts: list[str] = Field(default_factory=list)
    acts: list[ActSummary] = Field(default_factory=list)
    cost: CostReport = Field(default_factory=CostReport)
    transcript: str = ""
    stems: dict[str, str] = Field(default_factory=dict)
    checkpoints: list[str] = Field(default_factory=list)
    qc: QCReport | None = None
    rundown: Rundown | None = None
    dry_run: bool = False
    resume_command: str = ""
    """How to pick this run back up if it was interrupted."""


# ---------- checkpoints ----------


class ActCheckpoint(BaseModel):
    """Sidecar written next to each act's audio.

    Holds everything the audio cannot: what was said, how long each turn ran, and
    what it cost. Without it a resumed run reports zero usage, has no transcript,
    and cannot rebuild per-speaker stems.
    """

    act_id: str
    title: str
    stop_reason: str
    usage: RealtimeUsage
    turns: list[Turn]


class RunManifest(BaseModel):
    """One file, named by run id alone, that makes `resume` self-sufficient."""

    run_id: str
    slug: str
    created: float
    rundown: Rundown
    brief: SimulationBrief


def checkpoint_storage() -> StorageBackend:
    """Where the run manifest and act checkpoints live — never the ``-o`` target.

    ``-o ./out/ep1.mp3`` repoints the entire "audio" path type at ``./out`` for
    that one invocation, but the resume hint the CLI prints (and the ``resume``
    field in the ceiling envelope) carries no ``-o``. Writing the bookkeeping
    under the override would put it somewhere ``simulate --resume <id>`` never
    looks, on exactly the runs that need to be resumed. It also keeps 2N+1
    intermediate files out of the user's output directory.

    Deliverables — the episode and the stems — still follow ``-o``; only what is
    addressed by run id alone is pinned here.
    """
    from ..config import is_path_configured
    from ..storage.local import LocalStorageBackend

    default = get_default_storage()
    if isinstance(default, LocalStorageBackend) and not is_path_configured("audio"):
        # No media dir at all: there is nowhere more durable than wherever this
        # run is already writing, so keep the checkpoints beside the episode
        # rather than failing a recording that would otherwise succeed. Say so —
        # resume then only works from the same working directory.
        active = get_storage()
        if active is not default:
            logger.warning(
                "No media directory configured (set SANZARU_MEDIA_PATH or AUDIO_PATH), so this run's "
                "checkpoints go to the output directory - resume will need the same -o"
            )
        return active
    return default


def _manifest_name(run_id: str) -> str:
    return f"simrun_{run_id}.json"


def _act_audio_name(slug: str, run_id: str, act_id: str) -> str:
    return f"{slug}_{run_id}_{act_id}.mp3"


def _act_meta_name(slug: str, run_id: str, act_id: str) -> str:
    return f"{slug}_{run_id}_{act_id}.json"


def new_run_id() -> str:
    """Short, sortable, and unique enough for one media directory.

    The CLI mints this *before* recording starts and prints it, so an interrupted
    run is always resumable.
    """
    return format(int(time.time() * 1000) % 0x100000000, "08x")


# ---------- planning ----------


def _max_sessions(brief: SimulationBrief) -> int:
    if brief.max_concurrent_sessions > 0:
        return brief.max_concurrent_sessions
    raw = os.getenv("SANZARU_REALTIME_MAX_SESSIONS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("SANZARU_REALTIME_MAX_SESSIONS=%r is not an integer - using %d", raw, DEFAULT_MAX_SESSIONS)
            return DEFAULT_MAX_SESSIONS
        if value > 0:
            return value
    return DEFAULT_MAX_SESSIONS


async def resolve_rundown(brief: SimulationBrief) -> Rundown:
    """Use the supplied rundown, or plan one from the premise."""
    if brief.rundown is not None:
        rundown = brief.rundown
        if not rundown.hosts:
            raise ValueError("rundown has no hosts")
        if not rundown.acts:
            raise ValueError("rundown has no acts")
        return rundown
    if not brief.premise.strip():
        raise ValueError("simulate_podcast needs either a rundown or a premise")
    return await plan_rundown(
        RundownRequest(
            premise=brief.premise,
            acts=brief.acts,
            target_minutes=brief.target_minutes,
            title=brief.title,
            style=brief.style,
            hosts=list(brief.hosts),
            turn_seconds=brief.turn_seconds,
            model=brief.planner_model,
        )
    )


UPCOMING_LOOKAHEAD = 2
"""How many later acts an act is warned off. Drift is overwhelmingly into the
*next* material, and naming all eleven remaining acts would bury the warning."""


def annotate_upcoming(rundown: Rundown) -> Rundown:
    """Tell each act what later acts own, unless the rundown already says.

    Derived rather than planned: it is exact by construction, it works for a
    hand-edited rundown that never went through the planner, and a human who
    writes their own `upcoming` still wins.
    """
    acts = list(rundown.acts)
    updated: list[ActBrief] = []
    for index, act in enumerate(acts):
        if act.upcoming or index == len(acts) - 1:
            updated.append(act)
            continue
        later = acts[index + 1 : index + 1 + UPCOMING_LOOKAHEAD]
        updated.append(act.model_copy(update={"upcoming": "; ".join(f"{a.title} — {a.topic}" for a in later)}))
    return rundown.model_copy(update={"acts": updated})


def project_run(rundown: Rundown, brief: SimulationBrief) -> CostReport:
    """Project usage and cost for a rundown without recording anything."""
    total = RealtimeUsage()
    for act in rundown.acts:
        total = total + project_usage(
            seconds=act.target_seconds,
            turns=act.max_turns,
            hosts=len(rundown.hosts),
        )
    cost = usage_cost(total, brief.model)
    return CostReport(
        usd=None if cost is None else round(cost, 4),
        usage=total,
        limit_usd=brief.max_cost_usd,
        unpriced_models=[] if cost is not None else [brief.model],
        estimated=True,
    )


# ---------- recording ----------


@dataclass(slots=True)
class _RecordedAct:
    """One act, however it got here."""

    brief: ActBrief
    result: ActResult
    mp3: bytes
    reused: bool = False
    checkpoint: str = ""
    turn_pcm: list[bytes] = field(default_factory=list)
    """Per-turn audio for stems. Sliced proportionally for resumed acts."""


async def _load_checkpoint(
    storage: StorageBackend,
    slug: str,
    run_id: str,
    brief_act: ActBrief,
) -> _RecordedAct | None:
    """Read one act back from disk, or None when it is not (fully) there."""
    audio_name = _act_audio_name(slug, run_id, brief_act.id)
    meta_name = _act_meta_name(slug, run_id, brief_act.id)
    if not (await storage.exists("audio", audio_name) and await storage.exists("audio", meta_name)):
        return None
    try:
        mp3 = await storage.read("audio", audio_name)
        meta = ActCheckpoint.model_validate_json((await storage.read("audio", meta_name)).decode())
        # Decoding belongs inside the guard, not after it. A truncated write —
        # LocalStorageBackend.write is a plain open+write, so a crash leaves one
        # reachable — and a zero-audio act both fail here rather than at the
        # read, and one bad act must not make every later --resume exit 1.
        pcm = await anyio.to_thread.run_sync(decode_to_pcm, mp3, "mp3")
        parts = await anyio.to_thread.run_sync(slice_pcm_by_durations, pcm, [t.seconds for t in meta.turns])
    except Exception as exc:  # noqa: BLE001 — a corrupt checkpoint just means re-record
        logger.warning("Checkpoint for %s unusable (%s) - re-recording", brief_act.id, exc)
        return None

    result = ActResult(
        act_id=meta.act_id,
        audio=[TurnAudio(turn=turn, pcm=part) for turn, part in zip(meta.turns, parts, strict=False)],
        usage=meta.usage,
        stop_reason=meta.stop_reason,
    )
    return _RecordedAct(
        brief=brief_act,
        result=result,
        mp3=mp3,
        reused=True,
        checkpoint=audio_name,
        turn_pcm=parts,
    )


async def _record_act(
    act: ActBrief,
    index: int,
    rundown: Rundown,
    brief: SimulationBrief,
    settings: SimulationSettings,
    *,
    checkpoints: FileSystemRepository,
    slug: str,
    run_id: str,
    budget: CostBudget,
    limiter: anyio.CapacityLimiter,
    on_progress: Callable[[str], None] | None,
) -> _RecordedAct:
    """Record one act and checkpoint it before returning."""

    def _note(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    async with limiter:
        _note(f"act {index + 1}/{len(rundown.acts)} recording ({act.title})")
        result = await run_act(
            act,
            rundown.hosts,
            settings,
            is_first_act=index == 0,
            is_last_act=index == len(rundown.acts) - 1,
            start_index=index % len(rundown.hosts),
            budget=budget,
            on_turn=lambda turn: _note(
                f"act {index + 1}/{len(rundown.acts)} turn {turn.index + 1} [{turn.speaker_name}] {turn.seconds:.1f}s"
            ),
        )

    turn_pcm = [ta.pcm for ta in result.audio]
    mp3 = await anyio.to_thread.run_sync(encode_pcm, result.join_pcm(), "mp3", brief.output_bitrate)

    audio_name = _act_audio_name(slug, run_id, act.id)
    await checkpoints.write_audio_file(audio_name, mp3)
    meta = ActCheckpoint(
        act_id=act.id,
        title=act.title,
        stop_reason=result.stop_reason,
        usage=result.usage,
        turns=result.turns,
    )
    await checkpoints.write_audio_file(_act_meta_name(slug, run_id, act.id), meta.model_dump_json(indent=2).encode())
    budget.mark_act_complete(act.id)

    _note(
        f"act {index + 1}/{len(rundown.acts)} recorded {len(result.audio)} turns, "
        f"{result.seconds:.0f}s audio ({result.stop_reason})"
    )
    return _RecordedAct(brief=act, result=result, mp3=mp3, checkpoint=audio_name, turn_pcm=turn_pcm)


async def _record_all(
    rundown: Rundown,
    brief: SimulationBrief,
    settings: SimulationSettings,
    *,
    checkpoints: FileSystemRepository,
    slug: str,
    run_id: str,
    budget: CostBudget,
    only: Sequence[str] | None = None,
    reuse: dict[str, _RecordedAct] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[_RecordedAct]:
    """Record every act that is not already available, in parallel.

    Args:
        only: When set, re-record exactly these act ids (the `--qc-retry` path)
            and reuse everything else.
        reuse: Already-available acts, keyed by act id.
    """
    reuse = dict(reuse or {})
    todo = [
        (index, act)
        for index, act in enumerate(rundown.acts)
        if (act.id in only if only is not None else act.id not in reuse)
    ]

    sessions = _max_sessions(brief)
    hosts = max(1, len(rundown.hosts))
    # An act is indivisible — it opens one session per host — so a cap below
    # that is exceeded rather than honoured. Say so instead of quietly going over.
    if hosts > sessions:
        logger.warning(
            "%d hosts need %d concurrent sessions but the cap is %d; running one act at a time "
            "and exceeding the cap (raise SANZARU_REALTIME_MAX_SESSIONS)",
            hosts,
            hosts,
            sessions,
        )
    parallel_acts = max(1, sessions // hosts)
    if todo:
        logger.info(
            "Recording %d act(s) with up to %d in parallel (%d sessions / %d hosts)",
            len(todo),
            parallel_acts,
            sessions,
            len(rundown.hosts),
        )
    limiter = anyio.CapacityLimiter(parallel_acts)

    async def _one(index: int, act: ActBrief) -> _RecordedAct:
        return await _record_act(
            act,
            index,
            rundown,
            brief,
            settings,
            checkpoints=checkpoints,
            slug=slug,
            run_id=run_id,
            budget=budget,
            limiter=limiter,
            on_progress=on_progress,
        )

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _one, index, act) for index, act in todo]
    # Read by index — the limiter delays task entry, it never reorders results.
    for capture in captures:
        recorded = capture.result()
        reuse[recorded.result.act_id] = recorded

    missing = [act.id for act in rundown.acts if act.id not in reuse]
    if missing:  # pragma: no cover - defensive; every act is either reused or recorded
        raise RuntimeError(f"acts never produced audio: {', '.join(missing)}")
    return [reuse[act.id] for act in rundown.acts]


# ---------- assembly ----------


def _build_timeline(recorded: Sequence[_RecordedAct], act_gap_ms: int) -> list[TimelineItem]:
    """Speaker-attributed timeline for the whole episode, for stems."""
    from ..audio.realtime.types import pcm_silence

    gap = pcm_silence(act_gap_ms)
    timeline: list[TimelineItem] = []
    for index, act in enumerate(recorded):
        for turn_audio, pcm in zip(act.result.audio, act.turn_pcm, strict=False):
            timeline.append((turn_audio.turn.speaker_id, pcm))
        if gap and index < len(recorded) - 1:
            timeline.append((None, gap))
    return timeline


def _transcript(recorded: Sequence[_RecordedAct]) -> str:
    blocks: list[str] = []
    for act in recorded:
        blocks.append(f"## {act.brief.title}")
        blocks.extend(f"**{turn.speaker_name}:** {turn.text}" for turn in act.result.turns if turn.text)
    return "\n\n".join(blocks)


def _summaries(recorded: Sequence[_RecordedAct]) -> list[ActSummary]:
    return [
        ActSummary(
            act_id=act.result.act_id,
            title=act.brief.title,
            turns=len(act.result.audio),
            seconds=round(act.result.seconds, 1),
            stop_reason=act.result.stop_reason,
            truncated_turns=sum(1 for t in act.result.turns if t.truncated),
            usage=act.result.usage,
            reused=act.reused,
        )
        for act in recorded
    ]


async def _write_stems(
    recorded: Sequence[_RecordedAct],
    rundown: Rundown,
    brief: SimulationBrief,
    *,
    repo: FileSystemRepository,
    slug: str,
    run_id: str,
) -> dict[str, str]:
    """One time-aligned track per host, rendered and written one at a time.

    Sequential on purpose: each stem is a full-length copy of the episode in raw
    PCM, and holding all of them at once is what would actually hurt.
    """
    timeline = _build_timeline(recorded, brief.act_gap_ms)
    stems: dict[str, str] = {}
    for host in rundown.hosts:
        data = await anyio.to_thread.run_sync(render_stem, timeline, host.id, brief.output_format, brief.output_bitrate)
        name = f"{slug}_{run_id}_stem_{host.id}.{brief.output_format}"
        await repo.write_audio_file(name, data)
        stems[host.id] = name
    return stems


# ---------- entry point ----------


async def simulate_podcast(
    brief: SimulationBrief,
    on_progress: Callable[[str], None] | None = None,
) -> SimulatedPodcastResult:
    """Record a podcast by having realtime agents converse, then mix it down.

    Args:
        brief: The episode request — either a full `rundown` or a `premise` to
            plan one from, plus model, budget, and output settings.
        on_progress: Called with one-line status updates as acts and turns land.
            The CLI routes these to stderr; MCP callers can ignore them.

    Returns:
        The finished episode, its per-act summaries, the transcript the models
        reported speaking, and what it cost.

    Raises:
        ValueError: If the brief is unusable (no premise and no rundown).
        CostCeilingError: If `max_cost_usd` is crossed mid-run. Acts that
            finished are already checkpointed and named in the exception.
    """
    started = time.monotonic()
    # Deliverables follow whatever storage this invocation configured (`-o`);
    # bookkeeping does not. See `checkpoint_storage`.
    repo = FileSystemRepository()
    ckpt_storage = checkpoint_storage()
    checkpoints = FileSystemRepository(ckpt_storage)

    run_id = brief.run_id or new_run_id()
    rundown: Rundown | None = None
    effective = brief

    # `resume <run_id>` alone must be enough: the manifest carries the rundown
    # and the settings the run started with.
    if brief.resume:
        manifest_name = _manifest_name(run_id)
        if await ckpt_storage.exists("audio", manifest_name):
            manifest = RunManifest.model_validate_json((await ckpt_storage.read("audio", manifest_name)).decode())
            rundown = manifest.rundown
            # One rule, both directions: a field the caller actually passed this
            # time wins, everything else is restored from the manifest.
            #
            # Enumerating the caller's fields by hand got both halves wrong.
            # `max_cost_usd` was taken unconditionally, so following the ceiling
            # abort's own resume hint — which carries no flags — re-ran the job
            # uncapped, disabling the exact safety that fired. `output_format`,
            # `output_bitrate` and `act_gap_ms` were dropped unconditionally, so
            # passing them on resume did nothing. `model_fields_set` is the only
            # honest signal of caller intent, which is why the CLI leaves flags
            # the user did not type out of the payload entirely.
            restored = {name: getattr(brief, name) for name in brief.model_fields_set}
            # The manifest owns the rundown; its stored brief keeps `rundown`
            # empty so it stays a valid standalone description of this run.
            restored.update({"resume": True, "run_id": run_id, "rundown": None})
            effective = manifest.brief.model_copy(update=restored)
            logger.info("Resuming run %s: %s", run_id, rundown.title)
        elif brief.rundown is None:
            raise ValueError(
                f"no run manifest for run_id {run_id!r} — pass the rundown explicitly, or check the audio directory"
            )

    if rundown is None:
        rundown = await resolve_rundown(effective)
    rundown = annotate_upcoming(rundown)

    slug = _safe_title(rundown.title)
    settings = SimulationSettings(
        model=effective.model,
        show_title=rundown.title,
        premise=rundown.premise,
        style=rundown.style,
        max_turn_tokens=effective.turn_tokens,
        turn_seconds=effective.turn_seconds,
        turn_timeout_s=effective.turn_timeout_s,
    )
    resume_command = f"sanzaru podcast simulate --resume {run_id}"

    if effective.dry_run:
        projection = project_run(rundown, effective)
        return SimulatedPodcastResult(
            run_id=run_id,
            title=rundown.title,
            duration_seconds=round(rundown.total_target_seconds(), 1),
            turn_count=rundown.total_max_turns(),
            hosts=[h.name for h in rundown.hosts],
            acts=[
                ActSummary(
                    act_id=act.id,
                    title=act.title,
                    turns=act.max_turns,
                    seconds=act.target_seconds,
                    stop_reason="projected",
                )
                for act in rundown.acts
            ],
            cost=projection,
            rundown=rundown,
            dry_run=True,
            resume_command=resume_command,
        )

    # Written before any recording starts so an interrupted first act is still
    # resumable.
    await checkpoints.write_audio_file(
        _manifest_name(run_id),
        RunManifest(
            run_id=run_id,
            slug=slug,
            created=time.time(),
            rundown=rundown,
            # The rundown is stripped so the manifest has one source of truth,
            # and `resume` is set so what remains is still a valid brief on its
            # own — it describes exactly the invocation that picks this run up.
            brief=effective.model_copy(update={"run_id": run_id, "rundown": None, "resume": True}),
        )
        .model_dump_json(indent=2)
        .encode(),
    )

    reuse: dict[str, _RecordedAct] = {}
    if effective.resume:
        for act in rundown.acts:
            loaded = await _load_checkpoint(ckpt_storage, slug, run_id, act)
            if loaded is not None:
                reuse[act.id] = loaded
        if reuse and on_progress is not None:
            on_progress(f"resuming run {run_id}: reusing {len(reuse)}/{len(rundown.acts)} acts")

    budget = CostBudget(effective.max_cost_usd)
    for existing in reuse.values():
        # Marked before charging: a reused act's checkpoint is already on disk,
        # so it counts as safe even if replaying its spend is what trips the
        # ceiling on this very line.
        budget.mark_act_complete(existing.result.act_id)
        budget.charge(existing.result.usage, effective.model)

    recorded = await _record_all(
        rundown,
        effective,
        settings,
        checkpoints=checkpoints,
        slug=slug,
        run_id=run_id,
        budget=budget,
        reuse=reuse,
        on_progress=on_progress,
    )

    qc_report: QCReport | None = None
    if effective.qc:
        if on_progress is not None:
            on_progress(f"qc: transcribing {len(recorded)} acts with {effective.transcribe_model}")
        qc_report = await run_qc(
            rundown,
            {a.result.act_id: a.mp3 for a in recorded},
            {a.result.act_id: a.result.turns for a in recorded},
            transcribe_model=effective.transcribe_model,
            judge_model=effective.judge_model,
            limiter=anyio.CapacityLimiter(4),
        )
        if effective.qc_retry and qc_report.flagged_acts:
            if on_progress is not None:
                on_progress(f"qc flagged {', '.join(qc_report.flagged_acts)} - re-recording once")
            recorded = await _record_all(
                rundown,
                effective,
                settings,
                checkpoints=checkpoints,
                slug=slug,
                run_id=run_id,
                budget=budget,
                only=qc_report.flagged_acts,
                reuse={a.result.act_id: a for a in recorded},
                on_progress=on_progress,
            )
            qc_report = await run_qc(
                rundown,
                {a.result.act_id: a.mp3 for a in recorded},
                {a.result.act_id: a.result.turns for a in recorded},
                transcribe_model=effective.transcribe_model,
                judge_model=effective.judge_model,
                limiter=anyio.CapacityLimiter(4),
            )

    if on_progress is not None:
        on_progress("stitching episode")
    act_pcm = [a.result.join_pcm() for a in recorded]
    gaps = [effective.act_gap_ms] * (len(act_pcm) - 1) + [0]
    final_audio = await anyio.to_thread.run_sync(
        lambda: _stitch_audio(
            segment_bytes_list=act_pcm,
            pause_ms_list=gaps,
            intro_ms=effective.intro_silence_ms,
            outro_ms=effective.outro_silence_ms,
            normalize_loudness=effective.normalize_loudness,
            output_format=effective.output_format,
            output_bitrate=effective.output_bitrate,
            decode=pcm_to_segment,
        )
    )

    output_filename = effective.filename or f"{slug}_{run_id}.{effective.output_format}"
    await repo.write_audio_file(output_filename, final_audio)
    logger.info("Simulated podcast written: %s (%s bytes)", output_filename, f"{len(final_audio):,}")

    stems: dict[str, str] = {}
    if effective.stems:
        if on_progress is not None:
            on_progress(f"rendering {len(rundown.hosts)} stems")
        stems = await _write_stems(recorded, rundown, effective, repo=repo, slug=slug, run_id=run_id)

    total_usage = RealtimeUsage()
    for recorded_act in recorded:
        total_usage = total_usage + recorded_act.result.usage
    spent = usage_cost(total_usage, effective.model)

    logger.info(
        "Simulation complete: %d acts, %d turns, %.0fs audio in %.0fs wall clock",
        len(recorded),
        sum(len(a.result.audio) for a in recorded),
        sum(a.result.seconds for a in recorded),
        time.monotonic() - started,
    )

    return SimulatedPodcastResult(
        run_id=run_id,
        title=rundown.title,
        output_file=output_filename,
        duration_seconds=round(sum(a.result.seconds for a in recorded), 1),
        turn_count=sum(len(a.result.audio) for a in recorded),
        hosts=[h.name for h in rundown.hosts],
        acts=_summaries(recorded),
        cost=CostReport(
            usd=None if spent is None else round(spent, 4),
            usage=total_usage,
            limit_usd=effective.max_cost_usd,
            unpriced_models=budget.unpriced_models,
        ),
        transcript=_transcript(recorded),
        stems=stems,
        checkpoints=[a.checkpoint for a in recorded if a.checkpoint],
        qc=qc_report,
        rundown=rundown,
        resume_command=resume_command,
    )


__all__ = [
    "ActCheckpoint",
    "CostReport",
    "RunManifest",
    "SimulatedPodcastResult",
    "SimulationBrief",
    "annotate_upcoming",
    "new_run_id",
    "project_run",
    "resolve_rundown",
    "simulate_podcast",
]
