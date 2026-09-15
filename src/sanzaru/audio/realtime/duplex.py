# SPDX-License-Identifier: MIT
"""Duplex recording: every Live host hears every other host, all act long.

The cued loop (`producer.run_act`) is a half-duplex protocol laid over a
full-duplex model, and a real recording showed the seam three ways: a steering
note read aloud, a host restarting from its last line because the cue arrived
before it had absorbed the exchange, and a false start on most turns as the
model began reacting to what it heard and then restarted on the cue. Here the
seam is removed rather than patched: the hosts' sessions run for the whole act,
each host's output frames are fed into every other host's ears as they arrive,
and the models negotiate turn-taking natively — which is what the model is for.

The producer still has a job, just not the floor. It steers on the *act clock*
with `session.thinking.append` (verified silent — see `LiveAgent.think`):
who opens, which talking point comes next and who raises it, the caller's
`turn_notes`, when to start landing the close, and — if two hosts keep talking
over each other — who should yield. The act ends once the close has landed and
everyone has gone quiet, or at a hard cap.

The act's audio is the time-aligned mix of the hosts' streams; the transcript
comes from each host's timestamped `session.output_transcript.delta` fragments,
grouped into utterances and ordered by start time, so QC, summaries and
checkpoints see ordinary `Turn`s. Per-host streams are kept as stems.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anyio

from ...config import logger
from ...exceptions import RealtimeAPIError

# Imported for types and helpers only; `producer` imports this module lazily
# inside `run_act`, so there is no cycle at import time.
from . import producer as _producer
from .budget import CostBudget
from .live_agent import LiveAgent, mix_pcm16
from .types import ActBrief, ActResult, HostSpec, Turn, TurnAudio, pcm_seconds

CLOSE_AT_FRACTION = 0.85
"""When, as a share of `target_seconds`, the closer is told to start landing."""

WRAP_AT_FRACTION = 1.0
"""When everyone is told the segment is over, if it has not landed by itself."""

HARD_CAP_FACTOR = 1.5
HARD_CAP_GRACE_S = 30.0
"""A duplex act is cut at `target * HARD_CAP_FACTOR + HARD_CAP_GRACE_S` (or the
act wall budget, whichever is sooner). Nothing else bounds two models that
keep finding something to say."""

END_SILENCE_FACTOR = 2.0
"""The act ends once everyone has been quiet for this many end-of-turn silence
gaps after the close was cued. A close has a beat in it; one gap is a breath."""

COLLISION_S = 1.5
"""Sustained simultaneous speech that earns the interrupter a yield note."""

COLLISION_COOLDOWN_S = 6.0
STALL_S = 5.0
"""Everyone silent this long mid-act gets the next host a nudge."""

STALL_COOLDOWN_S = 8.0
UTTERANCE_GAP_S = 1.2
"""Transcript fragments from one host closer together than this are one turn."""

TICK_S = 0.1

NOTE_SEND_TIMEOUT_S = 5.0
"""Bound on sending one producer note. A stalled `thinking.append` must not
hold the loop: the hard cap, the health check and the budget charge all live
there. A lost note is logged and the act goes on without it."""


def build_duplex_rules(host: HostSpec, others: Sequence[HostSpec], opener: HostSpec, turn_seconds: float) -> str:
    """The turn-taking contract for a duplex table, appended to the act instructions."""
    names = ", ".join(other.name for other in others) or "your co-host"
    lines = [
        "",
        "LIVE CONVERSATION (you hear each other continuously, like a phone call):",
        f"  - {opener.name} opens the segment. Everyone else: listen first, then respond.",
        f"  - Take turns naturally. Keep each turn around {turn_seconds:.0f} seconds, then stop and let "
        f"{names} answer.",
        "  - Never talk over someone mid-sentence. If you both start at once, stop and let them go.",
        "  - Short acknowledgements ('right', 'mm') are fine; do not narrate or fill every pause.",
        "  - Silence while the other person thinks is normal. Wait for it.",
        "  - Never repeat a point you have already made. If you have said it, move on.",
        "  - You will receive silent producer notes. Follow them; never read them aloud, never mention "
        "them, never say you are waiting for one.",
    ]
    if host.id == opener.id:
        lines.append("  - You open. Begin as soon as the segment starts.")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ScheduledNote:
    """A silent producer note due at `at_s` on the act clock."""

    at_s: float
    host_ids: tuple[str, ...]
    text: str
    label: str
    kind: str = "note"
    """`open`, `point`, `turn_note`, `close`, `wrap`, or `note`."""


def plan_notes(
    brief: ActBrief,
    hosts: Sequence[HostSpec],
    order: Sequence[int],
    *,
    is_first_act: bool,
    is_last_act: bool,
) -> list[ScheduledNote]:
    """Lay the producer's plan out on the act clock.

    The cued loop's per-turn schedule maps onto time: turn `i` of `max_turns`
    lands at `target * i / max_turns`. Talking points are spread across the act
    and each is handed to a rotating host to raise, the caller's `turn_notes`
    ride on their turn's time to their turn's host, and the close goes to the
    host who would have taken the last planned turn.
    """
    target = brief.target_seconds
    everyone = tuple(host.id for host in hosts)
    opener = hosts[order[0]]
    closer = hosts[order[(brief.max_turns - 1) % len(order)]]
    notes: list[ScheduledNote] = []

    opening = _producer._note_for_turn(
        brief, 0, is_final_turn=False, is_first_act=is_first_act, is_last_act=is_last_act, point_index=None
    ).text
    notes.append(ScheduledNote(0.0, (opener.id,), f"You open the segment now. {opening or ''}".strip(), "open", "open"))
    listeners = tuple(host.id for host in hosts if host.id != opener.id)
    if listeners:
        notes.append(
            ScheduledNote(
                0.0,
                listeners,
                f"{opener.name} opens the segment. Listen, then respond to what they actually said.",
                "open (listeners)",
                "open",
            )
        )

    points = brief.talking_points
    if len(points) > 1:
        # The first point is what the opener puts on the table; the rest are
        # spread over the stretch before the close is cued.
        span = target * CLOSE_AT_FRACTION
        for k, point in enumerate(points[1:], start=1):
            bringer = hosts[order[k % len(order)]]
            at = span * k / len(points)
            notes.append(
                ScheduledNote(
                    at,
                    everyone,
                    f"Next, when there is a natural opening: {point} — {bringer.name} raises it; the others react.",
                    f"point {k + 1}",
                    "point",
                )
            )

    by_index = brief.turn_notes.get(brief.max_turns - 1)
    index_is_takeover = not brief.closing_note and by_index is not None
    for index, text in sorted(brief.turn_notes.items()):
        if not text or (index_is_takeover and index == brief.max_turns - 1):
            continue
        at = min(target * index / max(1, brief.max_turns), target * CLOSE_AT_FRACTION - 1.0)
        host = hosts[order[index % len(order)]]
        notes.append(ScheduledNote(max(0.0, at), (host.id,), text, f"turn_note {index}", "turn_note"))

    closing = brief.closing_note or (by_index if index_is_takeover else None)
    closing_text = closing or _producer._closing_note(brief, is_last_act=is_last_act)
    close_at = target * CLOSE_AT_FRACTION
    notes.append(
        ScheduledNote(close_at, (closer.id,), f"Start landing the segment now: {closing_text}", "close", "close")
    )
    others = tuple(host.id for host in hosts if host.id != closer.id)
    if others:
        notes.append(
            ScheduledNote(
                close_at,
                others,
                f"{closer.name} is landing the segment. One short reply at most, then let it end. "
                "Do not open a new topic.",
                "close (others)",
                "close",
            )
        )
    wrap_text = (
        "The segment is over. Finish your sentence and stop talking."
        if not is_last_act
        else "The episode is over. Finish your sentence, say goodbye once if you have not, and stop talking."
    )
    notes.append(ScheduledNote(target * WRAP_AT_FRACTION, everyone, wrap_text, "wrap", "wrap"))
    return sorted(notes, key=lambda note: note.at_s)


def group_utterances(
    fragments: dict[str, list[tuple[int, int, str]]],
    offsets_s: dict[str, float],
    gap_s: float = UTTERANCE_GAP_S,
) -> list[tuple[str, float, float, str]]:
    """Turn per-host transcript fragments into (host_id, start_s, end_s, text), ordered by start.

    `fragments` are `(start_ms, end_ms, text)` on each host's *session*
    timeline; `offsets_s[host]` moves them onto the act clock (session zero
    minus act zero, usually negative). Fragments from one host separated by
    less than `gap_s` are one utterance.
    """
    utterances: list[tuple[str, float, float, str]] = []
    for host_id, items in fragments.items():
        offset = offsets_s.get(host_id, 0.0)
        current: list[str] = []
        start = end = 0.0
        for start_ms, end_ms, text in sorted(items, key=lambda item: item[0]):
            frag_start = start_ms / 1000.0 + offset
            frag_end = max(frag_start, end_ms / 1000.0 + offset)
            if current and frag_start - end > gap_s:
                utterances.append((host_id, start, end, "".join(current).strip()))
                current = []
            if not current:
                start = frag_start
            current.append(text)
            end = frag_end if len(current) == 1 else max(end, frag_end)
        if current:
            utterances.append((host_id, start, end, "".join(current).strip()))
    return sorted((u for u in utterances if u[3]), key=lambda u: (u[1], u[0]))


def _pad_to(streams: dict[str, bytes], length: int) -> dict[str, bytes]:
    return {host: pcm[:length].ljust(length, b"\x00") for host, pcm in streams.items()}


async def run_duplex_act(
    brief: ActBrief,
    hosts: Sequence[HostSpec],
    settings: _producer.SimulationSettings,
    *,
    is_first_act: bool = False,
    is_last_act: bool = False,
    start_index: int = 0,
    connect: _producer.ConnectFactory | None = None,
    budget: CostBudget | None = None,
    on_turn: Callable[[Turn], None] | None = None,
) -> ActResult:
    """Record one act with every host live to every other host.

    Same signature as `producer.run_act`, which dispatches here for an
    all-Live table in duplex mode. Returns an `ActResult` carrying `mixed_pcm`,
    `stems`, `collision_seconds`, and turns rebuilt from the transcripts.
    """
    if len(hosts) < 2:
        raise ValueError(f"act {brief.id!r}: duplex recording needs at least two hosts")
    connect_fn = connect or _producer._default_connect
    order = _producer._resolve_order(brief, hosts, start_index)
    opener = hosts[order[0]]
    closer_id = hosts[order[(brief.max_turns - 1) % len(order)]].id
    silence_s = settings.live_turn_silence_s
    end_silence = silence_s * END_SILENCE_FACTOR
    wall_budget = _producer.act_wall_budget_seconds(settings.act_budget_s)
    hard_cap = min(wall_budget, brief.target_seconds * HARD_CAP_FACTOR + HARD_CAP_GRACE_S)
    result = ActResult(act_id=brief.id)

    async with contextlib.AsyncExitStack() as stack:
        agents: list[LiveAgent] = []
        for host in hosts:
            model = host.model or settings.model
            connection = await stack.enter_async_context(connect_fn(model))
            agent = LiveAgent(
                host,
                connection,  # type: ignore[arg-type]  # all-Live table: run_act checked every model
                model=model,
                turn_seconds=settings.turn_seconds,
                sample_rate=settings.sample_rate,
                end_of_turn_silence_s=silence_s,
                mode="duplex",
            )
            await stack.enter_async_context(agent)
            agents.append(agent)
        by_id = {agent.id: agent for agent in agents}

        async def _configure(agent: LiveAgent) -> None:
            others = [h for h in hosts if h.id != agent.id]
            base = _producer.build_instructions(
                brief, agent.spec, others, settings, is_first_act=is_first_act, is_last_act=is_last_act
            )
            await agent.configure(
                base + build_duplex_rules(agent.spec, others, opener, settings.turn_seconds), start_clock=False
            )

        # All sessions come up together, and nobody's timeline moves until
        # everyone is up, wired to everyone else, and recording. Configured one
        # by one with clocks running, the opener spoke while a slower peer was
        # still starting — frames that nobody heard and nobody recorded.
        async with anyio.create_task_group() as setup:
            for agent in agents:
                setup.start_soon(_configure, agent)
        for agent in agents:
            agent.set_listeners(agents, always=True)
        for agent in agents:
            agent.start_recording()
        t0 = time.monotonic()
        for agent in agents:
            agent.start_clock()

        async def _note(agent: LiveAgent, text: str, label: str) -> None:
            """Send a producer note, bounded so a stalled send cannot hold the loop."""
            remaining = hard_cap - (time.monotonic() - t0)
            bound = max(TICK_S, min(NOTE_SEND_TIMEOUT_S, remaining))
            try:
                with anyio.fail_after(bound):
                    await agent.think(text, label=label)
            except TimeoutError:
                logger.warning(
                    "%s: producer note %r to %s did not send within %.1fs - dropped", brief.id, label, agent.name, bound
                )

        notes = plan_notes(brief, hosts, order, is_first_act=is_first_act, is_last_act=is_last_act)
        pending = list(notes)
        close_sent_at: float | None = None
        wrap_sent_at: float | None = None
        overlap_since: float | None = None
        last_collision_note = -1e9
        last_stall_note = -1e9
        last_charge = t0
        stall_rotation = 1
        current_point = brief.talking_points[0] if brief.talking_points else brief.topic
        stop_reason = "target_seconds"

        try:
            while True:
                now = time.monotonic()
                elapsed = now - t0
                # A dead session must never read as a quiet one: "everyone went
                # silent after the close" is also what two dropped connections
                # look like, and a finished act with a hole in it would then be
                # checkpointed as complete. Same rule as the cued loop's
                # `speak()` requiring a real end of turn.
                for agent in agents:
                    agent.check_alive()

                while pending and pending[0].at_s <= elapsed:
                    note = pending.pop(0)
                    for host_id in note.host_ids:
                        await _note(by_id[host_id], note.text, note.label)
                    if note.kind == "point":
                        current_point = note.text
                    elif note.kind == "close":
                        close_sent_at = close_sent_at or now
                    elif note.kind == "wrap":
                        wrap_sent_at = now
                    logger.debug("%s: producer note at %.0fs -> %s: %s", brief.id, elapsed, note.host_ids, note.label)

                speaking = [agent for agent in agents if agent.speaking]
                if len(speaking) >= 2:
                    result.collision_seconds += TICK_S
                    overlap_since = overlap_since or now
                    if now - overlap_since >= COLLISION_S and now - last_collision_note >= COLLISION_COOLDOWN_S:
                        interrupter = max(speaking, key=lambda a: a.speech_onset_at or 0.0)
                        other = next(a for a in speaking if a is not interrupter)
                        await _note(
                            interrupter,
                            f"You are talking over {other.name}. Stop now, let them finish, then respond.",
                            "yield",
                        )
                        last_collision_note = now
                        logger.info(
                            "%s: %s talked over %s for %.1fs - asked to yield",
                            brief.id,
                            interrupter.name,
                            other.name,
                            now - overlap_since,
                        )
                else:
                    overlap_since = None

                last_voice = max((a.last_speech_at or 0.0) for a in agents)
                anyone_spoke = any(a.ever_spoke for a in agents)
                quiet_for = (now - last_voice) if anyone_spoke else 0.0

                if close_sent_at is not None:
                    closer_landed = (by_id[closer_id].last_speech_at or 0.0) >= close_sent_at
                    if anyone_spoke and quiet_for >= end_silence and (closer_landed or wrap_sent_at is not None):
                        for agent in agents:
                            agent.check_alive()
                        stop_reason = "target_seconds"
                        break
                elif anyone_spoke and quiet_for >= STALL_S and now - last_stall_note >= STALL_COOLDOWN_S:
                    nudged = agents[order[stall_rotation % len(order)]]
                    stall_rotation += 1
                    await _note(
                        nudged,
                        f"The conversation has stalled. Pick it up now with a fresh thought on: {current_point}",
                        "stall nudge",
                    )
                    last_stall_note = now

                if elapsed >= hard_cap:
                    stop_reason = "wall_clock"
                    logger.warning(
                        "%s: duplex act cut at %.0fs (target %.0fs) - the hosts did not land the close",
                        brief.id,
                        elapsed,
                        brief.target_seconds,
                    )
                    break

                if budget is not None and now - last_charge >= 15.0:
                    for agent in agents:
                        spent = agent.take_usage()
                        result.add_usage(agent.model, spent)
                        budget.charge(spent, agent.model)
                    last_charge = now

                await anyio.sleep(TICK_S)
        except TimeoutError as exc:  # pragma: no cover - defensive
            raise RealtimeAPIError(f"{brief.id}: duplex act timed out") from exc

        for agent in agents:
            agent.stop_recording()
        for agent in agents:
            tail = await agent.finish()
            result.add_usage(agent.model, tail)
            if budget is not None:
                budget.charge(tail, agent.model)

        streams = {agent.id: agent.stream_pcm for agent in agents}
        length = max((len(pcm) for pcm in streams.values()), default=0)
        length -= length % 2
        result.stems = _pad_to(streams, length)
        result.mixed_pcm = mix_pcm16(list(result.stems.values())) if length else b""
        result.stop_reason = stop_reason
        result.collision_seconds = round(result.collision_seconds, 1)

        offsets = {agent.id: ((agent.clock_started_at or t0) - t0) for agent in agents}
        fragments = {agent.id: list(agent.transcript_deltas) for agent in agents}
        for index, (host_id, start, end, text) in enumerate(group_utterances(fragments, offsets)):
            turn = Turn(
                act_id=brief.id,
                index=index,
                speaker_id=host_id,
                speaker_name=by_id[host_id].name,
                text=text,
                seconds=round(max(0.0, end - start), 2),
            )
            result.audio.append(TurnAudio(turn=turn, pcm=b""))
            if on_turn is not None:
                on_turn(turn)

        logger.info(
            "%s: duplex act %.0fs of audio, %d utterances, %.1fs of overlap (%s)",
            brief.id,
            pcm_seconds(result.mixed_pcm or b""),
            len(result.audio),
            result.collision_seconds,
            stop_reason,
        )
        if not any(agent.ever_spoke for agent in agents):
            raise RealtimeAPIError(f"{brief.id}: no host spoke during the duplex act")

    return result


__all__ = [
    "ScheduledNote",
    "build_duplex_rules",
    "group_utterances",
    "plan_notes",
    "run_duplex_act",
]
