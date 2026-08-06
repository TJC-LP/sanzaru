# SPDX-License-Identifier: MIT
"""The producer: who talks next, and what they get nudged toward.

This is the quality-critical module. A spike with no producer — agents simply
alternating on a topic brief — drifted immediately: turns ran 30 seconds against
an instruction that said "two or three sentences", and the conversation slid into
mutual agreement. The same agents with a hard length rule, a `max_output_tokens`
backstop, and a per-turn steering note held 9-17 second turns and stayed on brief.

So the producer does three jobs:

1. **Floor control.** Exactly one agent is asked to respond at a time; its audio
   is then played into every other agent's ears. Nobody talks over anybody.
2. **Coverage.** Talking points are walked deliberately across the act rather
   than left to chance, so a brief with five points does not produce an act
   about the first one.
3. **Landing.** The last turns are steered toward a conclusion and, for a
   non-final act, toward the handoff the next act was briefed to pick up.

Steering notes are system messages, invisible to the audience.

All three jobs have *defaults*, not fixed behaviour. The caller of this tool is
usually itself an agent, and it is a better producer than a set of f-strings —
it knows which talking point deserves dwelling on and when one host should
follow their own thought. It cannot sit in this loop live (that would cost a
round-trip per turn and break parallel act recording), so it directs
declaratively instead, via `ActBrief.direction`, `turn_notes` and
`speaking_order`. Absent those, everything below applies.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio

from ...config import logger
from ...exceptions import RealtimeAPIError
from .agent import RealtimeAgent
from .budget import CostBudget
from .pricing import OUTPUT_TOKENS_PER_SECOND
from .types import (
    DEFAULT_TURN_SECONDS,
    MAX_ACT_TURNS,
    REALTIME_SAMPLE_RATE,
    REALTIME_VOICES,
    ActBrief,
    ActResult,
    HostSpec,
    Turn,
    TurnAudio,
)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from openai.resources.realtime.realtime import AsyncRealtimeConnection

    ConnectFactory = Callable[[str], AbstractAsyncContextManager[AsyncRealtimeConnection]]


DEFAULT_REALTIME_MODEL = "gpt-realtime-2.1"

DEFAULT_VOICES: tuple[str, ...] = REALTIME_VOICES
"""Alias, kept because this is the name the assignment code reads.

It must stay the same object as `REALTIME_VOICES`: `HostSpec` validates an
unknown voice against that tuple while `assign_voices` draws from this one, so
two hand-maintained copies would eventually warn about a voice we ourselves
assigned."""

TURN_TOKEN_HEADROOM = 1.5
"""How far past the target a turn may run before the token cap stops it.

Tuned against live recordings: at 1.0 the cap did the work the prompt should do
and clipped 17 of 29 turns mid-sentence; at 2.0 nothing clipped but turns
stretched to ~22s against a 15s rule. 1.5 leaves the prompt in charge while
still bounding a runaway."""

DEFAULT_TURN_TOKENS = 0
"""0 → derive from `turn_seconds` (see `turn_token_cap`)."""

TURN_EXTENSION_FACTOR = 1.5
"""How far past `max_turns` an act may extend to fill its `target_seconds`.

`max_turns` was a hard stop, which made act length a function of how long the
model's turns happen to run: gpt-realtime's full-size turns measured ~10s
against the mini's ~17.5s under the same 15s rule, so full-model acts stopped
a third short of their target. The turn budget stays what the planner priced —
extension turns only happen inside the time the act was already budgeted for,
and the absolute ceiling (`MAX_ACT_TURNS`) still binds."""

EXTENSION_NOTE = (
    "Take this somewhere new: one concrete example, implication, or sharper objection. "
    "Do not summarize, do not repeat anything already said, and do not start wrapping up."
)
"""Default steering for turns past the planned budget. An act that outlives its
talking points drifts into recap loops — the audible form is two hosts trading
restatements and premature goodbyes — so extension turns get pushed toward new
ground instead of left to run on momentum."""


def turn_token_cap(turn_seconds: float) -> int:
    """Per-turn `max_output_tokens`: a runaway backstop, not a target.

    A fixed 520 — 26 seconds at the audio rate alone — truncated 17 of 29 turns
    in a live recording, because `max_output_tokens` counts the *transcript*
    alongside the audio. Budgeting both rates and doubling for headroom means the
    cap only fires on a genuine monologue, and the prompt's length rule stays the
    thing that actually shapes turn length.
    """
    return max(256, int(turn_seconds * OUTPUT_TOKENS_PER_SECOND * TURN_TOKEN_HEADROOM))


TURN_TIMEOUT_FACTOR = 6.0
TURN_TIMEOUT_FLOOR = 60.0
TURN_TIMEOUT_ENV = "SANZARU_REALTIME_TURN_TIMEOUT"


def turn_timeout_seconds(turn_seconds: float, explicit: float = 0.0) -> float:
    """Wall-clock bound for one turn: generous, but finite.

    A realtime session that stops answering has nothing else to stop it — it
    holds a `CapacityLimiter` slot forever, inside a blocking MCP tool, in a
    server with no job registry. That is the one failure the act checkpoints
    were advertised as surviving and did not.

    A turn is generated at roughly speaking rate plus a couple of round-trips,
    so 6x the target (never under a minute) only fires on a genuine stall.
    Precedence: an explicit setting, then SANZARU_REALTIME_TURN_TIMEOUT, then
    the derived bound.
    """
    if explicit > 0:
        return explicit
    raw = os.getenv(TURN_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning("%s=%r is not a number - deriving the turn timeout instead", TURN_TIMEOUT_ENV, raw)
        else:
            if value > 0:
                return value
    return max(TURN_TIMEOUT_FLOOR, turn_seconds * TURN_TIMEOUT_FACTOR)


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    """Episode-wide knobs the producer needs for every act."""

    model: str = DEFAULT_REALTIME_MODEL
    show_title: str = "Untitled"
    premise: str = ""
    style: str = ""
    max_turn_tokens: int = DEFAULT_TURN_TOKENS
    turn_seconds: float = DEFAULT_TURN_SECONDS
    sample_rate: int = REALTIME_SAMPLE_RATE
    turn_timeout_s: float = 0.0
    """0 → SANZARU_REALTIME_TURN_TIMEOUT, else derived; see `turn_timeout_seconds`."""


# ---------- prompts ----------


def build_instructions(
    brief: ActBrief,
    host: HostSpec,
    others: Sequence[HostSpec],
    settings: SimulationSettings,
    *,
    is_first_act: bool,
    is_last_act: bool,
) -> str:
    """The session instructions one agent records this act under.

    Kept stable for the whole act so prompt caching can hold it — that is what
    keeps uncached input flat per turn instead of growing with the transcript.
    """
    lines = [
        f"You are recording an episode of a podcast called '{settings.show_title}'.",
    ]
    if settings.premise:
        lines.append(f"The show is about: {settings.premise}")
    if settings.style:
        lines.append(f"Tone and format: {settings.style}")

    lines.append("")
    lines.append(f"THIS SEGMENT — {brief.title}: {brief.topic}")
    if brief.direction:
        lines.append(f"How to play it: {brief.direction}")
    if brief.talking_points:
        lines.append("Ground to cover, roughly in this order:")
        lines.extend(f"  - {point}" for point in brief.talking_points)

    if brief.prior_context:
        # Acts record in separate sessions and in parallel, so this is the only
        # thing standing between act 4 and a second cold open.
        lines.append("")
        lines.append(f"ALREADY COVERED EARLIER IN THIS EPISODE: {brief.prior_context}")
        lines.append("Do not re-introduce the show, yourselves, or that ground. Pick up mid-conversation.")
    elif is_first_act:
        lines.append("")
        lines.append("This is the top of the episode. Welcome listeners once, briefly, then get into it.")

    if brief.upcoming:
        # Symmetric to prior_context. Without it an act happily explains the next
        # act's material, and the next act then sounds like a rerun.
        lines.append("")
        lines.append(f"LATER SEGMENTS OWN THIS GROUND — do not go there now: {brief.upcoming}")
        lines.append("If the conversation drifts toward it, say it is coming up and steer back.")

    if brief.handoff and not is_last_act:
        lines.append("")
        lines.append(f"LEAVE THE CONVERSATION HERE: {brief.handoff}")
        lines.append("Do not sign off or thank listeners — the episode continues after this segment.")
    elif is_last_act:
        lines.append("")
        lines.append(
            "The episode ends after this segment, but do NOT start wrapping up on your own — keep the "
            "conversation moving until the producer cues the close. When cued, land the through-line and "
            "sign off briefly, exactly once. If your co-host has already said goodbye, one short goodbye "
            "back and nothing more."
        )

    lines.extend(
        [
            "",
            "HOW TO TALK:",
            f"  - HARD LENGTH RULE: one or two sentences per turn. Never three. Under "
            f"{settings.turn_seconds:.0f} seconds. If you have two things to say, say one and let "
            "the other person answer — this is a fast back-and-forth, not a monologue.",
            "  - React to what the other person actually just said. Disagree where you genuinely would.",
            "  - Never repeat, echo, or rephrase a sentence the other person just said, and never quote "
            "it back as agreement. Add something new, sharpen it, or push back.",
            "  - Interrupt yourself, trail off, think out loud. Real conversation, not narration.",
            "  - Never speak stage directions, never summarize what was already said, and never speak a "
            "name followed by a colon — yours or anyone's. You are heard, not read.",
            "  - Introduce yourself at most once, in the whole episode.",
            "",
            host.persona or f"You are {host.name}.",
        ]
    )

    if others:
        names = ", ".join(other.name for other in others)
        lines.append("")
        lines.append(f"You are talking with: {names}. You will HEAR them speak; respond to what they actually said.")

    return "\n".join(lines)


def _steering_note(
    brief: ActBrief,
    turn_index: int,
    *,
    is_final_turn: bool,
    is_first_act: bool,
    is_last_act: bool,
    point_index: int | None,
) -> str | None:
    """The producer note for one turn, or None to let the agents run.

    A caller-supplied `turn_notes` entry wins outright — including on the final
    turn, where taking it over means taking on the job of landing the act. An
    empty string is a deliberate "say nothing this turn", distinct from having
    no opinion.
    """
    override = brief.turn_notes.get(turn_index)
    if override is not None:
        return override or None

    if is_final_turn:
        if is_last_act:
            return "Final turn of the episode: land the through-line and sign off warmly. Two sentences."
        if brief.handoff:
            return f"Final turn of this segment: bring it to a natural rest on — {brief.handoff}. Two sentences. Do not sign off."
        return "Final turn of this segment: bring it to a natural rest. Two sentences. Do not sign off."

    if turn_index == 0:
        if is_first_act:
            return "Open the episode in ONE short sentence, then put the topic to your co-host."
        return (
            "Pick up mid-conversation — no greeting, no re-introduction. "
            "One short sentence that moves onto this segment's topic."
        )

    if point_index is not None:
        return f"Move the conversation onto: {brief.talking_points[point_index]}"

    return None


def _point_schedule(brief: ActBrief) -> dict[int, int]:
    """Map turn index → talking point to introduce on that turn.

    Points are spread across the turns available *before* the closing turn, so
    the last point still gets air rather than being cut off by the sign-off. The
    first point is not scheduled: the opening note already puts the topic on the
    table.
    """
    points = brief.talking_points
    if len(points) <= 1 or brief.max_turns <= 2:
        return {}
    usable = brief.max_turns - 1
    # Turn 0 opens the act and turn `usable` lands it, so points ride in between.
    slots = list(range(1, usable))
    to_place = min(len(points) - 1, len(slots))
    if to_place < len(points) - 1:
        # A short act with a long list genuinely cannot cover it. Say which
        # points go uncovered rather than quietly overwriting one on its turn,
        # which is what this did before and nothing downstream could detect.
        logger.warning(
            "act %r: %d turns leave room for %d of %d talking points, so these will not be "
            "introduced: %s - raise max_turns or cut points",
            brief.id,
            brief.max_turns,
            to_place + 1,
            len(points),
            "; ".join(points[to_place + 1 :]),
        )

    schedule: dict[int, int] = {}
    previous = 0
    for position, point_index in enumerate(range(1, to_place + 1)):
        # Spread proportionally, but never later than the last turn that still
        # leaves a slot for every point behind this one, and never on a turn
        # already spoken for. Both bounds together make every point land.
        latest = slots[len(slots) - (to_place - position)]
        turn = min(max(round(point_index * usable / len(points)), previous + 1), latest)
        schedule[turn] = point_index
        previous = turn
    return schedule


# ---------- recording ----------


def _default_connect(model: str) -> AbstractAsyncContextManager[AsyncRealtimeConnection]:
    from ...config import get_client

    return get_client().realtime.connect(model=model)


def _resolve_order(brief: ActBrief, hosts: Sequence[HostSpec], start_index: int) -> list[int]:
    """Who holds the floor, as host indices to cycle through.

    Defaults to round-robin from `start_index` (rotated per act so the same voice
    does not open every segment). A `speaking_order` on the brief replaces that
    outright, so a caller can choreograph a host following their own point.

    An act that carries `turn_notes` but no `speaking_order` pins the rotation to
    the first listed host instead: index-keyed notes are written against *some*
    assumed rotation, and the per-act start rotation silently reassigned them —
    a note written as Dan's objection landed in Maya's mouth on odd acts, which
    is how a mandated reveal got dropped in a live run. Deterministic beats
    varied here.
    """
    if not brief.speaking_order:
        if brief.turn_notes and len(hosts) > 1 and start_index != 0:
            logger.info(
                "act %r: turn_notes present without speaking_order - pinning the rotation to the "
                "first listed host so notes land on the intended speaker (set speaking_order to "
                "choreograph explicitly)",
                brief.id,
            )
            start_index = 0
        return [(start_index + i) % len(hosts) for i in range(len(hosts))]

    by_id = {host.id: index for index, host in enumerate(hosts)}
    unknown = [host_id for host_id in brief.speaking_order if host_id not in by_id]
    if unknown:
        raise ValueError(
            f"act {brief.id!r} speaking_order names hosts that are not in the episode: {', '.join(unknown)}"
        )
    return [by_id[host_id] for host_id in brief.speaking_order]


async def run_act(
    brief: ActBrief,
    hosts: Sequence[HostSpec],
    settings: SimulationSettings,
    *,
    is_first_act: bool = False,
    is_last_act: bool = False,
    start_index: int = 0,
    connect: ConnectFactory | None = None,
    budget: CostBudget | None = None,
    on_turn: Callable[[Turn], None] | None = None,
) -> ActResult:
    """Record one act: open a connection per host, then run the floor.

    Args:
        brief: What this act is about and how long it gets.
        hosts: Participants, in speaking rotation order.
        settings: Episode-wide model/style/length knobs.
        is_first_act / is_last_act: Whether to cold-open and whether to sign off.
        start_index: Which host takes the first turn under the default
            round-robin (rotated per act so the same voice does not open every
            segment). Ignored when the brief sets `speaking_order`.
        connect: Connection factory, injected by tests.
        budget: Shared cost ceiling, charged after every turn.
        on_turn: Progress callback, called as each turn lands.

    Returns:
        The act's audio, turn list, and usage.
    """
    if not hosts:
        raise ValueError(f"act {brief.id!r} has no hosts")

    connect_fn = connect or _default_connect
    result = ActResult(act_id=brief.id)
    schedule = _point_schedule(brief)
    max_turn_tokens = settings.max_turn_tokens or turn_token_cap(settings.turn_seconds)
    turn_timeout = turn_timeout_seconds(settings.turn_seconds, settings.turn_timeout_s)
    order = _resolve_order(brief, hosts, start_index)

    async with contextlib.AsyncExitStack() as stack:
        agents: list[RealtimeAgent] = []
        for host in hosts:
            model = host.model or settings.model
            connection = await stack.enter_async_context(connect_fn(model))
            agents.append(
                RealtimeAgent(
                    host,
                    connection,
                    model=model,
                    max_turn_tokens=max_turn_tokens,
                    sample_rate=settings.sample_rate,
                )
            )

        for agent in agents:
            others = [h for h in hosts if h.id != agent.id]
            await agent.configure(
                build_instructions(
                    brief,
                    agent.spec,
                    others,
                    settings,
                    is_first_act=is_first_act,
                    is_last_act=is_last_act,
                )
            )

        turn_index = 0
        closing_delivered = False
        # The planned budget is a scheduling assumption, not a stop: an act ends
        # on its closing turn near target_seconds, and may borrow extra turns to
        # get there when the model's turns run shorter than assumed.
        hard_cap = min(MAX_ACT_TURNS, max(brief.max_turns, int(brief.max_turns * TURN_EXTENSION_FACTOR + 0.999)))

        while True:
            over_time = result.seconds >= brief.target_seconds
            if closing_delivered:
                result.stop_reason = "target_seconds" if over_time else "complete"
                break
            if turn_index >= hard_cap:
                result.stop_reason = "max_turns"
                break
            if turn_index == brief.max_turns:
                logger.info(
                    "act %r: %d planned turns filled %.0fs of a %.0fs target - extending up to %d turns",
                    brief.id,
                    brief.max_turns,
                    result.seconds,
                    brief.target_seconds,
                    hard_cap,
                )

            # Cue the close when one more average turn would reach the target,
            # not after the target is already overshot: the closing turn itself
            # still has to fit. Never on the opening turn, though — `target_seconds`
            # only has to be > 0, and an act targeting less than one turn would
            # otherwise open with its own sign-off. `over_time` and the hard cap
            # still bind, so a genuine one-turn act (max_turns=1) still lands.
            avg_turn = (result.seconds / turn_index) if turn_index >= 2 else settings.turn_seconds
            close_is_due = turn_index >= 1 and result.seconds + avg_turn >= brief.target_seconds
            is_final_turn = over_time or close_is_due or turn_index == hard_cap - 1
            agent = agents[order[turn_index % len(order)]]

            # A caller note on the last *planned* turn is the caller taking over
            # the landing. Extension moves the landing, so the note rides with
            # the closing turn wherever it ends up rather than firing mid-act.
            closing_override = brief.turn_notes.get(brief.max_turns - 1) if brief.max_turns > 1 else None
            if closing_override is not None and is_final_turn:
                note = closing_override or None
            elif closing_override is not None and turn_index == brief.max_turns - 1:
                note = EXTENSION_NOTE
            else:
                note = _steering_note(
                    brief,
                    turn_index,
                    is_final_turn=is_final_turn,
                    is_first_act=is_first_act,
                    is_last_act=is_last_act,
                    point_index=schedule.get(turn_index),
                )
                if note is None and not is_final_turn and turn_index >= brief.max_turns - 1:
                    # Past the planned budget the point schedule has nothing left
                    # to say, and an unsteered extension turn is where recap
                    # loops start.
                    note = EXTENSION_NOTE
            # One bound over the whole turn rather than over `speak()` alone: a
            # stalled session hangs on the steer, or on feeding the others, just
            # as readily. Everything downstream — the act's checkpoint, the
            # resume path, the CLI's exit code — only needs a hung turn to end
            # in *some* exception instead of holding a limiter slot forever.
            try:
                with anyio.fail_after(turn_timeout):
                    if note:
                        await agent.steer(note)

                    spoken = await agent.speak()
                    turn = Turn(
                        act_id=brief.id,
                        index=turn_index,
                        speaker_id=agent.id,
                        speaker_name=agent.name,
                        text=spoken.text,
                        seconds=round(spoken.seconds, 2),
                        truncated=spoken.truncated,
                    )
                    result.audio.append(TurnAudio(turn=turn, pcm=spoken.pcm))
                    result.usage = result.usage + spoken.usage
                    logger.debug("%s turn %d [%s] %.1fs", brief.id, turn_index + 1, agent.name, turn.seconds)
                    if on_turn is not None:
                        on_turn(turn)
                    if budget is not None:
                        budget.charge(spoken.usage, agent.model)

                    # Everyone else hears it. This is what makes it a
                    # conversation rather than N monologues interleaved.
                    for other in agents:
                        if other is not agent:
                            await other.hear(spoken.pcm)
            except TimeoutError as exc:
                raise RealtimeAPIError(
                    f"{brief.id}: {agent.name}'s turn {turn_index + 1} made no progress for "
                    f"{turn_timeout:.0f}s — the realtime session looks stalled "
                    f"(raise {TURN_TIMEOUT_ENV} if turns legitimately run this long)"
                ) from exc

            if is_final_turn:
                closing_delivered = True
            turn_index += 1

    return result
