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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...config import logger
from .agent import RealtimeAgent
from .budget import CostBudget
from .pricing import OUTPUT_TOKENS_PER_SECOND
from .types import (
    REALTIME_SAMPLE_RATE,
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

DEFAULT_VOICES: tuple[str, ...] = (
    "marin",
    "cedar",
    "alloy",
    "sage",
    "verse",
    "coral",
    "ash",
    "ballad",
    "echo",
    "shimmer",
)
"""Realtime API voices, in assignment order. marin/cedar lead because they are the
most natural of the set in conversation — that pairing is what the spike used."""

DEFAULT_TURN_SECONDS = 15.0
"""Target upper bound for one turn, stated in the prompt. Enforced socially, not
mechanically — a hard cut mid-sentence sounds far worse than a long turn."""

TURN_TOKEN_HEADROOM = 1.5
"""How far past the target a turn may run before the token cap stops it.

Tuned against live recordings: at 1.0 the cap did the work the prompt should do
and clipped 17 of 29 turns mid-sentence; at 2.0 nothing clipped but turns
stretched to ~22s against a 15s rule. 1.5 leaves the prompt in charge while
still bounding a runaway."""

DEFAULT_TURN_TOKENS = 0
"""0 → derive from `turn_seconds` (see `turn_token_cap`)."""


def turn_token_cap(turn_seconds: float) -> int:
    """Per-turn `max_output_tokens`: a runaway backstop, not a target.

    A fixed 520 — 26 seconds at the audio rate alone — truncated 17 of 29 turns
    in a live recording, because `max_output_tokens` counts the *transcript*
    alongside the audio. Budgeting both rates and doubling for headroom means the
    cap only fires on a genuine monologue, and the prompt's length rule stays the
    thing that actually shapes turn length.
    """
    return max(256, int(turn_seconds * OUTPUT_TOKENS_PER_SECOND * TURN_TOKEN_HEADROOM))


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
        lines.append("This is the end of the episode. Land the through-line and sign off warmly, once.")

    lines.extend(
        [
            "",
            "HOW TO TALK:",
            f"  - HARD LENGTH RULE: one or two sentences per turn. Never three. Under "
            f"{settings.turn_seconds:.0f} seconds. If you have two things to say, say one and let "
            "the other person answer — this is a fast back-and-forth, not a monologue.",
            "  - React to what the other person actually just said. Disagree where you genuinely would.",
            "  - Interrupt yourself, trail off, think out loud. Real conversation, not narration.",
            "  - Never speak stage directions, never say your own name as a label, never summarize what "
            "was already said.",
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
    usable = max(1, brief.max_turns - 1)
    schedule: dict[int, int] = {}
    for point_index in range(1, len(points)):
        turn = round(point_index * usable / len(points))
        turn = min(max(turn, 1), usable - 1) if usable > 1 else 1
        # Never overwrite: two points landing on one turn would drop one.
        while turn in schedule and turn + 1 < usable:
            turn += 1
        schedule[turn] = point_index
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
    """
    if not brief.speaking_order:
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

        while True:
            if turn_index >= brief.max_turns:
                result.stop_reason = "complete" if closing_delivered else "max_turns"
                break
            over_time = result.seconds >= brief.target_seconds
            if over_time and closing_delivered:
                result.stop_reason = "target_seconds"
                break

            is_final_turn = over_time or turn_index == brief.max_turns - 1
            agent = agents[order[turn_index % len(order)]]

            note = _steering_note(
                brief,
                turn_index,
                is_final_turn=is_final_turn,
                is_first_act=is_first_act,
                is_last_act=is_last_act,
                point_index=schedule.get(turn_index),
            )
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

            # Everyone else hears it. This is what makes it a conversation
            # rather than N monologues interleaved.
            for other in agents:
                if other is not agent:
                    await other.hear(spoken.pcm)

            if is_final_turn:
                closing_delivered = True
            turn_index += 1

    return result
