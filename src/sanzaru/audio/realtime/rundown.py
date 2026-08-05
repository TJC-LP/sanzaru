# SPDX-License-Identifier: MIT
"""Pre-production: turn a premise into an act-by-act rundown.

Acts are recorded in parallel, in separate sessions, so no act can hear what
another act said. Coherence has to be arranged *before* recording — that is what
this module is for. A text model expands a premise into acts that each carry:

- `prior_context` — what earlier acts already covered, so act 4 doesn't
  re-introduce the show or re-litigate act 2's argument
- `handoff` — where this act must leave the conversation

Splitting this out as its own command (`sanzaru podcast rundown`) is deliberate:
a rundown is cheap and a recording is not, so the plan should be inspectable and
hand-editable before anyone spends realtime money on it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ...config import logger
from .producer import DEFAULT_TURN_SECONDS, DEFAULT_VOICES
from .types import ActBrief, HostSpec, Rundown

DEFAULT_PLANNER_MODEL = "gpt-5.5"
"""A text model, not a realtime one. One call per episode, and its output shapes
everything downstream, so this defaults to a strong model rather than a cheap one."""

DEFAULT_ACTS = 4
DEFAULT_TARGET_MINUTES = 12.0
MAX_ACTS = 24
"""Guard against a rundown that opens hundreds of concurrent sessions."""


# ---------- planner schema ----------
# Separate from the domain types on purpose: the model picks content, we assign
# mechanics (voices, ids, per-act duration and turn budgets).


class PlannedHost(BaseModel):
    name: str
    role: str
    persona: str


class PlannedAct(BaseModel):
    title: str
    topic: str
    talking_points: list[str]
    prior_context: str
    handoff: str


class PlannedRundown(BaseModel):
    title: str
    style: str
    hosts: list[PlannedHost]
    acts: list[PlannedAct]


class RundownRequest(BaseModel):
    """Everything `plan_rundown` needs, in one validated object."""

    premise: str
    acts: int = DEFAULT_ACTS
    target_minutes: float = DEFAULT_TARGET_MINUTES
    title: str | None = None
    style: str | None = None
    hosts: list[HostSpec] = Field(default_factory=list)
    turn_seconds: float = DEFAULT_TURN_SECONDS
    model: str = DEFAULT_PLANNER_MODEL


def slugify(value: str, fallback: str = "host") -> str:
    """Lowercase identifier from a display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or fallback


def turns_for(seconds: float, turn_seconds: float = DEFAULT_TURN_SECONDS) -> int:
    """Turn budget for an act of a given length.

    Turns land at roughly the stated cap — measured 16s against a 15s
    instruction — so budget one turn per `turn_seconds` plus one. Under-budgeting
    here is not harmless: an act that runs out of *seconds* before it runs out of
    turns stops on the duration budget instead of on its scripted closing turn,
    and its last talking point never gets air.
    """
    return max(4, math.ceil(seconds / max(4.0, turn_seconds)) + 1)


def assign_voices(hosts: Sequence[HostSpec]) -> list[HostSpec]:
    """Give every host a distinct realtime voice, keeping explicit choices."""
    taken = {h.voice for h in hosts if h.voice}
    available = [v for v in DEFAULT_VOICES if v not in taken]
    assigned: list[HostSpec] = []
    for host in hosts:
        if host.voice:
            assigned.append(host)
            continue
        voice = available.pop(0) if available else DEFAULT_VOICES[len(assigned) % len(DEFAULT_VOICES)]
        assigned.append(host.model_copy(update={"voice": voice}))
    return assigned


def _merge_hosts(planned: Sequence[PlannedHost], supplied: Sequence[HostSpec]) -> list[HostSpec]:
    """Supplied hosts win; the planner only fills gaps (and personas it wrote)."""
    if supplied:
        merged: list[HostSpec] = []
        for index, host in enumerate(supplied):
            persona = host.persona
            if not persona and index < len(planned):
                persona = planned[index].persona
            merged.append(host.model_copy(update={"persona": persona}))
        return assign_voices(merged)

    hosts = [
        HostSpec(
            id=slugify(host.name, f"host{index + 1}"),
            name=host.name,
            voice="",
            persona=f"{host.persona}".strip(),
        )
        for index, host in enumerate(planned)
    ]
    # Two hosts sharing a slug would collapse into one speaker downstream.
    seen: set[str] = set()
    for index, host in enumerate(hosts):
        if host.id in seen:
            hosts[index] = host.model_copy(update={"id": f"{host.id}{index + 1}"})
        seen.add(hosts[index].id)
    return assign_voices(hosts)


def _prompt(request: RundownRequest) -> str:
    """Instructions for the planner, including why prior_context matters."""
    lines = [
        "You are the producer of a podcast. Plan an episode as a rundown of "
        f"exactly {request.acts} acts totalling about {request.target_minutes:.0f} minutes.",
        "",
        f"PREMISE: {request.premise}",
    ]
    if request.title:
        lines.append(f"The episode title must be exactly: {request.title}")
    if request.style:
        lines.append(f"Tone and format: {request.style}")

    if request.hosts:
        roster = "; ".join(f"{h.name}" + (f" ({h.persona})" if h.persona else "") for h in request.hosts)
        lines.append(f"The hosts are fixed — use exactly these people, in this order: {roster}")
        lines.append("Write a persona for any host who does not already have one.")
    else:
        lines.append(
            "Invent two hosts with genuinely different jobs and temperaments — "
            "one who moves the conversation and translates, one with hands-on detail "
            "who pushes back. Give each a persona written in the second person "
            "('You are ...'), 2-3 sentences."
        )

    lines.extend(
        [
            "",
            "CRITICAL CONSTRAINT: each act is recorded SEPARATELY and IN PARALLEL by "
            "voice models that cannot hear the other acts. Coherence has to come "
            "entirely from what you write.",
            "  - `prior_context` for act N must state, in one or two sentences, what "
            "acts 1..N-1 already covered, so act N never re-introduces the show or "
            "repeats earlier ground. Act 1's prior_context must be an empty string.",
            "  - `handoff` for act N states where the conversation should be left so "
            "act N+1 can pick it up. The final act's handoff must be an empty string.",
            "  - Acts must not overlap. Each one owns distinct ground.",
            "",
            f"Each act runs about {request.target_minutes * 60 / request.acts:.0f} seconds — "
            f"roughly {turns_for(request.target_minutes * 60 / request.acts, request.turn_seconds)} "
            "short exchanges, two of which are spent opening and landing it.",
            "So give each act 3-4 concrete talking points, not 5+: fewer points "
            "explored properly beat a checklist nobody finishes. Points are specific "
            "claims, examples, or disagreements, not topic labels — 'why the bug moves "
            "between renders' is a talking point, 'debugging' is not. Keep each one to "
            "a single sentence.",
            "",
            "`style` is one line of tone/format direction applied to every act.",
        ]
    )
    return "\n".join(lines)


async def plan_rundown(request: RundownRequest) -> Rundown:
    """Expand a premise into a full, recordable rundown.

    Raises:
        ValueError: If the request is malformed or the planner returns no plan.
    """
    if not request.premise.strip():
        raise ValueError("premise is required to plan a rundown")
    if request.acts < 1:
        raise ValueError(f"acts must be >= 1 (got {request.acts})")
    if request.acts > MAX_ACTS:
        raise ValueError(f"acts must be <= {MAX_ACTS} (got {request.acts}) — each act opens one session per host")
    if request.target_minutes <= 0:
        raise ValueError(f"target_minutes must be > 0 (got {request.target_minutes})")

    from ...config import get_client

    client = get_client()
    logger.info("Planning a %d-act, %.0f-minute rundown with %s", request.acts, request.target_minutes, request.model)

    response = await client.responses.parse(
        model=request.model,
        input=_prompt(request),
        text_format=PlannedRundown,
    )
    planned = response.output_parsed
    if planned is None:
        raise ValueError(f"rundown planner ({request.model}) returned no structured plan")

    return build_rundown(planned, request)


def build_rundown(planned: PlannedRundown, request: RundownRequest) -> Rundown:
    """Turn a planner response into a rundown, assigning the mechanics ourselves."""
    hosts = _merge_hosts(planned.hosts, request.hosts)
    if not hosts:
        raise ValueError("rundown has no hosts")

    acts = planned.acts[: request.acts]
    if not acts:
        raise ValueError("rundown planner returned no acts")
    if len(acts) < request.acts:
        logger.warning("Planner returned %d acts, asked for %d", len(acts), request.acts)

    seconds_each = request.target_minutes * 60.0 / len(acts)
    briefs = [
        ActBrief(
            id=f"act{index + 1}",
            title=act.title,
            topic=act.topic,
            talking_points=list(act.talking_points),
            target_seconds=round(seconds_each, 1),
            max_turns=turns_for(seconds_each, request.turn_seconds),
            prior_context="" if index == 0 else act.prior_context,
            handoff="" if index == len(acts) - 1 else act.handoff,
        )
        for index, act in enumerate(acts)
    ]

    return Rundown(
        title=request.title or planned.title,
        premise=request.premise,
        style=request.style or planned.style,
        hosts=hosts,
        acts=briefs,
    )
