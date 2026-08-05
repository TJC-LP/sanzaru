# SPDX-License-Identifier: MIT
"""Realtime conversation simulation — podcasts that are performed, not read.

`tools/simulate_podcast.py` is the entry point; this package holds the pieces:

- `types` — rundown/act/turn/usage values and the PCM16 helpers
- `agent` — one persona on one Realtime connection (speak / hear / steer)
- `producer` — floor control, coverage steering, and act budgets
- `rundown` — pre-production: a premise becomes parallel-recordable acts
- `budget` — a shared cost ceiling, charged every turn
- `pricing` — token→dollars, and the measured rates a dry run projects from
- `qc` — transcribe the rendered audio and judge it against the rundown

Importing this package pulls in pydantic but not `openai`; every SDK import is
function-local or TYPE_CHECKING-only.
"""

from .budget import CostBudget
from .producer import (
    DEFAULT_REALTIME_MODEL,
    DEFAULT_TURN_SECONDS,
    DEFAULT_TURN_TOKENS,
    DEFAULT_VOICES,
    SimulationSettings,
    build_instructions,
    run_act,
)
from .rundown import DEFAULT_PLANNER_MODEL, RundownRequest, build_rundown, plan_rundown, turns_for
from .types import (
    REALTIME_SAMPLE_RATE,
    ActBrief,
    ActResult,
    ActSummary,
    HostSpec,
    RealtimeUsage,
    Rundown,
    Turn,
    TurnAudio,
    pcm_seconds,
    pcm_silence,
)

__all__ = [
    "DEFAULT_PLANNER_MODEL",
    "DEFAULT_REALTIME_MODEL",
    "DEFAULT_TURN_SECONDS",
    "DEFAULT_TURN_TOKENS",
    "DEFAULT_VOICES",
    "REALTIME_SAMPLE_RATE",
    "ActBrief",
    "ActResult",
    "ActSummary",
    "CostBudget",
    "HostSpec",
    "RealtimeUsage",
    "Rundown",
    "RundownRequest",
    "SimulationSettings",
    "Turn",
    "TurnAudio",
    "build_instructions",
    "build_rundown",
    "pcm_seconds",
    "pcm_silence",
    "plan_rundown",
    "run_act",
    "turns_for",
]
