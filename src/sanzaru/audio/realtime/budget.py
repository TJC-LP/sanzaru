# SPDX-License-Identifier: MIT
"""A shared spend ceiling for one simulation run.

Realtime is the first thing sanzaru does where a prompt-shaped mistake can cost
real money in a loop, so the ceiling is checked after *every turn* across every
parallel act rather than once per act. One budget object is shared by all acts;
anyio tasks are cooperative on a single event loop and `charge()` never awaits,
so the accumulator needs no lock.

Blowing the ceiling raises `CostCeilingError`, which cancels the sibling acts.
Acts that already finished are on disk as checkpoints, so the run is resumable
rather than lost.
"""

from __future__ import annotations

from ...exceptions import CostCeilingError
from .pricing import usage_cost
from .types import RealtimeUsage


class CostBudget:
    """Accumulates spend and aborts the run when it crosses `limit_usd`."""

    def __init__(self, limit_usd: float | None = None) -> None:
        self.limit_usd = limit_usd
        self._spent = 0.0
        self._unpriced_models: set[str] = set()
        # dict-as-ordered-set: an act id appears once, in the order it landed.
        self._completed_acts: dict[str, None] = {}

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def completed_acts(self) -> list[str]:
        """Act ids whose checkpoint is on disk right now.

        This is the number the CLI prints on a ceiling abort ("N act(s)
        checkpointed and safe"), and the user decides whether to resume on it,
        so it has to be a set of what is actually there — not a running tally of
        writes.
        """
        return list(self._completed_acts)

    @property
    def unpriced_models(self) -> list[str]:
        """Models charged at zero because no price is known for them.

        Reported rather than silently ignored: a ceiling that quietly stops
        counting is worse than no ceiling.
        """
        return sorted(self._unpriced_models)

    def charge(self, usage: RealtimeUsage, model: str) -> None:
        """Add one turn's usage; raise CostCeilingError once over the ceiling."""
        cost = usage_cost(usage, model)
        if cost is None:
            self._unpriced_models.add(model)
            return
        self._spent += cost
        if self.limit_usd is not None and self._spent > self.limit_usd:
            raise CostCeilingError(
                f"cost ceiling reached: ${self._spent:.2f} spent of ${self.limit_usd:.2f} limit",
                spent_usd=self._spent,
                limit_usd=self.limit_usd,
                completed_acts=self.completed_acts,
            )

    def mark_act_complete(self, act_id: str) -> None:
        """Record an act whose checkpoint is safely written.

        Idempotent: `--qc-retry` re-records and re-checkpoints an act that is
        already on disk, and a resumed run marks the checkpoints it read back.
        Neither may inflate the count.
        """
        self._completed_acts[act_id] = None
