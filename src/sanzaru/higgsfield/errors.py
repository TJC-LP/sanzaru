# SPDX-License-Identifier: MIT
"""Errors raised by the Higgsfield video backend.

Stdlib only, so the CLI's error classifier can import it without paying for
httpx. Every HTTP failure becomes one `HiggsfieldAPIError` carrying the status
code and a `kind`, because the right reaction differs per kind and the API
does not use HTTP codes the conventional way: the per-account concurrency cap
answers **400** (not 429, no Retry-After), and "insufficient credits" is 403.
"""

from __future__ import annotations

import re
from typing import Literal

from ..exceptions import SanzaruError

ErrorKind = Literal[
    "invalid",
    "concurrency",
    "auth",
    "credits",
    "not_found",
    "validation",
    "blocked",
    "server",
    "disabled",
    "transport",
    "ambiguous_submit",
    "other",
]

_KIND_BY_STATUS: dict[int, ErrorKind] = {
    400: "invalid",
    401: "auth",
    403: "credits",
    404: "not_found",
    422: "validation",
    423: "blocked",
    503: "disabled",
}

_CONCURRENCY_DETAIL = re.compile(r"maximum number of concurrent requests", re.IGNORECASE)

CONCURRENCY_LIMIT = 4
"""Higgsfield's documented per-account cap on queued + processing requests.
It is enforced server-side and shared with the Higgsfield app and connector,
so no local limiter can guarantee it — it is surfaced, not prevented."""


class HiggsfieldAPIError(SanzaruError):
    """An HTTP or transport failure talking to api.higgsfield.ai."""

    def __init__(self, message: str, *, status_code: int | None, detail: str, kind: ErrorKind) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.kind: ErrorKind = kind

    @property
    def transient(self) -> bool:
        """Worth retrying a *read* (status, estimate, upload URL). Never a submit.

        503 counts: the API documents it as "model is disabled or not ready",
        and a gateway 503 on a status poll is the ordinary transient case.
        """
        return self.kind in ("server", "transport", "disabled") or self.status_code == 429


class HiggsfieldConcurrencyError(HiggsfieldAPIError):
    """The account already has `CONCURRENCY_LIMIT` jobs queued or processing.

    A rejected submit created nothing and charged nothing, so resubmitting
    after a running job finishes is safe — unlike most submit failures.
    """


class CostCapExceededError(ValueError):
    """A job's estimated cost is over the caller's `max_cost_usd`; nothing was submitted.

    A `ValueError` so the CLI reports it as usage (exit 2): the fix is in the
    request (a lower resolution, a shorter clip, a higher cap), not a retry.
    """

    def __init__(self, message: str, *, estimate_usd: float | None, limit_usd: float, basis: str, model: str) -> None:
        super().__init__(message)
        self.estimate_usd = estimate_usd
        self.limit_usd = limit_usd
        self.basis = basis
        self.model = model


class UnpricedVideoError(CostCapExceededError):
    """A cap was set but nothing could price the job, so the cap cannot be enforced.

    Refusing is the point: a cap that silently does nothing is worse than no
    cap — the same rule the simulated-podcast budget follows for unpriced models.
    """


def error_from_response(status_code: int, detail: str) -> HiggsfieldAPIError:
    """Map a non-2xx response to the typed error its status and detail imply."""
    detail = detail[:500]
    if status_code == 400 and _CONCURRENCY_DETAIL.search(detail):
        return HiggsfieldConcurrencyError(
            f"Higgsfield allows {CONCURRENCY_LIMIT} jobs in flight per account (the Higgsfield app counts too). "
            "Wait for a running job to finish (wait_for), then resubmit — nothing was charged.",
            status_code=status_code,
            detail=detail,
            kind="concurrency",
        )
    kind: ErrorKind = _KIND_BY_STATUS.get(status_code, "server" if status_code >= 500 else "other")
    return HiggsfieldAPIError(
        f"Higgsfield API error {status_code}: {detail}", status_code=status_code, detail=detail, kind=kind
    )
