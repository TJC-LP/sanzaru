# SPDX-License-Identifier: MIT
"""What a Higgsfield video job will cost, before it is submitted.

`POST /estimate/<slug>` is free and answers in one of two shapes: a numeric
`{"type": "estimate", "usd", "credits", "discount"}` for most models, or only
prose — `{"type": "description", "pricing_description"}` — for Seedance 2.5 and
Wan 3.0. Seedance is the default model, so a `max_cost_usd` cap that trusted
only the numeric shape would be unenforceable exactly where it matters most.
Seedance's prose does state a precise formula, so it is priced locally:

    billable tokens = ceil(width × height × (input video s + generated s) × 24 / 1024)
    USD             = tokens / 1000 × rate   (0.0214 at 480p/720p, 0.0234 at 1080p)

which reproduces the published per-second rates exactly: 1280×720 →
$0.4622/s, 854×480 → $0.2056/s. Input video seconds count, so an edit bills
the source twice (the output is as long as the source) and an extend bills the
source plus the added duration.

Rates can go stale. `SANZARU_HIGGSFIELD_PRICE_<FAMILY>` overrides them
(`0.0214` for every tier, or `0.0214,0.0234` for 480/720p and 1080p) — read
from the real environment only: like `SANZARU_REALTIME_PRICE_*`, it is not on
the `.env` allowlist, because a planted file must not be able to lower the
prices a cost ceiling is enforced against.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from typing_extensions import TypedDict

from .errors import CostCapExceededError, UnpricedVideoError
from .types import EstimateResponse

logger = logging.getLogger("sanzaru")

PRICES_CAPTURED = "2026-09-28"


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """USD per 1,000 video tokens, per resolution tier.

    `video_input_factor` scales the rate for jobs with a video input (edit,
    extend). Seedance bills those at 0.6x — $0.01284/1k at 480/720p, per the
    video-edit and video-extend estimate text captured 2026-09-28 — over
    input *plus* generated seconds.
    """

    usd_per_1k_tokens: Mapping[str, float]
    video_input_factor: float = 1.0


PRICES: dict[str, TokenPricing] = {
    "seedance-2.5": TokenPricing(
        usd_per_1k_tokens={"480p": 0.0214, "720p": 0.0214, "1080p": 0.0234}, video_input_factor=0.6
    ),
}

# Output frame sizes by (resolution, aspect ratio). Only the 16:9 row is
# confirmed (it reproduces the published per-second rates); the others are the
# usual equal-area buckets and are UNVERIFIED — confirm with ffprobe on real
# outputs in the live smoke before relying on the exact figure.
DIMENSIONS: dict[tuple[str, str], tuple[int, int]] = {
    ("480p", "16:9"): (854, 480),
    ("480p", "4:3"): (736, 544),
    ("480p", "1:1"): (640, 640),
    ("480p", "3:4"): (544, 736),
    ("480p", "9:16"): (480, 854),
    ("480p", "21:9"): (960, 416),
    ("720p", "16:9"): (1280, 720),
    ("720p", "4:3"): (1112, 834),
    ("720p", "1:1"): (960, 960),
    ("720p", "3:4"): (834, 1112),
    ("720p", "9:16"): (720, 1280),
    ("720p", "21:9"): (1470, 630),
    ("1080p", "16:9"): (1920, 1080),
    ("1080p", "9:16"): (1080, 1920),
}

_DEFAULT_RESOLUTION = "720p"
_DEFAULT_ASPECT = "16:9"
_DEFAULT_DURATION = 5
_FPS_FACTOR = 24


def price_env_name(family: str) -> str:
    """The `SANZARU_HIGGSFIELD_PRICE_<FAMILY>` variable that would price `family`."""
    return "SANZARU_HIGGSFIELD_PRICE_" + family.upper().replace("-", "_").replace(".", "_")


def _env_override(family: str) -> TokenPricing | None:
    """Read `SANZARU_HIGGSFIELD_PRICE_<FAMILY>`; a malformed value warns and falls back.

    Someone who set the variable wanted it to take effect, so a value that
    cannot be used is reported rather than silently billed at the table rate.
    """
    key = price_env_name(family)
    raw = os.getenv(key)
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    try:
        if len(parts) not in (1, 2):
            raise ValueError(f"expected 1 or 2 comma-separated values, got {len(parts)}")
        values = [float(p) for p in parts]
        if any(v < 0 or not math.isfinite(v) for v in values):
            raise ValueError("values must be finite and non-negative")
    except ValueError as exc:
        logger.warning(
            "%s=%r is not usable (%s) - falling back to table prices for %r. "
            "Expected USD per 1k video tokens: <480p/720p>[,<1080p>]",
            key,
            raw,
            exc,
            family,
        )
        return None
    standard = values[0]
    high = values[1] if len(values) == 2 else values[0]
    # The override replaces the token rate only; the video-input discount is a
    # property of the model, not of the price list, so it carries over.
    base = PRICES.get(family)
    return TokenPricing(
        usd_per_1k_tokens={"480p": standard, "720p": standard, "1080p": high},
        video_input_factor=base.video_input_factor if base else 1.0,
    )


def prices_for(family: str) -> TokenPricing | None:
    """Token prices for a model family, or None when nothing prices it locally."""
    return _env_override(family) or PRICES.get(family)


def frame_size(resolution: str, aspect_ratio: str | None) -> tuple[int, int] | None:
    """Output dimensions; with no aspect ratio (framing follows an image), the largest in the tier."""
    if aspect_ratio is not None:
        return DIMENSIONS.get((resolution, aspect_ratio))
    sizes = [size for (res, _), size in DIMENSIONS.items() if res == resolution]
    return max(sizes, key=lambda wh: wh[0] * wh[1]) if sizes else None


def _as_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def local_estimate(
    family: str,
    operation: str,
    arguments: Mapping[str, object],
    input_video_seconds: float | None = None,
) -> float | None:
    """USD for a job priced from the local table, or None when it cannot be priced.

    text: the requested duration. image: the duration at the largest frame in
    the tier (framing follows the image, so this is an upper bound — the right
    direction for a cap). extend: source + added duration. edit: the output is
    as long as the source, so the source is billed twice. An edit or extend
    whose source duration is unknown is unpriceable (None), never guessed.
    """
    pricing = prices_for(family)
    if pricing is None:
        return None
    resolution = _as_str(arguments.get("resolution"), _DEFAULT_RESOLUTION)
    rate = pricing.usd_per_1k_tokens.get(resolution)
    if rate is None:
        return None
    duration = _as_int(arguments.get("duration"), _DEFAULT_DURATION)
    if operation == "text":
        size = frame_size(resolution, _as_str(arguments.get("aspect_ratio"), _DEFAULT_ASPECT))
        seconds = float(duration)
    elif operation == "image":
        size = frame_size(resolution, None)
        seconds = float(duration)
    elif operation in ("edit", "extend"):
        if input_video_seconds is None or input_video_seconds <= 0:
            return None
        size = frame_size(resolution, None)
        generated = input_video_seconds if operation == "edit" else float(duration)
        seconds = input_video_seconds + generated
        rate *= pricing.video_input_factor
    else:
        return None
    if size is None:
        return None
    width, height = size
    tokens = math.ceil(width * height * seconds * _FPS_FACTOR / 1024)
    return tokens / 1000 * rate


class VideoCost(TypedDict):
    """What a job is expected to cost, and how sanzaru knows."""

    usd: float | None
    credits: float | None
    basis: Literal["api", "local_table", "unpriced", "unavailable"]
    usd_after_discount: float | None
    pricing_description: str | None
    note: str | None


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def resolve_cost(remote: EstimateResponse | None, local: float | None, *, estimate_failed: bool = False) -> VideoCost:
    """Combine the API's estimate with the local table into one `VideoCost`."""
    if remote is not None and remote.get("type") == "estimate":
        discount = remote.get("discount")
        return VideoCost(
            usd=_to_float(remote.get("usd")),
            credits=_to_float(remote.get("credits")),
            basis="api",
            usd_after_discount=_to_float(discount.get("usd")) if discount else None,
            pricing_description=None,
            note=None,
        )
    description = remote.get("pricing_description") if remote is not None else None
    if local is not None:
        return VideoCost(
            usd=round(local, 4),
            credits=None,
            basis="local_table",
            usd_after_discount=None,
            pricing_description=description,
            note=f"priced from sanzaru's table (captured {PRICES_CAPTURED}); the API gives only a description",
        )
    if estimate_failed:
        return VideoCost(
            usd=None,
            credits=None,
            basis="unavailable",
            usd_after_discount=None,
            pricing_description=None,
            note="the estimate request failed",
        )
    return VideoCost(
        usd=None,
        credits=None,
        basis="unpriced",
        usd_after_discount=None,
        pricing_description=description,
        note="no numeric price is available for this model",
    )


def cap_amount(cost: VideoCost) -> float | None:
    """The figure a cap is checked against: the larger of list and discounted price.

    Whether the API's `usd` is before or after the account discount is not yet
    settled, so the conservative reading is used.
    """
    values = [v for v in (cost["usd"], cost["usd_after_discount"]) if v is not None]
    return max(values) if values else None


def enforce_cap(cost: VideoCost, max_cost_usd: float | None, model: str) -> None:
    """Refuse a job over `max_cost_usd`, or one no price exists for while a cap is set."""
    if max_cost_usd is None:
        return
    amount = cap_amount(cost)
    if amount is None:
        raise UnpricedVideoError(
            f"max_cost_usd={max_cost_usd} cannot be enforced for {model}: {cost['note'] or 'no price available'}. "
            "Neither the API nor the local table could price it (for Seedance edit/extend the local price "
            "needs the source clip's duration). Drop the cap, or choose a model the API prices. Nothing was submitted.",
            estimate_usd=None,
            limit_usd=max_cost_usd,
            basis=cost["basis"],
            model=model,
        )
    if amount > max_cost_usd:
        raise CostCapExceededError(
            f"estimated ${amount:.2f} for {model} is over max_cost_usd=${max_cost_usd:.2f} "
            f"(basis: {cost['basis']}). Lower the resolution or duration, or raise the cap. Nothing was submitted.",
            estimate_usd=amount,
            limit_usd=max_cost_usd,
            basis=cost["basis"],
            model=model,
        )
