# SPDX-License-Identifier: MIT
"""Cost estimation for realtime simulation.

Every guardrail in this feature (`--max-cost`, `--dry-run`, the running total on
stderr) needs to turn token counts into dollars, and realtime prices audio ~32x
higher than text on input, so a blended rate is not good enough.

Two honest caveats, both surfaced to the user rather than hidden:

1. **Prices go stale.** The table below is list pricing captured on 2026-08-05.
   Override any of it with `SANZARU_REALTIME_PRICE_<MODEL>` (see `_env_override`)
   without waiting for a release.
2. **Estimates are estimates.** `--dry-run` projects from the measured rates in
   `RATES`, which came from a live spike, not from a pricing page. Actual spend
   is reported from the API's own usage numbers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .types import RealtimeUsage, is_live_model

logger = logging.getLogger("sanzaru")

# ---------- measured generation rates ----------
# From live recordings on gpt-realtime-2.1 and -mini. Used only to project a dry
# run and to size the per-turn token backstop; actual usage always comes back
# from the API.
AUDIO_OUTPUT_TOKENS_PER_SECOND = 20.0
"""Audio output tokens per second of finished speech. Remarkably stable — three
independently recorded acts came in at 19.96, 19.99 and 20.05."""

TEXT_OUTPUT_TOKENS_PER_SECOND = 12.0
"""Transcript tokens per second of speech. The API returns a text transcript
alongside the audio and bills it separately, so a turn's real output rate is
this *plus* the audio rate. Missing that is what makes a naive
`max_output_tokens` truncate turns well short of their intended length."""

OUTPUT_TOKENS_PER_SECOND = AUDIO_OUTPUT_TOKENS_PER_SECOND + TEXT_OUTPUT_TOKENS_PER_SECOND
"""What `max_output_tokens` actually counts against. Measured 29-37 across acts;
the spread is transcript-driven (words per second varies far more than audio
frames do), so anything derived from this needs headroom."""

AUDIO_INPUT_TOKENS_PER_SECOND = 24.0
"""Audio input tokens per second heard. Every agent hears every other agent."""

TEXT_TOKENS_PER_TURN = 241.0
"""Uncached text input per turn. Measured flat across a session — prompt caching
keeps the instructions block from re-billing, which is why cost stays linear in
turns rather than quadratic."""

CACHE_HIT_RATE = 0.66
"""Share of input tokens served from cache in the spike."""


@dataclass(frozen=True, slots=True)
class ModelPrices:
    """USD per 1M tokens, plus a per-minute rate for models billed by duration."""

    text_input: float
    cached_text_input: float
    audio_input: float
    cached_audio_input: float
    audio_output: float
    text_output: float
    per_minute: float = 0.0
    """USD per session-minute. The Live API (`gpt-live-*`) bills only this —
    every token rate above is zero for it — and every realtime model bills
    only tokens, so the two halves of `usage_cost` never both apply."""


# List prices per 1M tokens, captured 2026-08-05 from OpenAI's pricing page.
PRICES: dict[str, ModelPrices] = {
    "gpt-realtime-2.1": ModelPrices(
        text_input=4.00,
        cached_text_input=0.40,
        audio_input=32.00,
        cached_audio_input=0.40,
        audio_output=64.00,
        text_output=24.00,
    ),
    "gpt-realtime-2.1-mini": ModelPrices(
        text_input=0.60,
        cached_text_input=0.06,
        audio_input=10.00,
        cached_audio_input=0.30,
        audio_output=20.00,
        text_output=2.40,
    ),
}

# The Live API is duration-billed: $0.05 per session-minute, metered per second,
# for every host's session across the whole act (listening included). No tokens.
PRICES["gpt-live-1"] = ModelPrices(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, per_minute=0.05)

# Older aliases bill like the generation they belong to.
PRICES["gpt-realtime-2"] = PRICES["gpt-realtime-2.1"]
PRICES["gpt-realtime"] = PRICES["gpt-realtime-2.1"]
PRICES["gpt-realtime-mini"] = PRICES["gpt-realtime-2.1-mini"]

TRANSCRIBE_USD_PER_MINUTE = 0.0045
"""gpt-transcribe, used only by the QC pass."""


def price_env_name(model: str) -> str:
    """The `SANZARU_REALTIME_PRICE_<MODEL>` variable that would price `model`.

    One spelling rule, used by the override reader and by every message that
    tells a user which variable to set: upper-cased, with `-` and `.` as `_`.
    """
    return "SANZARU_REALTIME_PRICE_" + model.upper().replace("-", "_").replace(".", "_")


PRICE_ENV_FIELDS = "text_in,cached_text_in,audio_in,cached_audio_in,audio_out,text_out[,per_minute]"
"""The value shape every message about `SANZARU_REALTIME_PRICE_<MODEL>` quotes."""


def _env_override(model: str) -> ModelPrices | None:
    """Read `SANZARU_REALTIME_PRICE_<MODEL>` — six or seven comma-separated values.

    Order matches ModelPrices:
    text_in,cached_text_in,audio_in,cached_audio_in,audio_out,text_out — USD per
    1M tokens — with an optional seventh, per_minute, in USD per session-minute
    for a duration-billed model. The six-value form is unchanged and leaves
    per_minute at 0. The model name is upper-cased with `-`/`.` becoming `_`,
    e.g. `SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1=4,0.4,32,0.4,64,24` or
    `SANZARU_REALTIME_PRICE_GPT_LIVE_1=0,0,0,0,0,0,0.05`.

    A malformed value warns and falls through to the table: someone who set this
    variable wanted it to take effect, and silently billing them at list price is
    the one outcome they were trying to avoid.
    """
    key = price_env_name(model)
    raw = os.getenv(key)
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    try:
        if len(parts) not in (6, 7):
            raise ValueError(f"expected 6 or 7 comma-separated values, got {len(parts)}")
        values = [float(p) for p in parts]
    except ValueError as exc:
        logger.warning(
            "%s=%r is not usable (%s) - falling back to table prices for %r. Expected %s",
            key,
            raw,
            exc,
            model,
            PRICE_ENV_FIELDS,
        )
        return None
    return ModelPrices(*values)


def prices_for(model: str) -> ModelPrices | None:
    """Prices for a model, or None when it is not in the table and has no override."""
    override = _env_override(model)
    if override is not None:
        return override
    if model in PRICES:
        return PRICES[model]
    # Dated snapshots (gpt-realtime-2025-08-28) bill like their base model. The
    # *longest* match wins, not the first: "gpt-realtime-2.1" is a prefix of
    # "gpt-realtime-2.1-mini", so insertion order priced every dated -mini
    # snapshot at the full tier — 3.2x over on audio output, silently, in both
    # the dry-run projection and the cost ceiling.
    base = max((known for known in PRICES if model.startswith(known)), key=len, default="")
    return PRICES[base] if base else None


def usage_cost(usage: RealtimeUsage, model: str) -> float | None:
    """USD for one act's usage, or None when the model has no known prices."""
    prices = prices_for(model)
    if prices is None:
        return None
    tokens = (
        usage.uncached_text_tokens * prices.text_input
        + usage.cached_text_tokens * prices.cached_text_input
        + usage.uncached_audio_tokens * prices.audio_input
        + usage.cached_audio_tokens * prices.cached_audio_input
        + usage.output_audio_tokens * prices.audio_output
        + usage.output_text_tokens * prices.text_output
    ) / 1_000_000
    return tokens + usage.live_seconds / 60.0 * prices.per_minute


def project_usage(*, seconds: float, turns: int, hosts: int, model: str | None = None) -> RealtimeUsage:
    """Project one act's usage from its target duration and turn count.

    Each agent hears everything *except* its own turns, so audio input scales
    with (hosts - 1) speakers' worth of the act rather than with hosts.

    A Live model (`is_live_model`) is projected on the other axis entirely: no
    tokens, and every host's session is open — and billed — for the whole act,
    listening included, so billable seconds are `seconds * hosts`. Wall clock
    runs longer than the audio it produces (each turn is generated before the
    others hear it), so this is a floor, which is why the dry-run output calls
    it an estimate.
    """
    if model is not None and is_live_model(model):
        return RealtimeUsage(live_seconds=seconds * hosts)
    audio_out = seconds * AUDIO_OUTPUT_TOKENS_PER_SECOND
    text_out = seconds * TEXT_OUTPUT_TOKENS_PER_SECOND
    listeners = max(0, hosts - 1)
    audio_in = seconds * AUDIO_INPUT_TOKENS_PER_SECOND * listeners
    text_in = turns * TEXT_TOKENS_PER_TURN
    return RealtimeUsage(
        input_tokens=int(audio_in + text_in),
        output_tokens=int(audio_out + text_out),
        input_audio_tokens=int(audio_in),
        input_text_tokens=int(text_in),
        cached_audio_tokens=int(audio_in * CACHE_HIT_RATE),
        cached_text_tokens=int(text_in * CACHE_HIT_RATE),
        output_audio_tokens=int(audio_out),
        output_text_tokens=int(text_out),
    )
