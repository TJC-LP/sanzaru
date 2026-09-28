"""Local pricing for models the estimate endpoint only describes in prose.

Seedance 2.5 is the default model and its estimate is a sentence, so a cap
would be unenforceable on the default without this table.
"""

import logging
import math

import pytest

from sanzaru.higgsfield.errors import CostCapExceededError, UnpricedVideoError
from sanzaru.higgsfield.pricing import (
    DIMENSIONS,
    cap_amount,
    enforce_cap,
    local_estimate,
    price_env_name,
    resolve_cost,
)

pytestmark = pytest.mark.unit


def _seconds_cost(w: int, h: int, seconds: float, rate: float = 0.0214) -> float:
    return math.ceil(w * h * seconds * 24 / 1024) / 1000 * rate


def test_published_per_second_rates():
    """The formula reproduces the rates Higgsfield states in the Seedance description."""
    assert local_estimate("seedance-2.5", "text", {"duration": 1}) == pytest.approx(0.4622, abs=1e-4)
    assert local_estimate("seedance-2.5", "text", {"duration": 1, "resolution": "480p"}) == pytest.approx(
        0.2056, abs=1e-4
    )


def test_five_seconds_at_720p_and_480p():
    assert local_estimate("seedance-2.5", "text", {"duration": 5}) == pytest.approx(2.3112, abs=1e-4)
    assert local_estimate("seedance-2.5", "text", {"duration": 5, "resolution": "480p"}) == pytest.approx(
        _seconds_cost(854, 480, 5), abs=1e-9
    )
    assert _seconds_cost(854, 480, 5) == pytest.approx(1.028, abs=1e-3)


def test_defaults_are_the_api_defaults():
    assert local_estimate("seedance-2.5", "text", {}) == local_estimate(
        "seedance-2.5", "text", {"duration": 5, "resolution": "720p", "aspect_ratio": "16:9"}
    )


def test_extend_bills_source_plus_added():
    got = local_estimate("seedance-2.5", "extend", {"duration": 5, "resolution": "480p"}, input_video_seconds=5.0)
    largest = max((wh for (res, _), wh in DIMENSIONS.items() if res == "480p"), key=lambda wh: wh[0] * wh[1])
    assert got == pytest.approx(_seconds_cost(*largest, 10) * 0.6, abs=1e-9)


def test_edit_bills_the_source_twice():
    got = local_estimate("seedance-2.5", "edit", {"resolution": "480p"}, input_video_seconds=6.0)
    largest = max((wh for (res, _), wh in DIMENSIONS.items() if res == "480p"), key=lambda wh: wh[0] * wh[1])
    assert got == pytest.approx(_seconds_cost(*largest, 12) * 0.6, abs=1e-9)


def test_video_input_rate_reproduces_the_published_figure():
    """edit/extend estimate text, 2026-09-28: $0.2773/s of input+generated at 720p 16:9 (0.6x)."""
    from sanzaru.higgsfield.pricing import PRICES

    pricing = PRICES["seedance-2.5"]
    per_second = math.ceil(1280 * 720 * 24 / 1024) / 1000 * pricing.usd_per_1k_tokens["720p"]
    assert per_second * pricing.video_input_factor == pytest.approx(0.2773, abs=1e-4)


def test_env_override_keeps_the_video_input_factor(monkeypatch):
    from sanzaru.higgsfield.pricing import price_env_name, prices_for

    monkeypatch.setenv(price_env_name("seedance-2.5"), "0.03")
    assert prices_for("seedance-2.5").video_input_factor == 0.6


@pytest.mark.parametrize("op", ["edit", "extend"])
def test_unknown_source_duration_is_unpriced(op):
    assert local_estimate("seedance-2.5", op, {"duration": 5}, input_video_seconds=None) is None


def test_image_is_priced_at_the_largest_frame():
    text_16x9 = local_estimate("seedance-2.5", "text", {"duration": 5})
    image = local_estimate("seedance-2.5", "image", {"duration": 5})
    assert image is not None and text_16x9 is not None
    assert image >= text_16x9


def test_unknown_family_is_unpriced():
    assert local_estimate("kling-3.0", "text", {"duration": 5}) is None


def test_env_override(monkeypatch):
    monkeypatch.setenv(price_env_name("seedance-2.5"), "0.0428")
    assert local_estimate("seedance-2.5", "text", {"duration": 1}) == pytest.approx(0.9245, abs=1e-4)


def test_env_override_two_tiers(monkeypatch):
    monkeypatch.setenv("SANZARU_HIGGSFIELD_PRICE_SEEDANCE_2_5", "0.01,0.02")
    assert local_estimate("seedance-2.5", "text", {"duration": 1, "resolution": "480p"}) == pytest.approx(
        _seconds_cost(854, 480, 1, 0.01)
    )


@pytest.mark.parametrize("bad", ["abc", "1,2,3", "-1"])
def test_malformed_override_warns_and_falls_back(monkeypatch, caplog, bad):
    monkeypatch.setenv("SANZARU_HIGGSFIELD_PRICE_SEEDANCE_2_5", bad)
    with caplog.at_level(logging.WARNING, logger="sanzaru"):
        got = local_estimate("seedance-2.5", "text", {"duration": 1})
    assert got == pytest.approx(0.4622, abs=1e-4)
    assert "not usable" in caplog.text


class TestResolveCost:
    def test_api_estimate(self):
        cost = resolve_cost(
            {
                "type": "estimate",
                "usd": "0.347",
                "credits": "5.544",
                "discount": {"percentage": "45", "credits": "4.5", "usd": "0.284"},
            },
            None,
        )
        assert cost["basis"] == "api"
        assert cost["usd"] == pytest.approx(0.347) and cost["credits"] == pytest.approx(5.544)
        assert cost["usd_after_discount"] == pytest.approx(0.284)
        assert cap_amount(cost) == pytest.approx(0.347)

    def test_description_with_local(self):
        cost = resolve_cost({"type": "description", "pricing_description": "roughly $0.46/s"}, 2.3112)
        assert cost["basis"] == "local_table"
        assert cost["usd"] == pytest.approx(2.3112)
        assert cost["pricing_description"] == "roughly $0.46/s"

    def test_description_without_local(self):
        cost = resolve_cost({"type": "description", "pricing_description": "per second"}, None)
        assert cost["basis"] == "unpriced" and cost["usd"] is None

    def test_estimate_failed(self):
        assert resolve_cost(None, None, estimate_failed=True)["basis"] == "unavailable"

    def test_estimate_failed_but_locally_priced(self):
        assert resolve_cost(None, 1.0, estimate_failed=True)["basis"] == "local_table"


class TestCap:
    def _cost(self, usd, after=None):
        return resolve_cost(
            {
                "type": "estimate",
                "usd": str(usd),
                "credits": "1",
                "discount": {"percentage": "1", "credits": "1", "usd": str(after)} if after is not None else None,
            },
            None,
        )

    def test_no_cap_never_refuses(self):
        enforce_cap(resolve_cost(None, None), None, "m")

    def test_under_cap(self):
        enforce_cap(self._cost(0.5), 1.0, "m")

    def test_over_cap(self):
        with pytest.raises(CostCapExceededError, match="Nothing was submitted") as info:
            enforce_cap(self._cost(2.0), 1.0, "m")
        assert info.value.estimate_usd == pytest.approx(2.0)
        assert info.value.limit_usd == 1.0

    def test_cap_uses_the_larger_figure(self):
        with pytest.raises(CostCapExceededError):
            enforce_cap(self._cost(0.9, after=1.2), 1.0, "m")

    def test_unpriced_with_cap_refuses(self):
        with pytest.raises(UnpricedVideoError, match="cannot be enforced"):
            enforce_cap(resolve_cost({"type": "description", "pricing_description": "x"}, None), 5.0, "m")

    def test_unpriced_is_a_cap_error(self):
        assert issubclass(UnpricedVideoError, CostCapExceededError)
        assert issubclass(CostCapExceededError, ValueError)
