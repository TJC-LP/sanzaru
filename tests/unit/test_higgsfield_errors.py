"""Status code and detail → typed error, because the API bends HTTP semantics."""

import pytest

from sanzaru.higgsfield.errors import (
    CONCURRENCY_LIMIT,
    CostCapExceededError,
    HiggsfieldAPIError,
    HiggsfieldConcurrencyError,
    UnpricedVideoError,
    error_from_response,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "invalid"),
        (401, "auth"),
        (403, "credits"),
        (404, "not_found"),
        (422, "validation"),
        (423, "blocked"),
        (500, "server"),
        (502, "server"),
        (503, "disabled"),
        (418, "other"),
    ],
)
def test_status_maps_to_kind(status, kind):
    err = error_from_response(status, "boom")
    assert err.kind == kind
    assert err.status_code == status


def test_the_concurrency_cap_arrives_as_a_400_and_is_typed():
    err = error_from_response(400, "Maximum number of concurrent requests (4) has been reached")
    assert isinstance(err, HiggsfieldConcurrencyError)
    assert err.kind == "concurrency"
    assert str(CONCURRENCY_LIMIT) in str(err)
    assert "nothing was charged" in str(err)


def test_an_ordinary_400_is_not_mistaken_for_concurrency():
    assert not isinstance(error_from_response(400, "duration must be <= 30"), HiggsfieldConcurrencyError)


@pytest.mark.parametrize(
    ("status", "transient"),
    [(500, True), (502, True), (503, True), (429, True), (400, False), (403, False), (423, False)],
)
def test_transient_is_for_reads_that_may_succeed_later(status, transient):
    assert error_from_response(status, "x").transient is transient


def test_transport_errors_are_transient_but_an_ambiguous_submit_is_not():
    transport = HiggsfieldAPIError("x", status_code=None, detail="", kind="transport")
    ambiguous = HiggsfieldAPIError("x", status_code=None, detail="", kind="ambiguous_submit")
    assert transport.transient
    assert not ambiguous.transient


def test_a_long_detail_is_truncated():
    assert len(error_from_response(400, "x" * 5000).detail) == 500


def test_cost_errors_are_value_errors_carrying_the_numbers():
    err = UnpricedVideoError("no price", estimate_usd=None, limit_usd=1.0, basis="unpriced", model="m")
    assert isinstance(err, CostCapExceededError)
    assert isinstance(err, ValueError)
    assert (err.limit_usd, err.basis, err.model) == (1.0, "unpriced", "m")
