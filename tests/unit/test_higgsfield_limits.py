"""Local bounds on Higgsfield traffic (per-call limiter, upload ceiling)."""

import logging

import pytest

from sanzaru.higgsfield.limits import make_limiter, max_upload_bytes

pytestmark = pytest.mark.unit


def test_default_limiter(monkeypatch):
    monkeypatch.delenv("SANZARU_HIGGSFIELD_MAX_CONCURRENCY", raising=False)
    assert make_limiter().total_tokens == 4


def test_env_limiter(monkeypatch):
    monkeypatch.setenv("SANZARU_HIGGSFIELD_MAX_CONCURRENCY", "2")
    assert make_limiter().total_tokens == 2


@pytest.mark.parametrize("bad", ["0", "-3", "many"])
def test_bad_limiter_falls_back(monkeypatch, caplog, bad):
    monkeypatch.setenv("SANZARU_HIGGSFIELD_MAX_CONCURRENCY", bad)
    with caplog.at_level(logging.WARNING, logger="sanzaru"):
        assert make_limiter().total_tokens == 4
    assert "not a positive integer" in caplog.text


def test_each_call_gets_a_fresh_limiter():
    assert make_limiter() is not make_limiter()


def test_upload_ceiling(monkeypatch):
    monkeypatch.delenv("SANZARU_HIGGSFIELD_MAX_UPLOAD_MB", raising=False)
    assert max_upload_bytes() == 200 * 1024 * 1024
    monkeypatch.setenv("SANZARU_HIGGSFIELD_MAX_UPLOAD_MB", "10")
    assert max_upload_bytes() == 10 * 1024 * 1024
