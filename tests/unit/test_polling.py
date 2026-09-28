# SPDX-License-Identifier: MIT
"""Unit tests for the polling wait loops (no real sleeps — fake clock)."""

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError

from sanzaru.higgsfield.errors import ErrorKind, HiggsfieldAPIError
from sanzaru.polling import (
    WaitTimeoutError,
    _higgsfield_transient,
    _openai_transient,
    wait_for_image,
    wait_for_video,
    wait_until_done,
)


class FakeClock:
    """Deterministic anyio.sleep/current_time replacement recording each sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def current_time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(mocker):
    clock = FakeClock()
    mocker.patch("sanzaru.polling.anyio.sleep", clock.sleep)
    mocker.patch("sanzaru.polling.anyio.current_time", clock.current_time)
    mocker.patch("sanzaru.polling.random.uniform", return_value=0.0)  # no jitter
    return clock


HF = "hf_d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"


def _video(status: str) -> dict[str, object]:
    """A VideoStatus as tools.video.get_video_status returns it."""
    return {
        "id": HF,
        "status": status,
        "done": status in {"completed", "failed", "nsfw", "canceled"},
        "error": None,
        "video_url": None,
    }


def _api_error(status_code: int) -> APIStatusError:
    response = httpx2.Response(status_code, request=httpx2.Request("GET", "https://api.test"))
    return APIStatusError("boom", response=response, body=None)


def _hf_error(kind: ErrorKind, status_code: int | None) -> HiggsfieldAPIError:
    return HiggsfieldAPIError("boom", status_code=status_code, detail="boom", kind=kind)


@pytest.mark.unit
async def test_wait_for_video_returns_completed_without_sleeping(mocker, fake_clock):
    """A job already terminal on first fetch returns immediately."""
    done = _video("completed")
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=done))

    result = await wait_for_video(HF)

    assert result is done
    assert fake_clock.sleeps == []


@pytest.mark.unit
async def test_wait_for_video_polls_with_adaptive_backoff(mocker, fake_clock):
    """Higgsfield cadence: 2 s, growing x1.5; progress reported for every fetch."""
    states = [_video("queued"), _video("in_progress"), _video("completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))
    seen: list[object] = []

    result = await wait_for_video(HF, on_progress=lambda v: seen.append(v["status"]))

    assert result["status"] == "completed"
    assert seen == ["queued", "in_progress", "completed"]
    assert fake_clock.sleeps == [2.0, 3.0]


@pytest.mark.unit
async def test_wait_for_video_backoff_caps_at_max_interval(mocker, fake_clock):
    """Adaptive delay never exceeds Higgsfield's documented 10 s cap."""
    states = [_video("in_progress")] * 6 + [_video("completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))

    await wait_for_video(HF)

    assert fake_clock.sleeps == [2.0, 3.0, 4.5, 6.75, 10.0, 10.0]


@pytest.mark.unit
async def test_wait_for_video_fixed_interval_disables_backoff(mocker, fake_clock):
    """An explicit interval is used verbatim for every poll."""
    states = [_video("queued"), _video("in_progress"), _video("completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))

    await wait_for_video(HF, interval=3.0)

    assert fake_clock.sleeps == [3.0, 3.0]


@pytest.mark.unit
async def test_fixed_interval_ignores_jitter_but_adaptive_applies_it(mocker, fake_clock):
    """Jitter perturbs only the adaptive schedule — a fixed interval is exact."""
    mocker.patch("sanzaru.polling.random.uniform", return_value=0.1)  # override fixture's 0.0

    states = [_video("queued"), _video("completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))
    await wait_for_video(HF, interval=3.0)
    assert fake_clock.sleeps == [3.0]  # verbatim, no ±10%

    states = [_video("queued"), _video("completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))
    await wait_for_video(HF)
    assert fake_clock.sleeps[-1] == pytest.approx(2.0 * 1.1)  # adaptive start 2s +10% jitter


@pytest.mark.unit
@pytest.mark.parametrize("state", ["failed", "nsfw", "canceled"])
async def test_wait_for_video_returns_a_terminal_non_completed_job(mocker, fake_clock, state):
    """A server-side failure, moderation or cancel is a terminal result, not an exception."""
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=_video(state)))

    result = await wait_for_video(HF)

    assert result["status"] == state


@pytest.mark.unit
async def test_wait_for_video_timeout_carries_last_state(mocker, fake_clock):
    """Deadline expiry raises WaitTimeoutError holding the last-seen status dict."""
    stuck = _video("in_progress")
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=stuck))

    with pytest.raises(WaitTimeoutError) as excinfo:
        await wait_for_video(HF, timeout=4.0)

    assert excinfo.value.last == stuck
    assert HF in str(excinfo.value)
    # Sleeps are clamped to the remaining deadline: 2s, then min(3, 2)=2.
    assert fake_clock.sleeps == [2.0, 2.0]


@pytest.mark.unit
async def test_wait_for_video_retries_transient_errors(mocker, fake_clock):
    """Higgsfield server, transport and 429 errors are retried in place until the deadline."""
    done = _video("completed")
    mocker.patch(
        "sanzaru.tools.video.get_video_status",
        mocker.AsyncMock(side_effect=[_hf_error("server", 500), _hf_error("transport", None), done]),
    )

    result = await wait_for_video(HF)

    assert result is done
    assert len(fake_clock.sleeps) == 2


@pytest.mark.unit
async def test_wait_for_video_raises_non_retryable_immediately(mocker, fake_clock):
    """A 404 (unknown id) propagates without retrying."""
    fetch = mocker.AsyncMock(side_effect=_hf_error("not_found", 404))
    mocker.patch("sanzaru.tools.video.get_video_status", fetch)

    with pytest.raises(HiggsfieldAPIError):
        await wait_for_video(HF)

    assert fetch.call_count == 1
    assert fake_clock.sleeps == []


@pytest.mark.unit
async def test_wait_for_video_does_not_retry_openai_errors(mocker, fake_clock):
    """The video loop's predicate is Higgsfield's: an OpenAI 503 is not its business."""
    fetch = mocker.AsyncMock(side_effect=_api_error(503))
    mocker.patch("sanzaru.tools.video.get_video_status", fetch)

    with pytest.raises(APIStatusError):
        await wait_for_video(HF)

    assert fetch.call_count == 1


@pytest.mark.unit
async def test_wait_for_image_polls_until_terminal(mocker, fake_clock):
    """Image waits use the 2s starting interval and dict payloads."""
    states = [
        {"id": "resp_x", "status": "queued", "created_at": 1.0},
        {"id": "resp_x", "status": "unknown", "created_at": 1.0},
        {"id": "resp_x", "status": "completed", "created_at": 1.0},
    ]
    mocker.patch("sanzaru.tools.image.get_image_status", mocker.AsyncMock(side_effect=states))

    result = await wait_for_image("resp_x")

    assert result["status"] == "completed"
    assert fake_clock.sleeps == [2.0, 3.0]


@pytest.mark.unit
async def test_wait_for_image_retries_transient_openai_errors(mocker, fake_clock):
    """Connection errors and 5xx/429 are retried in place for OpenAI image jobs."""
    done = {"id": "resp_x", "status": "completed", "created_at": 1.0}
    mocker.patch(
        "sanzaru.tools.image.get_image_status",
        mocker.AsyncMock(
            side_effect=[
                APIConnectionError(request=httpx2.Request("GET", "https://api.test")),
                _api_error(500),
                _api_error(429),
                done,
            ]
        ),
    )

    result = await wait_for_image("resp_x")

    assert result == done
    assert len(fake_clock.sleeps) == 3


@pytest.mark.unit
async def test_wait_for_image_raises_non_retryable_immediately(mocker, fake_clock):
    fetch = mocker.AsyncMock(side_effect=_api_error(404))
    mocker.patch("sanzaru.tools.image.get_image_status", fetch)

    with pytest.raises(APIStatusError):
        await wait_for_image("resp_missing")

    assert fetch.call_count == 1


@pytest.mark.unit
async def test_wait_for_image_timeout_carries_last_state(mocker, fake_clock):
    """Image deadline expiry mirrors the video behavior."""
    stuck = {"id": "resp_x", "status": "in_progress", "created_at": 1.0}
    mocker.patch("sanzaru.tools.image.get_image_status", mocker.AsyncMock(return_value=stuck))

    with pytest.raises(WaitTimeoutError) as excinfo:
        await wait_for_image("resp_x", timeout=3.0)

    assert excinfo.value.last == stuck


# ==================== PROVIDER-AGNOSTIC LOOP ====================


class _FlakyError(Exception):
    pass


def _fetcher(*items):
    """Replays items in order: exceptions are raised, anything else returned."""
    queue = list(items)

    async def fetch():
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return fetch


def _done(status: dict) -> bool:
    return status["status"] in {"completed", "failed", "nsfw", "canceled"}


@pytest.mark.unit
async def test_is_transient_decides_what_is_retried(fake_clock):
    """A custom exception is retried when the predicate accepts it."""
    fetch = _fetcher(_FlakyError(), {"status": "completed"})

    result = await wait_until_done(
        fetch, _done, describe="job", timeout=60, is_transient=lambda exc: isinstance(exc, _FlakyError)
    )

    assert result == {"status": "completed"}
    assert fake_clock.sleeps == [2.0]


@pytest.mark.unit
async def test_is_transient_rejection_propagates(fake_clock):
    fetch = _fetcher(_FlakyError(), {"status": "completed"})

    with pytest.raises(_FlakyError):
        await wait_until_done(fetch, _done, describe="job", timeout=60, is_transient=lambda exc: False)
    assert fake_clock.sleeps == []


@pytest.mark.unit
async def test_higgsfield_server_errors_are_retried(fake_clock):
    fetch = _fetcher(_hf_error("server", 500), _hf_error("transport", None), {"status": "completed"})

    result = await wait_until_done(fetch, _done, describe="job", timeout=60)

    assert result["status"] == "completed"
    assert fake_clock.sleeps == [2.0, 3.0]


@pytest.mark.unit
async def test_higgsfield_validation_errors_propagate(fake_clock):
    fetch = _fetcher(_hf_error("not_found", 404), {"status": "completed"})

    with pytest.raises(HiggsfieldAPIError):
        await wait_until_done(fetch, _done, describe="job", timeout=60)


@pytest.mark.unit
def test_transient_predicates_match_their_provider():
    assert _openai_transient(_api_error(503))
    assert _openai_transient(_api_error(429))
    assert not _openai_transient(_api_error(404))
    assert not _openai_transient(_hf_error("server", 500))
    assert _higgsfield_transient(_hf_error("server", 502))
    assert _higgsfield_transient(_hf_error("other", 429))
    assert not _higgsfield_transient(_hf_error("validation", 422))
    assert not _higgsfield_transient(_api_error(503))


@pytest.mark.unit
async def test_wait_until_done_uses_the_higgsfield_cadence(fake_clock):
    """Dict statuses, 2s ×1.5 → 10s cap, progress on every fetch."""
    fetch = _fetcher(*([{"status": "in_progress"}] * 6), {"status": "completed"})
    seen: list[str] = []

    result = await wait_until_done(
        fetch, _done, describe="job", timeout=600, on_progress=lambda s: seen.append(s["status"])
    )

    assert result == {"status": "completed"}
    assert fake_clock.sleeps == [2.0, 3.0, 4.5, 6.75, 10.0, 10.0]
    assert seen == ["in_progress"] * 6 + ["completed"]


@pytest.mark.unit
async def test_wait_until_done_timeout_carries_the_last_dict(fake_clock):
    fetch = _fetcher(*([{"status": "queued"}] * 50))

    with pytest.raises(WaitTimeoutError) as info:
        await wait_until_done(fetch, _done, describe="Video job hf_x", timeout=5)

    assert info.value.last == {"status": "queued"}
    assert "hf_x" in str(info.value)
