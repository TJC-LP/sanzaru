# SPDX-License-Identifier: MIT
"""Unit tests for the polling wait loops (no real sleeps — fake clock)."""

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from sanzaru.polling import WaitTimeoutError, wait_for_image, wait_for_video


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


def _video(mocker, status: str, progress: int = 0):
    video = mocker.MagicMock()
    video.status = status
    video.progress = progress
    return video


def _api_error(status_code: int) -> APIStatusError:
    response = httpx.Response(status_code, request=httpx.Request("GET", "https://api.test"))
    return APIStatusError("boom", response=response, body=None)


@pytest.mark.unit
async def test_wait_for_video_returns_completed_without_sleeping(mocker, fake_clock):
    """A job already terminal on first fetch returns immediately."""
    done = _video(mocker, "completed", 100)
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=done))

    result = await wait_for_video("video_x")

    assert result is done
    assert fake_clock.sleeps == []


@pytest.mark.unit
async def test_wait_for_video_polls_with_adaptive_backoff(mocker, fake_clock):
    """Interval grows ×1.5 per poll and progress is reported for every fetch."""
    states = [_video(mocker, "queued"), _video(mocker, "in_progress", 40), _video(mocker, "completed", 100)]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))
    seen: list[str] = []

    result = await wait_for_video("video_x", on_progress=lambda v: seen.append(v.status))

    assert result.status == "completed"
    assert seen == ["queued", "in_progress", "completed"]
    assert fake_clock.sleeps == [5.0, 7.5]


@pytest.mark.unit
async def test_wait_for_video_backoff_caps_at_max_interval(mocker, fake_clock):
    """Adaptive delay never exceeds the 20s video cap."""
    states = [_video(mocker, "in_progress")] * 6 + [_video(mocker, "completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))

    await wait_for_video("video_x")

    assert fake_clock.sleeps == [5.0, 7.5, 11.25, 16.875, 20.0, 20.0]


@pytest.mark.unit
async def test_wait_for_video_fixed_interval_disables_backoff(mocker, fake_clock):
    """An explicit interval is used verbatim for every poll."""
    states = [_video(mocker, "queued"), _video(mocker, "in_progress"), _video(mocker, "completed")]
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=states))

    await wait_for_video("video_x", interval=3.0)

    assert fake_clock.sleeps == [3.0, 3.0]


@pytest.mark.unit
async def test_wait_for_video_returns_failed_job(mocker, fake_clock):
    """A server-side failure is a terminal result, not an exception."""
    failed = _video(mocker, "failed")
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=failed))

    result = await wait_for_video("video_x")

    assert result.status == "failed"


@pytest.mark.unit
async def test_wait_for_video_timeout_carries_last_state(mocker, fake_clock):
    """Deadline expiry raises WaitTimeoutError holding the last-seen payload."""
    stuck = _video(mocker, "in_progress", 78)
    mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=stuck))

    with pytest.raises(WaitTimeoutError) as excinfo:
        await wait_for_video("video_x", timeout=12.0)

    assert excinfo.value.last is stuck
    assert "video_x" in str(excinfo.value)
    # Sleeps are clamped to the remaining deadline: 5s, then min(7.5, 7)=7.
    assert fake_clock.sleeps == [5.0, 7.0]


@pytest.mark.unit
async def test_wait_for_video_retries_transient_errors(mocker, fake_clock):
    """Connection errors and 5xx/429 are retried in place until the deadline."""
    done = _video(mocker, "completed")
    mocker.patch(
        "sanzaru.tools.video.get_video_status",
        mocker.AsyncMock(
            side_effect=[
                APIConnectionError(request=httpx.Request("GET", "https://api.test")),
                _api_error(500),
                _api_error(429),
                done,
            ]
        ),
    )

    result = await wait_for_video("video_x")

    assert result is done
    assert len(fake_clock.sleeps) == 3


@pytest.mark.unit
async def test_wait_for_video_raises_non_retryable_immediately(mocker, fake_clock):
    """A 404 (unknown ID) propagates without retrying."""
    fetch = mocker.AsyncMock(side_effect=_api_error(404))
    mocker.patch("sanzaru.tools.video.get_video_status", fetch)

    with pytest.raises(APIStatusError):
        await wait_for_video("video_missing")

    assert fetch.call_count == 1
    assert fake_clock.sleeps == []


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
async def test_wait_for_image_timeout_carries_last_state(mocker, fake_clock):
    """Image deadline expiry mirrors the video behavior."""
    stuck = {"id": "resp_x", "status": "in_progress", "created_at": 1.0}
    mocker.patch("sanzaru.tools.image.get_image_status", mocker.AsyncMock(return_value=stuck))

    with pytest.raises(WaitTimeoutError) as excinfo:
        await wait_for_image("resp_x", timeout=3.0)

    assert excinfo.value.last == stuck
