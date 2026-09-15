"""wait_for: the server-side wait that replaces model-driven polling.

The polling loops themselves are covered in tests/unit/test_polling.py; these
pin what the tool adds on top — mixed batches, the deadline that returns instead
of raising, per-job errors, downloads, and progress on every poll — and that the
MCP registration hides the injected Context from the schema.
"""

import importlib
import json
import sys

import httpx2
import pytest
from openai import APIStatusError

from sanzaru.tools import wait as wait_tools
from sanzaru.tools.wait import DEFAULT_WAIT_TIMEOUT, MAX_WAIT_IDS, MAX_WAIT_TIMEOUT, job_kind, wait_for


class FakeClock:
    """Drives polling.py's deadline and sleeps without real time passing."""

    def __init__(self) -> None:
        self.now = 0.0

    def current_time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(mocker):
    clock = FakeClock()
    mocker.patch("sanzaru.polling.anyio.sleep", clock.sleep)
    mocker.patch("sanzaru.polling.anyio.current_time", clock.current_time)
    mocker.patch("sanzaru.polling.random.uniform", return_value=0.0)
    return clock


def _video(mocker, status: str, progress: int = 0):
    video = mocker.MagicMock()
    video.status = status
    video.progress = progress
    return video


def _image(status: str) -> dict[str, object]:
    return {"id": "resp_1", "status": status, "created_at": 0.0}


def _api_error(status_code: int) -> APIStatusError:
    response = httpx2.Response(status_code, request=httpx2.Request("GET", "https://api.test"))
    return APIStatusError("boom", response=response, body=None)


@pytest.mark.unit
class TestJobKind:
    def test_prefixes(self):
        assert job_kind("video_abc") == "video"
        assert job_kind("resp_abc") == "image"

    @pytest.mark.parametrize("bad", ["abc", "vid_1", "response_1", ""])
    def test_unknown_prefix_is_a_usage_error(self, bad):
        with pytest.raises(ValueError, match="cannot infer job type"):
            job_kind(bad)


@pytest.mark.integration
class TestValidation:
    async def test_rejects_before_any_request(self, mocker):
        status = mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock())
        with pytest.raises(ValueError, match="at least one"):
            await wait_for([])
        with pytest.raises(ValueError, match="unique"):
            await wait_for(["video_a", "video_a"])
        with pytest.raises(ValueError, match="at most"):
            await wait_for([f"video_{i}" for i in range(MAX_WAIT_IDS + 1)])
        with pytest.raises(ValueError, match="timeout must be"):
            await wait_for(["video_a"], timeout=MAX_WAIT_TIMEOUT + 1)
        with pytest.raises(ValueError, match="cannot infer job type"):
            await wait_for(["video_a", "nope"])
        status.assert_not_called()

    def test_default_deadline_is_under_the_client_idle_window(self):
        assert DEFAULT_WAIT_TIMEOUT < 300


@pytest.mark.integration
class TestMixedBatch:
    async def test_waits_on_video_and_image_together_and_keeps_input_order(self, mocker, clock):
        mocker.patch(
            "sanzaru.tools.video.get_video_status",
            mocker.AsyncMock(side_effect=[_video(mocker, "in_progress", 40), _video(mocker, "completed", 100)]),
        )
        mocker.patch(
            "sanzaru.tools.image.get_image_status",
            mocker.AsyncMock(side_effect=[_image("in_progress"), _image("completed")]),
        )

        result = await wait_for(["resp_1", "video_1"])

        assert [job["id"] for job in result["jobs"]] == ["resp_1", "video_1"]
        assert result["all_done"] is True
        assert result["timed_out"] is False
        image, video = result["jobs"]
        assert image["kind"] == "image" and image["status"] == "completed" and image["progress"] is None
        assert video["kind"] == "video" and video["status"] == "completed" and video["progress"] == 100

    async def test_a_failed_job_is_done_not_an_error(self, mocker, clock):
        mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=_video(mocker, "failed")))

        (job,) = (await wait_for(["video_1"]))["jobs"]

        assert job["done"] is True
        assert job["status"] == "failed"
        assert job["error"] is None


@pytest.mark.integration
class TestDeadline:
    async def test_the_deadline_returns_with_the_last_seen_status(self, mocker, clock):
        mocker.patch(
            "sanzaru.tools.video.get_video_status",
            mocker.AsyncMock(return_value=_video(mocker, "in_progress", 55)),
        )

        result = await wait_for(["video_1"], timeout=30)

        (job,) = result["jobs"]
        assert result["timed_out"] is True
        assert result["all_done"] is False
        assert job["timed_out"] is True and job["done"] is False
        assert job["status"] == "in_progress" and job["progress"] == 55
        assert result["timeout_s"] == 30

    async def test_a_job_that_never_answered_reports_unknown(self, mocker, clock):
        mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=_api_error(503)))

        (job,) = (await wait_for(["video_1"], timeout=10))["jobs"]

        assert job["timed_out"] is True
        assert job["status"] == "unknown"


@pytest.mark.integration
class TestPerJobErrors:
    async def test_an_unknown_id_fails_its_own_job_only(self, mocker, clock):
        mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(side_effect=_api_error(404)))
        mocker.patch("sanzaru.tools.image.get_image_status", mocker.AsyncMock(return_value=_image("completed")))

        result = await wait_for(["video_missing", "resp_1"])

        missing, image = result["jobs"]
        assert missing["done"] is True
        assert missing["error"] is not None and "404" in missing["error"]
        assert image["done"] is True and image["error"] is None
        assert result["all_done"] is True


@pytest.mark.integration
class TestDownload:
    async def test_completed_jobs_are_downloaded_in_the_same_call(self, mocker, clock):
        mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=_video(mocker, "completed")))
        mocker.patch("sanzaru.tools.image.get_image_status", mocker.AsyncMock(return_value=_image("completed")))
        video_dl = mocker.patch(
            "sanzaru.tools.video.download_video",
            mocker.AsyncMock(return_value={"filename": "v.mp4", "variant": "video"}),
        )
        image_dl = mocker.patch(
            "sanzaru.tools.image.download_image",
            mocker.AsyncMock(return_value={"filename": "i.png", "size": (1, 1), "format": "PNG"}),
        )

        result = await wait_for(["video_1", "resp_1"], download=True)

        video, image = result["jobs"]
        assert video["download"] == {"filename": "v.mp4", "variant": "video"}
        assert image["download"] == {"filename": "i.png", "size": (1, 1), "format": "PNG"}
        video_dl.assert_awaited_once_with("video_1")
        image_dl.assert_awaited_once_with("resp_1")

    async def test_failed_and_timed_out_jobs_are_not_downloaded(self, mocker, clock):
        mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=_video(mocker, "failed")))
        mocker.patch("sanzaru.tools.image.get_image_status", mocker.AsyncMock(return_value=_image("in_progress")))
        video_dl = mocker.patch("sanzaru.tools.video.download_video", mocker.AsyncMock())
        image_dl = mocker.patch("sanzaru.tools.image.download_image", mocker.AsyncMock())

        result = await wait_for(["video_1", "resp_1"], download=True, timeout=5)

        assert all(job["download"] is None for job in result["jobs"])
        video_dl.assert_not_called()
        image_dl.assert_not_called()


@pytest.mark.integration
class TestProgress:
    async def test_progress_is_reported_on_every_poll_and_on_completion(self, mocker, clock):
        mocker.patch(
            "sanzaru.tools.video.get_video_status",
            mocker.AsyncMock(
                side_effect=[
                    _video(mocker, "queued", 0),
                    _video(mocker, "in_progress", 50),
                    _video(mocker, "completed", 100),
                ]
            ),
        )
        seen: list[tuple[int, int, str]] = []

        async def on_progress(settled: int, total: int, message: str) -> None:
            seen.append((settled, total, message))

        await wait_for(["video_1"], on_progress=on_progress)

        messages = [m for _, _, m in seen]
        assert "video_1: queued 0%" in messages
        assert "video_1: in_progress 50%" in messages
        assert messages[-1] == "video_1: completed 100%"
        assert seen[-1][:2] == (1, 1)
        assert all(total == 1 for _, total, _ in seen)

    async def test_a_failing_progress_callback_does_not_end_the_wait(self, mocker, clock):
        mocker.patch("sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=_video(mocker, "completed")))

        async def on_progress(settled: int, total: int, message: str) -> None:
            raise RuntimeError("client went away")

        result = await wait_for(["video_1"], on_progress=on_progress)

        assert result["all_done"] is True


@pytest.mark.integration
class TestMcpRegistration:
    @pytest.fixture
    async def schemas(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
        mcp = importlib.reload(importlib.import_module("sanzaru.server")).mcp
        try:
            yield {tool.name: tool.input_schema for tool in await mcp.list_tools()}
        finally:
            sys.modules.pop("sanzaru.server", None)

    async def test_wait_for_is_registered_without_leaking_the_context_parameter(self, schemas):
        schema = schemas["wait_for"]
        assert set(schema["properties"]) == {"ids", "timeout", "download"}
        assert schema["required"] == ["ids"]
        assert schema["properties"]["timeout"]["default"] == DEFAULT_WAIT_TIMEOUT
        assert schema["properties"]["download"]["default"] is False

    async def test_wait_for_forwards_client_progress(self, monkeypatch, tmp_path, mocker):
        """The MCP wrapper reports each poll through the request Context."""
        monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
        server = importlib.reload(importlib.import_module("sanzaru.server"))
        try:
            mocker.patch("sanzaru.polling.anyio.sleep", mocker.AsyncMock())
            mocker.patch(
                "sanzaru.tools.video.get_video_status",
                mocker.AsyncMock(side_effect=[_video(mocker, "in_progress", 10), _video(mocker, "completed", 100)]),
            )
            ctx = mocker.MagicMock()
            ctx.report_progress = mocker.AsyncMock()

            result = await server.mcp.call_tool("wait_for", {"ids": ["video_1"]}, ctx)

            payload = result.structured_content or json.loads(result.content[0].text)
            assert payload["all_done"] is True
            messages = [call.args[2] for call in ctx.report_progress.await_args_list]
            assert "video_1: in_progress 10%" in messages
            assert messages[-1] == "video_1: completed 100%"
        finally:
            sys.modules.pop("sanzaru.server", None)


@pytest.mark.unit
def test_module_exports_are_the_documented_bounds():
    assert wait_tools.MAX_WAIT_IDS == 20
    assert wait_tools.MAX_WAIT_TIMEOUT == 1800.0
