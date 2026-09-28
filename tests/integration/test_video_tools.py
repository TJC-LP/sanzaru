"""Video tools on Higgsfield, over an httpx.MockTransport that routes by path.

The negative properties matter most: a job refused by the cost cap — or a dry
run — makes no upload and no submit, and a submit happens exactly once.
"""

import json
import struct

import httpx
import pytest

from sanzaru.config import set_higgsfield_client
from sanzaru.higgsfield.client import HiggsfieldClient
from sanzaru.higgsfield.errors import CostCapExceededError, UnpricedVideoError
from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools import video

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

RID = "d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"
JOB = f"hf_{RID}"
PRIOR = "hf_11111111-2222-4333-8444-555555555555"
KLING_ESTIMATE = {"type": "estimate", "credits": "5.544", "usd": "0.347", "discount": None}
SEEDANCE_ESTIMATE = {"type": "description", "pricing_description": "roughly $0.4622 per second at 720p"}


def _mp4(seconds: float, timescale: int = 1000) -> bytes:
    """Smallest MP4 our mvhd reader accepts: ftyp + moov/mvhd (version 0)."""
    mvhd_body = struct.pack(">B3xIIII", 0, 0, 0, timescale, int(seconds * timescale)) + b"\x00" * 80
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"isom\x00\x00\x02\x00"
    return ftyp + moov


def _moov_last(seconds: float, payload: int = 4000) -> bytes:
    """Higgsfield's output layout: ftyp, free, mdat, then moov."""
    mvhd_body = struct.pack(">B3xIIII", 0, 0, 0, 1000, int(seconds * 1000)) + b"\x00" * 80
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"isom\x00\x00\x02\x00"
    return ftyp + struct.pack(">I4s", 8, b"free") + struct.pack(">I4s", 8 + payload, b"mdat") + b"\x00" * payload + moov


MOOV_LAST_MP4 = _moov_last(5.0)


class FakeHiggsfield:
    """Routes requests by path and records them."""

    def __init__(self, *, estimate=None, status=None, submit_status=200):
        self.estimate = estimate if estimate is not None else KLING_ESTIMATE
        self.status = status or {"status": "completed", "request_id": RID, "video": {"url": "https://cdn.test/o.mp4"}}
        self.submit_status = submit_status
        self.requests: list[httpx.Request] = []
        self.uploads = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.url.host == "upload.test":
            return httpx.Response(200)
        if request.url.host == "cdn.test":
            rng = request.headers.get("Range")
            if rng:
                start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                return httpx.Response(206, content=MOOV_LAST_MP4[start : end + 1])
            return httpx.Response(200, content=b"MP4BYTES")
        if path.startswith("/estimate/"):
            return httpx.Response(200, json=self.estimate)
        if path == "/files/generate-upload-url":
            self.uploads += 1
            return httpx.Response(
                200,
                json={
                    "public_url": f"https://files.test/u{self.uploads}",
                    "upload_url": f"https://upload.test/u{self.uploads}",
                    "upload_headers": {"Content-Type": json.loads(request.content)["content_type"]},
                },
            )
        if path.endswith("/status"):
            return httpx.Response(200, json=self.status)
        if path.endswith("/cancel"):
            return httpx.Response(202) if self.submit_status == 200 else httpx.Response(400, json={"detail": "started"})
        return httpx.Response(self.submit_status, json={"status": "queued", "request_id": RID})

    def posts_to(self, prefix: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST" and r.url.path.startswith(prefix)]

    def submits(self) -> list[httpx.Request]:
        skip = ("/estimate/", "/files/", "/requests/")
        return [
            r
            for r in self.requests
            if r.method == "POST" and r.url.host == "api.higgsfield.ai" and not r.url.path.startswith(skip)
        ]

    def puts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "PUT"]


@pytest.fixture(autouse=True)
def _no_backoff(mocker):
    mocker.patch("sanzaru.higgsfield.client.anyio.sleep", new=mocker.AsyncMock())


@pytest.fixture
def fake():
    yield FakeHiggsfield()
    set_higgsfield_client(None)


@pytest.fixture
def install(fake):
    def go(handler: FakeHiggsfield | None = None) -> FakeHiggsfield:
        handler = handler or fake
        set_higgsfield_client(HiggsfieldClient("kid", "secret", transport=httpx.MockTransport(handler)))
        return handler

    return go


@pytest.fixture
def storage(tmp_path, mocker):
    (tmp_path / "videos").mkdir()
    (tmp_path / "images").mkdir()
    backend = LocalStorageBackend(path_overrides={"video": tmp_path / "videos", "reference": tmp_path / "images"})
    mocker.patch("sanzaru.tools.video.get_storage", return_value=backend)
    mocker.patch("sanzaru.higgsfield.media.get_storage", return_value=backend, create=True)
    return tmp_path


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


class TestCreateVideo:
    async def test_text_to_video_submits_the_translated_body_once(self, install):
        fake = install()
        job = await video.create_video("waves at dusk", model="kling-3.0-std", duration=5, audio=False)
        (submit,) = fake.submits()
        assert submit.url.path == "/kling-video/v3.0/std/text-to-video"
        assert _body(submit) == {"prompt": "waves at dusk", "duration": 5, "sound": "off"}
        assert job["id"] == JOB and job["status"] == "queued"
        assert job["cost"]["basis"] == "api" and job["cost"]["usd"] == pytest.approx(0.347)

    async def test_default_model_is_seedance_priced_locally(self, install):
        fake = install(FakeHiggsfield(estimate=SEEDANCE_ESTIMATE))
        job = await video.create_video("waves", duration=5, resolution="720p", aspect_ratio="16:9")
        assert fake.submits()[0].url.path == "/bytedance/seedance-2.5/text-to-video"
        assert job["cost"]["basis"] == "local_table"
        assert job["cost"]["usd"] == pytest.approx(2.311, abs=1e-3)

    async def test_image_to_video_uploads_the_reference_then_sends_its_url(self, install, storage):
        (storage / "images" / "cat.png").write_bytes(b"\x89PNG....")
        fake = install()
        job = await video.create_video("the cat stretches", model="kling-3.0-std", reference_image="cat.png")
        assert len(fake.puts()) == 1
        assert "Authorization" not in fake.puts()[0].headers
        body = _body(fake.submits()[0])
        assert fake.submits()[0].url.path == "/kling-video/v3.0/std/image-to-video"
        assert body["image_url"] == "https://files.test/u1"
        assert job["arguments"]["image_url"] == "<upload:cat.png>"

    async def test_end_frame_maps_to_the_models_parameter(self, install, storage):
        for name in ("a.png", "b.png"):
            (storage / "images" / name).write_bytes(b"\x89PNG")
        fake = install()
        await video.create_video("morph", model="kling-3.0-std", reference_image="a.png", end_image="b.png")
        body = _body(fake.submits()[0])
        assert set(body) >= {"image_url", "last_image_url"}

    async def test_end_image_without_a_start_frame_is_refused_before_any_request(self, install):
        fake = install()
        with pytest.raises(ValueError, match="end_image needs reference_image"):
            await video.create_video("x", end_image="b.png")
        assert fake.requests == []

    async def test_invalid_arguments_fail_before_any_request(self, install):
        fake = install()
        with pytest.raises(ValueError):
            await video.create_video("x", model="kling-3.0-std", duration=99)
        assert fake.requests == []

    async def test_a_prior_job_id_is_used_as_the_start_frame_without_upload(self, install):
        fake = install()
        await video.create_video("continue", model="kling-3.0-std", reference_image=PRIOR)
        assert fake.puts() == []
        assert _body(fake.submits()[0])["image_url"] == "https://cdn.test/o.mp4"


class TestCostCap:
    async def test_over_cap_refuses_with_no_upload_and_no_submit(self, install, storage):
        (storage / "images" / "cat.png").write_bytes(b"\x89PNG")
        fake = install()
        with pytest.raises(CostCapExceededError) as info:
            await video.create_video("x", model="kling-3.0-std", reference_image="cat.png", max_cost_usd=0.10)
        assert info.value.limit_usd == 0.10
        assert fake.puts() == [] and fake.submits() == [] and fake.uploads == 0

    async def test_unpriced_raw_slug_with_a_cap_refuses(self, install):
        fake = install(FakeHiggsfield(estimate={"type": "description", "pricing_description": "varies"}))
        with pytest.raises(UnpricedVideoError):
            await video.create_video("x", model="vendor/new-model/text-to-video", max_cost_usd=5)
        assert fake.submits() == []

    async def test_estimate_failure_without_a_cap_still_submits(self, install):
        class Down(FakeHiggsfield):
            def __call__(self, request):
                if request.url.path.startswith("/estimate/"):
                    self.requests.append(request)
                    return httpx.Response(500, json={"detail": "boom"})
                return super().__call__(request)

        fake = install(Down())
        job = await video.create_video("x", model="kling-3.0-std")
        assert job["cost"]["basis"] == "unavailable"
        assert len(fake.submits()) == 1

    async def test_estimate_failure_with_a_cap_refuses(self, install):
        class Down(FakeHiggsfield):
            def __call__(self, request):
                if request.url.path.startswith("/estimate/"):
                    self.requests.append(request)
                    return httpx.Response(500, json={"detail": "boom"})
                return super().__call__(request)

        fake = install(Down())
        with pytest.raises(UnpricedVideoError):
            await video.create_video("x", model="kling-3.0-std", max_cost_usd=1)
        assert fake.submits() == []

    async def test_a_rejected_estimate_is_a_usage_error(self, install):
        class Picky(FakeHiggsfield):
            def __call__(self, request):
                if request.url.path.startswith("/estimate/"):
                    self.requests.append(request)
                    return httpx.Response(400, json={"detail": "duration: 99 is greater than the maximum of 15"})
                return super().__call__(request)

        install(Picky())
        with pytest.raises(ValueError, match="rejected the arguments"):
            await video.create_video("x", model="vendor/m/text-to-video", duration=30)

    async def test_dry_run_prices_without_uploading_or_submitting(self, install, storage):
        (storage / "images" / "cat.png").write_bytes(b"\x89PNG")
        fake = install()
        job = await video.create_video("x", model="kling-3.0-std", reference_image="cat.png", dry_run=True)
        assert job["id"] is None and job["status"] == "dry_run"
        assert job["cost"]["usd"] == pytest.approx(0.347)
        assert fake.puts() == [] and fake.submits() == []


class TestEditExtend:
    async def test_extend_uploads_the_source_and_bills_its_duration(self, install, storage):
        (storage / "videos" / "clip.mp4").write_bytes(_mp4(5.0))
        fake = install(FakeHiggsfield(estimate=SEEDANCE_ESTIMATE))
        job = await video.extend_video("the waves crash", "clip.mp4", duration=5, resolution="480p")
        body = _body(fake.submits()[0])
        assert fake.submits()[0].url.path == "/bytedance/seedance-2.5/video-extend"
        assert body["video_url"] == "https://files.test/u1" and body["duration"] == 5
        assert job["cost"]["basis"] == "local_table"
        assert job["cost"]["usd"] is not None and job["cost"]["usd"] > 0

    async def test_edit_sends_no_duration(self, install, storage):
        (storage / "videos" / "clip.mp4").write_bytes(_mp4(4.0))
        fake = install(FakeHiggsfield(estimate=SEEDANCE_ESTIMATE))
        await video.edit_video("make it snow", "clip.mp4")
        body = _body(fake.submits()[0])
        assert "duration" not in body and body["prompt"] == "make it snow"

    async def test_a_prior_job_is_extended_without_download_or_upload(self, install):
        fake = install(FakeHiggsfield(estimate=SEEDANCE_ESTIMATE))
        await video.extend_video("more", PRIOR, duration=4)
        assert fake.puts() == []
        assert _body(fake.submits()[0])["video_url"] == "https://cdn.test/o.mp4"

    async def test_a_chained_extend_is_priced_from_ranged_reads_of_the_prior_output(self, install):
        """Live smoke 2026-09-28 found this refused as unpriced: the source duration was never read."""
        fake = install(FakeHiggsfield(estimate=SEEDANCE_ESTIMATE))
        job = await video.extend_video("more", PRIOR, duration=4, resolution="480p", max_cost_usd=5)
        assert job["cost"]["basis"] == "local_table" and job["cost"]["usd"] is not None
        ranged = [r for r in fake.requests if r.url.host == "cdn.test"]
        assert ranged and all("Range" in r.headers for r in ranged)
        assert fake.puts() == [] and len(fake.submits()) == 1

    async def test_extend_cap_uses_the_local_price_before_uploading(self, install, storage):
        (storage / "videos" / "clip.mp4").write_bytes(_mp4(10.0))
        fake = install(FakeHiggsfield(estimate=SEEDANCE_ESTIMATE))
        with pytest.raises(CostCapExceededError):
            await video.extend_video("more", "clip.mp4", duration=10, max_cost_usd=0.5)
        assert fake.uploads == 0 and fake.submits() == []

    async def test_non_mp4_sources_are_refused(self, install, storage):
        (storage / "videos" / "clip.webm").write_bytes(b"webm")
        install()
        with pytest.raises(ValueError, match="(?i)mp4"):
            await video.extend_video("more", "clip.webm")

    async def test_raw_slugs_cannot_edit(self, install):
        install()
        with pytest.raises(ValueError, match="curated model"):
            await video.edit_video("x", "clip.mp4", model="vendor/m/video-edit")


class TestStatusDownloadCancel:
    @pytest.mark.parametrize(
        ("state", "done"),
        [("queued", False), ("in_progress", False), ("completed", True), ("failed", True), ("canceled", True)],
    )
    async def test_status_reports_terminal_states(self, install, state, done):
        install(FakeHiggsfield(status={"status": state, "request_id": RID}))
        status = await video.get_video_status(JOB)
        assert status["status"] == state and status["done"] is done and status["id"] == JOB

    async def test_nsfw_carries_an_explanation(self, install):
        install(FakeHiggsfield(status={"status": "nsfw", "request_id": RID}))
        status = await video.get_video_status(JOB)
        assert status["done"] and "moderation" in (status["error"] or "")

    async def test_a_sora_id_is_explained(self, install):
        install()
        with pytest.raises(ValueError, match="Sora"):
            await video.get_video_status("video_abc123")

    async def test_download_streams_the_output_into_storage(self, install, storage):
        fake = install()
        result = await video.download_video(JOB)
        assert result == {"filename": f"{JOB}.mp4", "format": "mp4"}
        assert (storage / "videos" / f"{JOB}.mp4").read_bytes() == b"MP4BYTES"
        cdn = [r for r in fake.requests if r.url.host == "cdn.test"]
        assert cdn and "Authorization" not in cdn[0].headers

    async def test_download_of_an_unfinished_job_fails(self, install, storage):
        install(FakeHiggsfield(status={"status": "in_progress", "request_id": RID}))
        with pytest.raises(ValueError, match="not completed"):
            await video.download_video(JOB)

    async def test_cancel_a_queued_job(self, install):
        install()
        assert await video.cancel_video(JOB) == {"id": JOB, "canceled": True}

    async def test_cancel_a_started_job_explains(self, install):
        install(FakeHiggsfield(submit_status=400))
        with pytest.raises(ValueError, match="only cancels queued"):
            await video.cancel_video(JOB)


class TestListLocalVideos:
    async def test_lists_the_video_directory(self, storage):
        (storage / "videos" / "a.mp4").write_bytes(b"x")
        (storage / "videos" / "b.mov").write_bytes(b"yy")
        result = await video.list_local_videos(sort_by="name", order="asc")
        assert [v["filename"] for v in result["data"]] == ["a.mp4", "b.mov"]
