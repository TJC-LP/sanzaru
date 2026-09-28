"""HiggsfieldClient over httpx.MockTransport.

The properties that matter most are negative ones: the generation POST is never
resent once it may have reached the server (a resend can bill twice), and the
credential never travels to the presigned-upload or output hosts.
"""

import json

import httpx
import pytest

from sanzaru.higgsfield.client import BASE_URL, HiggsfieldClient
from sanzaru.higgsfield.errors import HiggsfieldAPIError, HiggsfieldConcurrencyError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

SLUG = "bytedance/seedance-2.5/text-to-video"
RID = "d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"
AUTH = "Key kid:secret"


class Recorder:
    """A MockTransport handler that replays queued responses and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if callable(nxt):
            return nxt(request)
        return nxt

    def client(self) -> HiggsfieldClient:
        return HiggsfieldClient("kid", "secret", transport=httpx.MockTransport(self))


def ok(body, status=200):
    return httpx.Response(status, json=body)


@pytest.fixture(autouse=True)
def _no_backoff(mocker):
    mocker.patch("sanzaru.higgsfield.client.anyio.sleep", new=mocker.AsyncMock())


class TestSubmit:
    async def test_posts_the_arguments_to_the_slug_with_the_key(self):
        rec = Recorder(ok({"status": "queued", "request_id": RID}))
        result = await rec.client().submit(SLUG, {"prompt": "waves", "duration": 5})
        (req,) = rec.requests
        assert req.method == "POST"
        assert str(req.url) == f"{BASE_URL}/{SLUG}"
        assert req.headers["Authorization"] == AUTH
        assert json.loads(req.content) == {"prompt": "waves", "duration": 5}
        assert result["request_id"] == RID

    @pytest.mark.parametrize("status", [500, 502, 429])
    async def test_an_http_error_is_never_resent(self, status):
        rec = Recorder(httpx.Response(status, json={"detail": "busy"}), ok({"request_id": RID}))
        with pytest.raises(HiggsfieldAPIError):
            await rec.client().submit(SLUG, {"prompt": "x"})
        assert len(rec.requests) == 1

    async def test_a_timeout_after_sending_is_ambiguous_and_not_resent(self):
        rec = Recorder(httpx.ReadTimeout("slow"), ok({"request_id": RID}))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().submit(SLUG, {"prompt": "x"})
        assert excinfo.value.kind == "ambiguous_submit"
        assert "may have been created" in str(excinfo.value)
        assert not excinfo.value.transient
        assert len(rec.requests) == 1

    async def test_a_connect_error_is_retried_once(self):
        rec = Recorder(httpx.ConnectError("refused"), ok({"status": "queued", "request_id": RID}))
        assert (await rec.client().submit(SLUG, {"prompt": "x"}))["request_id"] == RID
        assert len(rec.requests) == 2

    async def test_two_connect_errors_surface_as_transport(self):
        rec = Recorder(httpx.ConnectError("refused"), httpx.ConnectError("refused"))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().submit(SLUG, {"prompt": "x"})
        assert excinfo.value.kind == "transport"
        assert len(rec.requests) == 2

    async def test_the_concurrency_cap_is_typed(self):
        rec = Recorder(
            httpx.Response(400, json={"detail": "Maximum number of concurrent requests (4) has been reached"})
        )
        with pytest.raises(HiggsfieldConcurrencyError):
            await rec.client().submit(SLUG, {"prompt": "x"})

    async def test_a_hostile_slug_never_reaches_the_network(self):
        rec = Recorder()
        with pytest.raises(ValueError):
            await rec.client().submit("requests/x/cancel", {})
        assert rec.requests == []

    async def test_a_response_without_request_id_is_an_error(self):
        rec = Recorder(ok({"status": "queued"}))
        with pytest.raises(HiggsfieldAPIError, match="no request_id"):
            await rec.client().submit(SLUG, {"prompt": "x"})


class TestStatusAndCancel:
    async def test_status_retries_a_503_then_succeeds(self):
        rec = Recorder(
            httpx.Response(503, json={"detail": "not ready"}),
            ok({"status": "completed", "request_id": RID, "video": {"url": "https://cdn.example/v.mp4"}}),
        )
        status = await rec.client().status(RID)
        assert status["video"]["url"] == "https://cdn.example/v.mp4"
        assert len(rec.requests) == 2
        assert str(rec.requests[0].url) == f"{BASE_URL}/requests/{RID}/status"

    async def test_status_gives_up_after_three_attempts(self):
        rec = Recorder(*(httpx.Response(500, json={"detail": "down"}) for _ in range(3)))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().status(RID)
        assert excinfo.value.kind == "server"
        assert len(rec.requests) == 3

    async def test_a_404_is_not_retried(self):
        rec = Recorder(httpx.Response(404, json={"detail": "Request not found"}))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().status(RID)
        assert excinfo.value.kind == "not_found"
        assert len(rec.requests) == 1

    @pytest.mark.parametrize(
        ("status", "kind"),
        [(401, "auth"), (403, "credits"), (404, "not_found"), (422, "validation"), (423, "blocked")],
    )
    async def test_status_codes_map_to_kinds(self, status, kind):
        rec = Recorder(httpx.Response(status, json={"detail": "x"}))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().status(RID)
        assert excinfo.value.kind == kind

    async def test_a_disabled_model_is_retried_then_reported(self):
        rec = Recorder(*(httpx.Response(503, json={"detail": "disabled"}) for _ in range(3)))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().status(RID)
        assert excinfo.value.kind == "disabled"

    async def test_non_json_status_is_an_error(self):
        rec = Recorder(httpx.Response(200, text="<html>gateway</html>"))
        with pytest.raises(HiggsfieldAPIError, match="unexpected status"):
            await rec.client().status(RID)

    async def test_cancel_posts_to_the_cancel_url(self):
        rec = Recorder(httpx.Response(202, json={}))
        await rec.client().cancel(RID)
        assert rec.requests[0].method == "POST"
        assert str(rec.requests[0].url) == f"{BASE_URL}/requests/{RID}/cancel"

    async def test_cancelling_a_started_job_is_a_400(self):
        rec = Recorder(httpx.Response(400, json={"detail": "already processing"}))
        with pytest.raises(HiggsfieldAPIError) as excinfo:
            await rec.client().cancel(RID)
        assert excinfo.value.status_code == 400


class TestEstimateAndModels:
    async def test_numeric_estimate(self):
        body = {"type": "estimate", "credits": "5.544", "usd": "0.347", "discount": None}
        rec = Recorder(ok(body))
        assert await rec.client().estimate("kling-video/v3.0/std/text-to-video", {"prompt": "x"}) == body
        assert str(rec.requests[0].url) == f"{BASE_URL}/estimate/kling-video/v3.0/std/text-to-video"

    async def test_description_only_estimate(self):
        body = {"type": "description", "pricing_description": "roughly $0.46 per second at 720p"}
        rec = Recorder(ok(body))
        assert (await rec.client().estimate(SLUG, {"prompt": "x"}))["type"] == "description"

    async def test_models_returns_items(self):
        rec = Recorder(ok({"total": 1, "items": [{"slug": SLUG, "title": "Seedance", "output_type": "video"}]}))
        assert [m["slug"] for m in await rec.client().models()] == [SLUG]


class TestUpload:
    async def test_presigned_flow_sends_the_grant_headers_and_no_key(self):
        grant = {
            "public_url": "https://cdn.example/u/cat.png",
            "upload_url": "https://storage.example/put?sig=1",
            "upload_headers": {"Content-Type": "image/png", "x-amz-acl": "public-read"},
        }
        rec = Recorder(ok(grant), httpx.Response(200))
        url = await rec.client().upload(b"PNGDATA", "image/png")
        grant_req, put_req = rec.requests
        assert url == grant["public_url"]
        assert json.loads(grant_req.content) == {"content_type": "image/png"}
        assert grant_req.headers["Authorization"] == AUTH
        assert put_req.method == "PUT"
        assert str(put_req.url) == grant["upload_url"]
        assert put_req.content == b"PNGDATA"
        assert put_req.headers["x-amz-acl"] == "public-read"
        assert "authorization" not in put_req.headers

    @pytest.mark.parametrize("field", ["upload_url", "public_url"])
    async def test_a_non_https_url_is_refused(self, field):
        grant = {"public_url": "https://cdn.example/x", "upload_url": "https://s.example/put", "upload_headers": {}}
        grant[field] = "http://evil.example/x"
        rec = Recorder(ok(grant))
        with pytest.raises(HiggsfieldAPIError, match="non-https"):
            await rec.client().upload(b"x", "image/png")
        assert len(rec.requests) == 1


class TestStreamOutput:
    async def test_streams_without_the_key(self):
        rec = Recorder(httpx.Response(200, content=b"MP4BYTES"))
        async with rec.client().stream_output("https://cdn.example/v.mp4") as chunks:
            data = b"".join([c async for c in chunks])
        assert data == b"MP4BYTES"
        assert "authorization" not in rec.requests[0].headers

    @pytest.mark.parametrize("status", [403, 404, 410])
    async def test_an_expired_output_says_so(self, status):
        rec = Recorder(httpx.Response(status))
        with pytest.raises(HiggsfieldAPIError, match="7 days"):
            async with rec.client().stream_output("https://cdn.example/v.mp4"):
                pass

    async def test_a_non_https_output_url_is_refused(self):
        rec = Recorder()
        with pytest.raises(HiggsfieldAPIError, match="non-https"):
            async with rec.client().stream_output("http://cdn.example/v.mp4"):
                pass
        assert rec.requests == []


class TestLifecycle:
    async def test_aclose_closes_both_clients(self):
        client = Recorder().client()
        await client.aclose()
        assert client._api.is_closed and client._bare.is_closed

    def test_repr_redacts_the_secret(self):
        text = repr(HiggsfieldClient("kid-12345", "super-secret"))
        assert "super-secret" not in text
        assert "kid-12345" not in text
