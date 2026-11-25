"""MCP Tasks (SEP-2663) over the 2026-07-28 wire, against the real HTTP app.

Driven through `build_http_app()` because the interesting properties are wire
properties: whether a `tools/call` answers with `resultType: "task"`, whether
`tasks/get` resolves across requests (there is no session), and whether a
background task keeps running after the request that started it has returned.
"""

import importlib
import json
import sys
import threading
import time

import pytest
from starlette.testclient import TestClient

from sanzaru.mcp_tasks import store
from sanzaru.storage.factory import set_storage_backend
from sanzaru.storage.local import LocalStorageBackend
from sanzaru.user_context import get_user_context

PROTOCOL = "2026-07-28"
TASKS = "io.modelcontextprotocol/tasks"
JOB = "hf_00000000-0000-4000-8000-000000000001"

_ENV_TO_CLEAR = (
    "SANZARU_HTTP_TOKEN",
    "SANZARU_ALLOW_UNAUTHENTICATED_HTTP",
    "SANZARU_ALLOWED_HOSTS",
    "SANZARU_ALLOWED_ORIGINS",
    "SANZARU_IDENTITY_HEADER",
    "SANZARU_REQUIRE_USER_CONTEXT",
)


def _load_server(monkeypatch, tmp_path, *, tasks: str):
    monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
    monkeypatch.setenv("HF_KEY", "kid:secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SANZARU_TASKS", tasks)
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    return importlib.reload(importlib.import_module("sanzaru.server"))


@pytest.fixture
def media(tmp_path):
    dirs = {"video": tmp_path / "videos", "reference": tmp_path / "images", "audio": tmp_path / "audio"}
    for d in dirs.values():
        d.mkdir()
    set_storage_backend(LocalStorageBackend(path_overrides=dirs))
    try:
        yield dirs
    finally:
        set_storage_backend(None)


@pytest.fixture
def server(monkeypatch, tmp_path, media):
    mod = _load_server(monkeypatch, tmp_path, tasks="1")
    try:
        yield mod
    finally:
        sys.modules.pop("sanzaru.server", None)


def _post(client, method, params=None, *, name=None, tasks=True, email=None):
    body = dict(params or {})
    body["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientCapabilities": {"extensions": {TASKS: {}} if tasks else {}},
        "io.modelcontextprotocol/clientInfo": {"name": "test-host", "version": "0"},
    }
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "host": "127.0.0.1:8000",
        "MCP-Protocol-Version": PROTOCOL,
        "MCP-Method": method,
    }
    if name is not None:
        headers["MCP-Name"] = name
    if email is not None:
        headers["x-forwarded-email"] = email
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": body}, headers=headers)
    # The SDK answers an invalid-params error with HTTP 400 on the 2026-07-28 wire.
    assert response.status_code in (200, 400), response.text
    text = response.text
    if not text.lstrip().startswith("{"):
        text = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)


def _payload(result):
    """The tool's dict: sanzaru's wrappers declare no output schema, so it arrives as one JSON text block."""
    (block,) = result["content"]
    return json.loads(block["text"])


def _call(client, tool, arguments, **kw):
    return _post(client, "tools/call", {"name": tool, "arguments": arguments}, name=tool, **kw)


def _get(client, task_id, **kw):
    return _post(client, "tasks/get", {"taskId": task_id}, name=task_id, **kw)


def _until(client, task_id, predicate, timeout=5.0, **kw):
    deadline = time.monotonic() + timeout
    while True:
        reply = _get(client, task_id, **kw)
        if predicate(reply) or time.monotonic() > deadline:
            return reply
        time.sleep(0.02)


_QUEUED = {
    "id": JOB,
    "status": "queued",
    "model": "seedance-2.5",
    "slug": "bytedance/seedance-2.5",
    "operation": "text",
    "cost": {"usd": 1.0, "source": "local"},
    "arguments": {"prompt": "a cat"},
}


# ---------------------------------------------------------------------------
# Off by default, and invisible to clients that did not ask
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOptIn:
    def test_off_by_default_advertises_nothing_and_answers_inline(self, monkeypatch, tmp_path, media, mocker):
        mod = _load_server(monkeypatch, tmp_path, tasks="")
        try:
            mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
            with TestClient(mod.build_http_app()) as client:
                discover = _post(client, "server/discover")["result"]
                reply = _call(client, "create_video", {"prompt": "a cat"})
            assert TASKS not in discover["capabilities"].get("extensions", {})
            assert reply["result"]["resultType"] == "complete"
            assert _payload(reply["result"])["id"] == JOB
        finally:
            sys.modules.pop("sanzaru.server", None)

    def test_an_unrecognised_switch_value_is_refused(self, monkeypatch, tmp_path, media):
        with pytest.raises(RuntimeError, match="SANZARU_TASKS"):
            _load_server(monkeypatch, tmp_path, tasks="enabled")
        sys.modules.pop("sanzaru.server", None)

    def test_discover_advertises_the_extension_when_on(self, server):
        with TestClient(server.build_http_app()) as client:
            discover = _post(client, "server/discover")["result"]
        assert TASKS in discover["capabilities"]["extensions"]

    def test_a_client_that_did_not_declare_tasks_gets_the_plain_result(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
        with TestClient(server.build_http_app()) as client:
            reply = _call(client, "create_video", {"prompt": "a cat"}, tasks=False)
        assert reply["result"]["resultType"] == "complete"
        assert _payload(reply["result"])["status"] == "queued"


# ---------------------------------------------------------------------------
# Provider-backed: the job id is the task
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProviderBacked:
    def test_video_job_becomes_a_task_and_completes_with_the_download(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
        status = mocker.patch(
            "sanzaru.tools.video.get_video_status",
            return_value={"id": JOB, "status": "in_progress", "done": False, "error": None, "video_url": None},
        )
        download = mocker.patch(
            "sanzaru.tools.video.download_video", return_value={"filename": f"{JOB}.mp4", "format": "mp4"}
        )
        with TestClient(server.build_http_app()) as client:
            created = _call(client, "create_video", {"prompt": "a cat"})["result"]
            assert created["resultType"] == "task"
            assert created["status"] == "working"
            assert created["pollIntervalMs"] == 10_000
            task_id = created["taskId"]
            assert JOB not in task_id  # the id is a fresh bearer token, not the provider's

            working = _get(client, task_id)["result"]
            assert working["status"] == "working"
            assert "in_progress" in working["statusMessage"]
            download.assert_not_called()

            status.return_value = {
                "id": JOB,
                "status": "completed",
                "done": True,
                "error": None,
                "video_url": "https://cdn/x.mp4",
            }
            done = _get(client, task_id)["result"]
            again = _get(client, task_id)["result"]

        assert done["status"] == "completed"
        assert done["resultType"] == "complete"
        structured = _payload(done["result"])
        assert structured["id"] == JOB and structured["status"] == "completed"
        assert structured["file"] == f"{JOB}.mp4"
        assert structured["cost"] == _QUEUED["cost"]  # the submission's fields survive
        # A completed record answers from storage: no second status read, no second download.
        assert again == done
        download.assert_called_once_with(JOB)

    def test_a_failed_render_is_a_completed_task_with_an_error_result(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
        mocker.patch(
            "sanzaru.tools.video.get_video_status",
            return_value={
                "id": JOB,
                "status": "nsfw",
                "done": True,
                "error": "rejected by moderation",
                "video_url": None,
            },
        )
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "create_video", {"prompt": "x"})["result"]["taskId"]
            done = _get(client, task_id)["result"]
        # SEP-2663: `failed` is for JSON-RPC errors; a tool that ran and failed is `completed`.
        assert done["status"] == "completed"
        assert done["result"]["isError"] is True
        assert "rejected by moderation" in done["result"]["content"][0]["text"]

    def test_a_status_read_failure_keeps_the_task_working(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
        mocker.patch("sanzaru.tools.video.get_video_status", side_effect=RuntimeError("503 upstream"))
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "create_video", {"prompt": "x"})["result"]["taskId"]
            reply = _get(client, task_id)["result"]
        assert reply["status"] == "working"
        assert "will retry" in reply["statusMessage"]

    def test_a_dry_run_submits_nothing_and_stays_a_plain_result(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value={**_QUEUED, "id": None, "status": "dry_run"})
        with TestClient(server.build_http_app()) as client:
            reply = _call(client, "create_video", {"prompt": "x", "dry_run": True})["result"]
        assert reply["resultType"] == "complete"
        assert _payload(reply)["status"] == "dry_run"

    def test_a_refused_submit_stays_a_plain_error(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", side_effect=ValueError("estimate $9.00 exceeds max_cost_usd"))
        with TestClient(server.build_http_app()) as client:
            reply = _call(client, "create_video", {"prompt": "x", "max_cost_usd": 1})["result"]
        assert reply["resultType"] == "complete"
        assert reply["isError"] is True

    def test_cancel_asks_the_provider_and_the_next_poll_settles_it(self, server, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
        status = mocker.patch(
            "sanzaru.tools.video.get_video_status",
            return_value={"id": JOB, "status": "queued", "done": False, "error": None, "video_url": None},
        )
        cancel = mocker.patch("sanzaru.tools.video.cancel_video", return_value={"id": JOB, "canceled": True})
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "create_video", {"prompt": "x"})["result"]["taskId"]
            ack = _post(client, "tasks/cancel", {"taskId": task_id}, name=task_id)["result"]
            status.return_value = {"id": JOB, "status": "canceled", "done": True, "error": None, "video_url": None}
            done = _get(client, task_id)["result"]
        assert ack["resultType"] == "complete"
        cancel.assert_called_once_with(JOB)
        assert done["status"] == "cancelled"

    def test_create_image_is_provider_backed_too(self, server, mocker):
        mocker.patch(
            "sanzaru.tools.image.create_image",
            return_value={"id": "resp_abc", "status": "queued", "created_at": 1.0},
        )
        mocker.patch(
            "sanzaru.tools.image.get_image_status",
            return_value={"id": "resp_abc", "status": "completed", "created_at": 1.0},
        )
        mocker.patch(
            "sanzaru.tools.image.download_image",
            return_value={"filename": "resp_abc.png", "size": (1024, 1024), "format": "png"},
        )
        with TestClient(server.build_http_app()) as client:
            created = _call(client, "create_image", {"prompt": "an icon"})["result"]
            done = _get(client, created["taskId"])["result"]
        assert created["resultType"] == "task"
        assert _payload(done["result"])["file"] == "resp_abc.png"


# ---------------------------------------------------------------------------
# Background: work that outlives its request
# ---------------------------------------------------------------------------


def _gate():
    """A tool body that blocks until the test releases it, recording who it ran as."""
    release = threading.Event()
    seen: dict[str, object] = {}

    async def body(*args, **kwargs):
        import anyio

        ctx = get_user_context()
        seen["user"] = ctx.email if ctx else None
        while not release.is_set():
            await anyio.sleep(0.01)
        return {"filename": "out.png", "size": (1024, 1024), "format": "png", "usage": None}

    return release, seen, body


@pytest.mark.integration
class TestBackground:
    def test_generate_image_runs_after_the_request_returned(self, server, mocker):
        release, _, body = _gate()
        mocker.patch("sanzaru.tools.images_api.generate_image", side_effect=body)
        with TestClient(server.build_http_app()) as client:
            created = _call(client, "generate_image", {"prompt": "an icon"})["result"]
            assert created["resultType"] == "task"
            assert _get(client, created["taskId"])["result"]["status"] == "working"
            release.set()
            done = _until(client, created["taskId"], lambda r: r["result"]["status"] != "working")["result"]
        assert done["status"] == "completed"
        assert _payload(done["result"])["filename"] == "out.png"

    def test_a_tool_error_completes_the_task_with_is_error(self, server, mocker):
        mocker.patch("sanzaru.tools.images_api.generate_image", side_effect=ValueError("prompt refused"))
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "generate_image", {"prompt": "x"})["result"]["taskId"]
            done = _until(client, task_id, lambda r: r["result"]["status"] != "working")["result"]
        assert done["status"] == "completed"
        assert done["result"]["isError"] is True
        assert "prompt refused" in done["result"]["content"][0]["text"]

    def test_cancel_stops_the_work(self, server, mocker):
        release, _, body = _gate()
        mocker.patch("sanzaru.tools.images_api.generate_image", side_effect=body)
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "generate_image", {"prompt": "x"})["result"]["taskId"]
            _post(client, "tasks/cancel", {"taskId": task_id}, name=task_id)
            done = _until(client, task_id, lambda r: r["result"]["status"] != "working")["result"]
        release.set()
        assert done["status"] == "cancelled"

    def test_a_record_whose_worker_stopped_beating_is_reported_failed(self, server, mocker):
        release, _, body = _gate()
        mocker.patch("sanzaru.tools.images_api.generate_image", side_effect=body)
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "generate_image", {"prompt": "x"})["result"]["taskId"]
            # Simulate another replica's dead worker: not ours, and silent for too long.
            mocker.patch.object(server._task_runner, "owns", return_value=False)
            mocker.patch(
                "sanzaru.mcp_tasks.extension.time.time",
                return_value=time.time() + 10 * 60,
            )
            reply = _get(client, task_id)["result"]
            release.set()
        assert reply["status"] == "failed"
        assert "stopped reporting progress" in reply["error"]["message"]

    def test_runs_as_the_caller_and_no_one_else_can_read_it(self, server, monkeypatch, mocker):
        monkeypatch.setenv("SANZARU_IDENTITY_HEADER", "x-forwarded-email")
        release, seen, body = _gate()
        mocker.patch("sanzaru.tools.images_api.generate_image", side_effect=body)
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "generate_image", {"prompt": "x"}, email="alice@example.com")["result"]["taskId"]
            mallory = _get(client, task_id, email="mallory@example.com")
            anonymous = _get(client, task_id)
            release.set()
            done = _until(client, task_id, lambda r: r["result"]["status"] != "working", email="alice@example.com")[
                "result"
            ]
        # The spawned work kept the request's identity: storage scoping follows it.
        assert seen["user"] == "alice@example.com"
        assert done["status"] == "completed"
        for reply in (mallory, anonymous):
            assert reply["error"]["code"] == -32602
            assert reply["error"]["message"] == "Failed to retrieve task: Task not found"


# ---------------------------------------------------------------------------
# Ids and records
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIdsAndRecords:
    @pytest.mark.parametrize("task_id", ["nope", "tkv_doesnotexist", "tkz_abc", "hf_" + "0" * 32, "../../etc"])
    def test_unknown_ids_are_invalid_params(self, server, task_id):
        with TestClient(server.build_http_app()) as client:
            reply = _get(client, task_id)
        assert reply["error"]["code"] == -32602

    def test_the_record_file_is_named_by_digest_not_by_id(self, server, media, mocker):
        mocker.patch("sanzaru.tools.video.create_video", return_value=_QUEUED)
        with TestClient(server.build_http_app()) as client:
            task_id = _call(client, "create_video", {"prompt": "x"})["result"]["taskId"]
        names = [p.name for p in media["video"].iterdir()]
        assert names == [store.record_name(task_id)]
        assert task_id not in names[0]

    def test_record_names_are_reserved_from_caller_chosen_outputs(self):
        from sanzaru.utils import reject_reserved_name

        name = store.record_name("tkv_x")
        with pytest.raises(ValueError, match="reserved"):
            reject_reserved_name(name)
        with pytest.raises(ValueError, match="reserved"):
            reject_reserved_name(store.cancel_marker_name("tkv_x"))
