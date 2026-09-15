"""The 2026-07-28 wire contract, and what the mcp 2.x migration must not lose.

A 2026-07-28 request is one self-contained POST: no `initialize`, no session id,
the negotiated version and client capabilities in `params._meta`, and routing
headers (`MCP-Protocol-Version`, `MCP-Method`, `MCP-Name`) mirroring the body.
These tests drive that path against `build_http_app()` because that is what a
host on the new revision actually sends; the legacy handshake keeps working
and is covered by the existing HTTP tests.
"""

import importlib
import json
import sys

import pytest
from starlette.testclient import TestClient

from sanzaru.storage.local import LocalStorageBackend

UI_URI = "ui://sanzaru/media-viewer.html"
APPS_EXTENSION = "io.modelcontextprotocol/ui"
PROTOCOL = "2026-07-28"

_ENVELOPE = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL,
    "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {APPS_EXTENSION: {"mimeTypes": ["text/html;profile=mcp-app"]}}
    },
    "io.modelcontextprotocol/clientInfo": {"name": "test-host", "version": "0"},
}


@pytest.fixture
def server(monkeypatch, tmp_path):
    """`sanzaru.server` re-imported with a media path so the viewer tools register.

    Tool registration happens at import, so a module another test imported
    without a media path has no `view_media`. Reload under the fixture's
    environment and drop the module afterwards (the pattern
    tests/audio/test_server.py uses) so no other test inherits this copy.
    """
    monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
    for name in (
        "SANZARU_HTTP_TOKEN",
        "SANZARU_ALLOW_UNAUTHENTICATED_HTTP",
        "SANZARU_ALLOWED_HOSTS",
        "SANZARU_ALLOWED_ORIGINS",
        "SANZARU_IDENTITY_HEADER",
        "SANZARU_REQUIRE_USER_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    mod = importlib.reload(importlib.import_module("sanzaru.server"))
    try:
        yield mod
    finally:
        sys.modules.pop("sanzaru.server", None)


def _modern_post(client, method, params=None, *, name=None, request_id=1):
    body_params = dict(params or {})
    body_params["_meta"] = _ENVELOPE
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "host": "127.0.0.1:8000",
        "MCP-Protocol-Version": PROTOCOL,
        "MCP-Method": method,
    }
    if name is not None:
        headers["MCP-Name"] = name
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": body_params},
        headers=headers,
    )


@pytest.mark.integration
class TestModernSingleExchange:
    def test_tools_list_answers_without_a_handshake(self, server):
        with TestClient(server.build_http_app()) as client:
            response = _modern_post(client, "tools/list")

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") is None
        result = response.json()["result"]
        assert {"view_media", "_get_media_data"} <= {t["name"] for t in result["tools"]}
        # Every 2026-07-28 result is stamped with the server's identity — the
        # package version, not the SDK's empty default.
        server_info = result["_meta"]["io.modelcontextprotocol/serverInfo"]
        assert server_info["name"] == "sanzaru"
        assert server_info["version"] == importlib.metadata.version("sanzaru")

    def test_discover_advertises_the_apps_extension_and_the_new_revision(self, server):
        with TestClient(server.build_http_app()) as client:
            response = _modern_post(client, "server/discover")

        assert response.status_code == 200
        result = response.json()["result"]
        assert PROTOCOL in result["supportedVersions"]
        assert APPS_EXTENSION in result["capabilities"]["extensions"]

    def test_the_media_viewer_is_an_apps_resource(self, server):
        with TestClient(server.build_http_app()) as client:
            listed = _modern_post(client, "resources/list").json()["result"]["resources"]
            read = _modern_post(client, "resources/read", {"uri": UI_URI}, name=UI_URI).json()["result"]

        (resource,) = [r for r in listed if r["uri"] == UI_URI]
        assert resource["mimeType"] == "text/html;profile=mcp-app"
        assert "blob:" in resource["_meta"]["ui"]["csp"]["resourceDomains"]
        assert resource["_meta"]["ui"]["prefersBorder"] is True
        (content,) = read["contents"]
        assert content["mimeType"] == "text/html;profile=mcp-app"
        assert "<html" in content["text"].lower()

    def test_ui_tools_carry_the_resource_binding(self, server):
        with TestClient(server.build_http_app()) as client:
            tools = {t["name"]: t for t in _modern_post(client, "tools/list").json()["result"]["tools"]}

        assert tools["view_media"]["_meta"]["ui"]["resourceUri"] == UI_URI
        assert tools["_get_media_data"]["_meta"]["ui"]["visibility"] == ["app"]
        assert tools["view_media"]["annotations"]["readOnlyHint"] is True


@pytest.mark.integration
class TestToolErrorsStayReadable:
    """mcp 2.x hides a non-`ToolError` exception's text from the client.

    Under 1.x every exception's message reached the model. sanzaru's tools raise
    `ValueError` *for* the model — "File not found", "not a valid OpenAI resource
    id", the traversal refusals — so `_llm_facing` has to put that text back.
    """

    def test_a_value_error_message_reaches_the_client(self, server, mocker, tmp_path):
        mocker.patch(
            "sanzaru.tools.media_viewer.get_storage",
            return_value=LocalStorageBackend(path_overrides={"video": tmp_path}),
        )
        with TestClient(server.build_http_app()) as client:
            response = _modern_post(
                client,
                "tools/call",
                {"name": "view_media", "arguments": {"media_type": "video", "filename": "nope.mp4"}},
                name="view_media",
            )

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "nope.mp4" in text
        assert "File not found" in text

    def test_a_traversal_refusal_reaches_the_client(self, server, mocker, tmp_path):
        mocker.patch(
            "sanzaru.tools.media_viewer.get_storage",
            return_value=LocalStorageBackend(path_overrides={"video": tmp_path}),
        )
        with TestClient(server.build_http_app()) as client:
            response = _modern_post(
                client,
                "tools/call",
                {"name": "view_media", "arguments": {"media_type": "video", "filename": "../etc/passwd"}},
                name="view_media",
            )

        result = response.json()["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        # The SDK's own prefix is fine (1.x had it too); what must survive is our text.
        assert text.startswith("Error executing tool view_media: ")
        assert "../etc/passwd" in text

    def test_an_unexpected_exception_is_logged_as_a_crash_but_still_explained(self, server, caplog):
        @server._llm_facing
        async def boom() -> None:
            raise TypeError("wrong kind of thing")

        with pytest.raises(server.ToolError, match="wrong kind of thing"):
            import anyio

            anyio.run(boom)
        assert any("crashed" in record.getMessage() for record in caplog.records)

    def test_anticipated_errors_are_not_logged_as_crashes(self, server, caplog):
        @server._llm_facing
        async def refuse() -> None:
            raise ValueError("no such file")

        import anyio

        with pytest.raises(server.ToolError, match="no such file"):
            anyio.run(refuse)
        assert not any("crashed" in record.getMessage() for record in caplog.records)

    def test_protocol_errors_pass_through_untouched(self, server):
        import anyio

        @server._llm_facing
        async def protocol() -> None:
            raise server.MCPError(-32602, "bad params")

        with pytest.raises(server.MCPError):
            anyio.run(protocol)


@pytest.mark.integration
def test_legacy_handshake_still_negotiates(server):
    """Pre-2026 hosts still initialize; the modern path is additive."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "host": "127.0.0.1:8000",
    }
    with TestClient(server.build_http_app()) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "old", "version": "0"},
                },
            },
            headers=headers,
        )

    assert response.status_code == 200
    payload = next(json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: "))
    assert payload["result"]["protocolVersion"] == "2025-06-18"
