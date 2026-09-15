# SPDX-License-Identifier: MIT
"""HTTP mode is authenticated on the artifact the docs hand out, not only on the CLI path.

`BearerTokenMiddleware` used to be installed inside `_run_http` alone, while the
production snippet in CLAUDE.md mounted the bare `mcp.streamable_http_app()` —
so anyone following the repo's own instructions served every tool with no
credential (finding 07 verbatim). `build_http_app` is the single seam both paths
now go through, and these tests pin what it guarantees: the token on /mcp, the
byte-level `_authorized` edge cases, the Host/Origin policy off loopback, and the
startup refusal with its escape hatch. All of it fails silently when it regresses,
which is why it is pinned at all.
"""

import importlib
import json

import pytest
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.testclient import TestClient

from sanzaru.exceptions import ConfigurationError
from sanzaru.server import (
    ALLOW_UNAUTH_ENV,
    ALLOWED_HOSTS_ENV,
    ALLOWED_ORIGINS_ENV,
    EXIT_CONFIG,
    HTTP_TOKEN_ENV,
    IDENTITY_HEADER_ENV,
    BearerTokenMiddleware,
    _authorized,
    _is_loopback,
)
from sanzaru.storage.local import LocalStorageBackend
from sanzaru.user_context import get_user_context

HTTP_ENV = (
    HTTP_TOKEN_ENV,
    ALLOW_UNAUTH_ENV,
    IDENTITY_HEADER_ENV,
    ALLOWED_HOSTS_ENV,
    ALLOWED_ORIGINS_ENV,
    "SANZARU_REQUIRE_USER_CONTEXT",
)

# TestClient's default Host is "testserver", which the loopback allowlist rejects
# — exactly as it would a rebound page. Requests that should succeed name a
# loopback host, and /mcp needs the streamable-HTTP content negotiation.
MCP_HEADERS = {
    "host": "127.0.0.1:8000",
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


@pytest.fixture(autouse=True)
def _isolated_http_env(monkeypatch):
    """Nothing here may depend on the developer's shell.

    The README tells operators to `export SANZARU_HTTP_TOKEN`; with it exported,
    every unauthenticated request in this module would 401 for reasons unrelated
    to the change under test.
    """
    for name in HTTP_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def server():
    """The *live* `sanzaru.server` module.

    tests/audio/test_server.py reloads the module and drops it from
    `sys.modules`, so anything bound at collection time — `mcp`, `build_http_app`
    — goes stale mid-session while `mocker.patch("sanzaru.server....")` targets
    the fresh copy. Resolving it per test is the same convention the /media
    tests follow with their in-function imports.
    """
    return importlib.import_module("sanzaru.server")


@pytest.fixture
def fresh_server_state(server, monkeypatch):
    """Start from the loopback policy and leave nothing behind.

    `build_http_app` rewrites the module-level Host/Origin policy that `/media`
    reads at request time; mcp 2.x builds a fresh session manager on every
    `streamable_http_app()` call, so that is the only process state to reset.
    """
    monkeypatch.setattr(server, "_transport_security", server._LOOPBACK_TRANSPORT_SECURITY)


def _rpc_result(response):
    """The JSON-RPC payload out of a single-event SSE body."""
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"no SSE data line in {response.text!r}")


@pytest.mark.unit
class TestIsLoopback:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "127.0.0.2", "127.255.255.254", "localhost", "LOCALHOST", "::1", "[::1]"]
    )
    def test_every_spelling_of_loopback(self, host):
        assert _is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.1", "192.168.1.1", "sanzaru.example.com", ""])
    def test_anything_reachable_is_not(self, host):
        assert not _is_loopback(host)


@pytest.mark.unit
class TestAuthorized:
    """Compared as bytes — the two bugs the docstring documents must stay fixed."""

    TOKEN = "s3cret"

    @staticmethod
    def _headers(raw):
        return Headers(raw=[] if raw is None else [(b"authorization", raw)])

    @pytest.mark.parametrize(
        "raw",
        [
            None,  # absent
            b"Basic czNjcmV0",  # wrong scheme, right secret
            b"s3cret",  # no scheme
            b"Bearer",  # no value
            b"Bearer ",  # empty value
            b"Bearer wrong",
            b"Bearer s3cre",  # one short
            b"Bearer s3cret!",  # one long
            b"Bearer S3CRET",  # tokens are case-sensitive even if the scheme is not
        ],
    )
    def test_rejects(self, raw):
        assert _authorized(self._headers(raw), self.TOKEN) is False

    @pytest.mark.parametrize("raw", [b"Bearer s3cret", b"bearer s3cret", b"BEARER s3cret", b"Bearer  s3cret "])
    def test_accepts_any_case_of_the_scheme(self, raw):
        assert _authorized(self._headers(raw), self.TOKEN) is True

    def test_non_ascii_header_bytes_are_a_401_not_a_500(self):
        # `Bearer \xff` reaches us as latin-1 text; compare_digest on str refused
        # it with TypeError, which the error middleware turned into a 500.
        assert _authorized(self._headers(b"Bearer \xff\xfe"), self.TOKEN) is False

    def test_a_non_ascii_token_round_trips(self):
        token = "sécret-🔑"
        assert _authorized(self._headers(b"Bearer " + token.encode("utf-8")), token) is True
        assert _authorized(self._headers(b"Bearer " + "sécret-🔒".encode()), token) is False


@pytest.mark.integration
class TestBuildHttpAppAuthenticatesMcp:
    def test_mcp_requires_the_bearer_token(self, server, monkeypatch, fresh_server_state):
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")

        with TestClient(server.build_http_app()) as client:
            anonymous = client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS)
            assert anonymous.status_code == 401
            assert anonymous.headers["www-authenticate"] == "Bearer"

            wrong = client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS | {"authorization": "Bearer wrong"})
            assert wrong.status_code == 401

            ok = client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS | {"authorization": "Bearer s3cret"})
            assert ok.status_code == 200
            assert "tools" in _rpc_result(ok)["result"]

    def test_the_bare_sdk_app_is_the_gap_the_seam_closes(self, server, monkeypatch, fresh_server_state):
        """Why the production snippet must mount `build_http_app`, not this.

        With the token exported, `server.mcp.streamable_http_app()` still answers /mcp
        to anyone: nothing about the SDK reads SANZARU_HTTP_TOKEN. Should this
        ever start failing, the SDK grew its own bearer check and the seam can
        be revisited.
        """
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        with TestClient(server.mcp.streamable_http_app(stateless_http=True)) as client:
            assert client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS).status_code == 200

    def test_returns_a_starlette_app_ready_to_mount(self, server, fresh_server_state):
        assert isinstance(server.build_http_app(), Starlette)

    def test_loopback_needs_no_token_and_keeps_the_sdk_allowlist(
        self, server, mocker, tmp_video_path, fresh_server_state
    ):
        """Loopback binds run open by default and on any port.

        Review §3 worried the allowlist named `127.0.0.1:8000` and would 421 a
        `--port 3000` server; the SDK's list is `127.0.0.1:*`, so it does not.
        """
        storage = LocalStorageBackend(path_overrides={"video": tmp_video_path})
        mocker.patch("sanzaru.server.get_storage", return_value=storage)

        app = server.build_http_app(host="127.0.0.1", port=3000)
        assert server.current_transport_security() is server._LOOPBACK_TRANSPORT_SECURITY

        with TestClient(app) as client:
            local = MCP_HEADERS | {"host": "127.0.0.1:3000"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=local).status_code == 200
            assert client.get("/media/video/none.mp4", headers={"host": "127.0.0.1:3000"}).status_code == 404
            assert client.get("/media/video/none.mp4", headers={"host": "[::1]:3000"}).status_code == 404

            rebound = MCP_HEADERS | {"host": "evil.example:3000"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=rebound).status_code == 421
            assert client.get("/media/video/none.mp4", headers={"host": "evil.example:3000"}).status_code == 421

    def test_identity_reaches_a_tool_under_the_stateless_session_manager(self, server, monkeypatch, fresh_server_state):
        """The contextvar set in middleware is visible inside the tool call.

        Whether it survives depends on when `StreamableHTTPSessionManager` copies
        context into the task it spawns per stateless request — an SDK property
        worth an in-tree pin rather than a throwaway probe.
        """
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        monkeypatch.setenv(IDENTITY_HEADER_ENV, "X-Forwarded-Email")

        async def whoami() -> str:
            ctx = get_user_context()
            return ctx.email if ctx is not None else "NONE"

        server.mcp.add_tool(whoami, name="_test_whoami")
        call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "_test_whoami", "arguments": {}}}
        auth = MCP_HEADERS | {"authorization": "Bearer s3cret"}
        try:
            with TestClient(server.build_http_app()) as client:
                with_identity = client.post(
                    "/mcp", json=call, headers=auth | {"x-forwarded-email": "alice@example.com"}
                )
                assert with_identity.status_code == 200
                assert _rpc_result(with_identity)["result"]["content"][0]["text"] == "alice@example.com"

                without = client.post("/mcp", json=call, headers=auth)
                assert _rpc_result(without)["result"]["content"][0]["text"] == "NONE"
        finally:
            server.mcp.remove_tool("_test_whoami")

    def test_a_duplicated_identity_header_is_refused_before_any_tool_runs(
        self, server, monkeypatch, fresh_server_state
    ):
        """An appending proxy forwards the client's copy first; neither copy is trusted."""
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        monkeypatch.setenv(IDENTITY_HEADER_ENV, "x-forwarded-email")

        headers = [
            *MCP_HEADERS.items(),
            ("authorization", "Bearer s3cret"),
            ("x-forwarded-email", "attacker@evil.example"),
            ("x-forwarded-email", "real@corp.example"),
        ]
        with TestClient(server.build_http_app()) as client:
            response = client.post("/mcp", content=json.dumps(TOOLS_LIST), headers=headers)

        assert response.status_code == 400
        assert response.text == "Ambiguous identity header"

    def test_identity_is_only_read_from_an_authenticated_request(self, server, monkeypatch, fresh_server_state):
        """Bearer wraps outermost: no token, no 400 about the identity header either."""
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        monkeypatch.setenv(IDENTITY_HEADER_ENV, "x-forwarded-email")

        with TestClient(server.build_http_app()) as client:
            response = client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS | {"x-forwarded-email": "garbage"})

        assert response.status_code == 401


@pytest.mark.integration
class TestNonLoopbackPolicy:
    """Off loopback, the Origin check survives — it is not dropped with the Host allowlist."""

    def test_the_unauthenticated_hatch_keeps_host_and_origin_validation(self, server, monkeypatch, fresh_server_state):
        monkeypatch.setenv(ALLOW_UNAUTH_ENV, "1")
        monkeypatch.setenv(ALLOWED_HOSTS_ENV, "sanzaru.example.com, sanzaru.example.com:*")

        app = server.build_http_app(host="0.0.0.0", port=8000)

        policy = server.current_transport_security()
        assert policy is not None
        assert policy.enable_dns_rebinding_protection is True
        assert policy.allowed_hosts == ["sanzaru.example.com", "sanzaru.example.com:*"]
        assert policy.allowed_origins == [
            "http://sanzaru.example.com",
            "https://sanzaru.example.com",
            "http://sanzaru.example.com:*",
            "https://sanzaru.example.com:*",
        ]

        with TestClient(app) as client:
            good = MCP_HEADERS | {"host": "sanzaru.example.com"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=good).status_code == 200
            assert (
                client.post(
                    "/mcp", json=TOOLS_LIST, headers=good | {"origin": "https://sanzaru.example.com:8443"}
                ).status_code
                == 200
            )

            rebound_page = good | {"origin": "http://evil.example"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=rebound_page).status_code == 403
            assert (
                client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS | {"host": "evil.example"}).status_code == 421
            )

            # /media applies the same object by hand
            media_page = {"host": "sanzaru.example.com", "origin": "http://evil.example"}
            assert client.get("/media/video/x.mp4", headers=media_page).status_code == 403
            assert client.get("/media/video/x.mp4", headers={"host": "evil.example"}).status_code == 421

    def test_explicit_origins_replace_the_derived_ones(self, server, monkeypatch, fresh_server_state):
        monkeypatch.setenv(ALLOW_UNAUTH_ENV, "1")
        monkeypatch.setenv(ALLOWED_HOSTS_ENV, "sanzaru.example.com")
        monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example, https://staging.app.example")

        server.build_http_app(host="0.0.0.0", port=8000)

        policy = server.current_transport_security()
        assert policy is not None
        assert policy.allowed_origins == ["https://app.example", "https://staging.app.example"]

    def test_the_hatch_without_an_allowlist_is_refused(self, server, monkeypatch, fresh_server_state):
        """The unauthenticated path must not also be the least-protected one."""
        monkeypatch.setenv(ALLOW_UNAUTH_ENV, "1")

        with pytest.raises(ConfigurationError, match=ALLOWED_HOSTS_ENV):
            server.build_http_app(host="0.0.0.0", port=8000)

    def test_a_token_without_an_allowlist_turns_validation_off(self, server, monkeypatch, fresh_server_state):
        """The token is the control; the SDK's loopback list would reject the real hostname."""
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")

        app = server.build_http_app(host="0.0.0.0", port=8000)

        policy = server.current_transport_security()
        assert policy is not None
        assert policy.enable_dns_rebinding_protection is False

        with TestClient(app) as client:
            real_host = MCP_HEADERS | {"host": "sanzaru.example.com"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=real_host).status_code == 401
            authed = real_host | {"authorization": "Bearer s3cret"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=authed).status_code == 200

    def test_a_token_plus_an_allowlist_keeps_both_controls(self, server, monkeypatch, fresh_server_state):
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        monkeypatch.setenv(ALLOWED_HOSTS_ENV, "sanzaru.example.com")

        app = server.build_http_app(host="0.0.0.0", port=8000)

        policy = server.current_transport_security()
        assert policy is not None
        assert policy.enable_dns_rebinding_protection is True

        with TestClient(app) as client:
            authed = MCP_HEADERS | {"host": "sanzaru.example.com", "authorization": "Bearer s3cret"}
            assert client.post("/mcp", json=TOOLS_LIST, headers=authed).status_code == 200
            assert client.post("/mcp", json=TOOLS_LIST, headers=authed | {"host": "evil.example"}).status_code == 421
            assert (
                client.post("/mcp", json=TOOLS_LIST, headers=authed | {"origin": "http://evil.example"}).status_code
                == 403
            )
            assert (
                client.post("/mcp", json=TOOLS_LIST, headers=MCP_HEADERS | {"host": "sanzaru.example.com"}).status_code
                == 401
            )

    def test_a_non_loopback_bind_does_not_leak_into_a_later_loopback_one(self, server, monkeypatch, fresh_server_state):
        """Each build derives its policy from scratch; the loopback allowlist is restored."""
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        server.build_http_app(host="0.0.0.0", port=8000)
        policy = server.current_transport_security()
        assert policy is not None and policy.enable_dns_rebinding_protection is False

        server.build_http_app(host="127.0.0.1", port=8000)
        assert server.current_transport_security() is server._LOOPBACK_TRANSPORT_SECURITY


@pytest.mark.integration
class TestRunHttpStartupPolicy:
    """The load-bearing half of finding 07: refuse to start rather than serve open."""

    def test_non_loopback_without_a_token_refuses_to_start(self, server, mocker, capsys, fresh_server_state):
        run = mocker.patch("uvicorn.run")

        with pytest.raises(SystemExit) as excinfo:
            server._run_http(host="0.0.0.0", port=8765)

        assert excinfo.value.code == EXIT_CONFIG
        err = capsys.readouterr().err
        assert "Refusing to serve 0.0.0.0:8765" in err
        assert HTTP_TOKEN_ENV in err and ALLOW_UNAUTH_ENV in err
        run.assert_not_called()

    @pytest.mark.parametrize("value", ["1", "true", "yes", " TRUE "])
    def test_the_escape_hatch_starts_unauthenticated(self, server, mocker, monkeypatch, value, fresh_server_state):
        monkeypatch.setenv(ALLOW_UNAUTH_ENV, value)
        monkeypatch.setenv(ALLOWED_HOSTS_ENV, "sanzaru.example.com")
        run = mocker.patch("uvicorn.run")

        server._run_http(host="0.0.0.0", port=8765)

        run.assert_called_once()
        assert isinstance(run.call_args.args[0], Starlette)
        assert run.call_args.kwargs["host"] == "0.0.0.0"
        assert run.call_args.kwargs["port"] == 8765

    @pytest.mark.parametrize("value", ["0", "no", "false", "maybe", ""])
    def test_anything_else_is_not_a_hatch(self, server, mocker, monkeypatch, value, fresh_server_state):
        monkeypatch.setenv(ALLOW_UNAUTH_ENV, value)
        monkeypatch.setenv(ALLOWED_HOSTS_ENV, "sanzaru.example.com")
        run = mocker.patch("uvicorn.run")

        with pytest.raises(SystemExit) as excinfo:
            server._run_http(host="0.0.0.0", port=8765)

        assert excinfo.value.code == EXIT_CONFIG
        run.assert_not_called()

    def test_the_hatch_without_an_allowlist_exits_with_the_config_code(
        self, server, mocker, monkeypatch, capsys, fresh_server_state
    ):
        monkeypatch.setenv(ALLOW_UNAUTH_ENV, "1")
        run = mocker.patch("uvicorn.run")

        with pytest.raises(SystemExit) as excinfo:
            server._run_http(host="0.0.0.0", port=8765)

        assert excinfo.value.code == EXIT_CONFIG
        assert ALLOWED_HOSTS_ENV in capsys.readouterr().err
        run.assert_not_called()

    def test_a_token_starts_a_non_loopback_bind(self, server, mocker, monkeypatch, fresh_server_state):
        monkeypatch.setenv(HTTP_TOKEN_ENV, "s3cret")
        run = mocker.patch("uvicorn.run")

        server._run_http(host="0.0.0.0", port=8765)

        run.assert_called_once()

    def test_loopback_starts_with_nothing_configured(self, server, mocker, fresh_server_state):
        run = mocker.patch("uvicorn.run")

        server._run_http(host="127.0.0.1", port=3000)

        run.assert_called_once()
        assert run.call_args.kwargs["port"] == 3000

    def test_run_server_routes_http_here(self, server, mocker, fresh_server_state):
        run_http = mocker.patch("sanzaru.server._run_http")

        server.run_server(transport="http", host="0.0.0.0", port=9000)

        run_http.assert_called_once_with(host="0.0.0.0", port=9000)


@pytest.mark.integration
class TestBearerTokenMiddlewareScopes:
    """Only the lifespan bypasses the check; `!= "http"` also waved websockets through."""

    @staticmethod
    def _rig():
        reached, sent = [], []

        async def inner(scope, receive, send):
            reached.append(scope["type"])

        async def send(message):
            sent.append(message)

        return inner, send, reached, sent

    async def test_http_without_a_token_gets_401(self):
        inner, send, reached, sent = self._rig()

        await BearerTokenMiddleware(inner, token="tok")({"type": "http", "headers": []}, None, send)

        assert reached == []
        assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 401

    async def test_websocket_without_a_token_is_closed_not_passed_through(self):
        inner, send, reached, sent = self._rig()

        await BearerTokenMiddleware(inner, token="tok")({"type": "websocket", "headers": []}, None, send)

        assert reached == []
        assert sent == [{"type": "websocket.close", "code": 1008, "reason": "Unauthorized"}]

    async def test_websocket_with_the_token_passes(self):
        inner, send, reached, sent = self._rig()
        scope = {"type": "websocket", "headers": [(b"authorization", b"Bearer tok")]}

        await BearerTokenMiddleware(inner, token="tok")(scope, None, send)

        assert reached == ["websocket"]
        assert sent == []

    async def test_lifespan_passes_without_a_token(self):
        inner, send, reached, sent = self._rig()

        await BearerTokenMiddleware(inner, token="tok")({"type": "lifespan"}, None, send)

        assert reached == ["lifespan"]
        assert sent == []
