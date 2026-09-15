# SPDX-License-Identifier: MIT
"""The identity header is opt-in — nothing is trusted unless the operator says so.

`UserContextMiddleware` is only installed when `SANZARU_IDENTITY_HEADER` is
explicitly set. Defaulting to `x-forwarded-email` trusted that header
everywhere: on a direct-exposed Databricks-backed server, any client holding
the bearer token could pick an arbitrary tenant namespace by writing the header
themselves, and a corporate proxy injecting it unasked silently re-pathed a
single-tenant deployment's files into per-user prefixes.

Once trusted, the header must arrive exactly once and well-formed. Starlette's
`Headers.get()` returns the *first* copy, so a proxy that appends instead of
replacing let the client's own value pick the tenant. Neither copy is trusted.
"""

import pytest

from sanzaru.server import IDENTITY_HEADER_ENV, UserContextMiddleware, identity_header_name
from sanzaru.user_context import get_user_context


@pytest.fixture(autouse=True)
def _isolated_identity_env(monkeypatch):
    monkeypatch.delenv(IDENTITY_HEADER_ENV, raising=False)
    monkeypatch.delenv("SANZARU_REQUIRE_USER_CONTEXT", raising=False)


@pytest.mark.unit
class TestIdentityHeaderIsOptIn:
    def test_unset_means_no_header_is_trusted(self):
        assert identity_header_name() is None

    def test_blank_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv(IDENTITY_HEADER_ENV, "   ")
        assert identity_header_name() is None

    def test_set_names_the_header(self, monkeypatch):
        monkeypatch.setenv(IDENTITY_HEADER_ENV, "X-Forwarded-Email")
        assert identity_header_name() == "x-forwarded-email"


@pytest.mark.integration
class TestUserContextMiddleware:
    """The middleware itself, once an operator has opted in."""

    @staticmethod
    def _scope(headers: list[tuple[bytes, bytes]], kind: str = "http"):
        return {"type": kind, "headers": headers}

    @staticmethod
    def _rig():
        """An inner app recording the bound identity, and a `send` recording the refusal."""
        seen, sent = [], []

        async def app(scope, receive, send):
            seen.append(get_user_context())

        async def send(message):
            sent.append(message)

        return app, send, seen, sent

    async def test_binds_the_identity_and_resets_it_after(self):
        app, send, seen, sent = self._rig()

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([(b"x-forwarded-email", b"user@example.com")]), None, send)

        assert seen[0] is not None
        assert seen[0].email == "user@example.com"
        assert sent == []
        assert get_user_context() is None, "the context must not leak past the request"

    async def test_no_header_means_no_identity(self):
        app, send, seen, sent = self._rig()

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([]), None, send)

        assert seen == [None]
        assert sent == []

    async def test_a_blank_header_is_treated_as_absent(self):
        app, send, seen, sent = self._rig()

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([(b"x-forwarded-email", b"   ")]), None, send)

        assert seen == [None]
        assert sent == []

    async def test_a_duplicated_header_is_refused_with_400_and_binds_nobody(self):
        """Client-then-proxy order is the appending-proxy case; the first copy used to win."""
        app, send, seen, sent = self._rig()
        headers = [
            (b"x-forwarded-email", b"attacker@evil.example"),
            (b"x-forwarded-email", b"real@corp.example"),
        ]

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope(headers), None, send)

        assert seen == [], "the inner app must never see a request with an ambiguous identity"
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 400
        assert get_user_context() is None

    async def test_a_malformed_identity_is_refused_not_downgraded(self):
        """The header is the proxy's word; garbage means a broken proxy, not an anonymous caller.

        Dropping to no identity used to let the request proceed under the shared
        volume root — silently, unless SANZARU_REQUIRE_USER_CONTEXT happened to be set.
        """
        app, send, seen, sent = self._rig()

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([(b"x-forwarded-email", b"not-an-email")]), None, send)

        assert seen == []
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 400

    async def test_websocket_scopes_are_bound_too(self):
        """Only the lifespan bypasses the middleware; `!= "http"` used to skip websockets."""
        app, send, seen, sent = self._rig()

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([(b"x-forwarded-email", b"user@example.com")], kind="websocket"), None, send)

        assert seen[0] is not None and seen[0].email == "user@example.com"

    async def test_a_duplicated_header_on_a_websocket_closes_the_socket(self):
        app, send, seen, sent = self._rig()
        headers = [(b"x-forwarded-email", b"a@x.example"), (b"x-forwarded-email", b"b@x.example")]

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope(headers, kind="websocket"), None, send)

        assert seen == []
        assert sent == [{"type": "websocket.close", "code": 1008, "reason": "Ambiguous identity header"}]

    async def test_lifespan_passes_through_unbound(self):
        app, send, seen, sent = self._rig()

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware({"type": "lifespan"}, None, send)

        assert seen == [None]
        assert sent == []
