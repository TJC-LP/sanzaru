# SPDX-License-Identifier: MIT
"""The identity header is opt-in — nothing is trusted unless the operator says so.

`UserContextMiddleware` is only installed when `SANZARU_IDENTITY_HEADER` is
explicitly set. Defaulting to `x-forwarded-email` trusted that header
everywhere: on a direct-exposed Databricks-backed server, any client holding
the bearer token could pick an arbitrary tenant namespace by writing the header
themselves, and a corporate proxy injecting it unasked silently re-pathed a
single-tenant deployment's files into per-user prefixes.
"""

import pytest

from sanzaru.server import IDENTITY_HEADER_ENV, UserContextMiddleware, identity_header_name
from sanzaru.user_context import get_user_context


@pytest.mark.unit
class TestIdentityHeaderIsOptIn:
    def test_unset_means_no_header_is_trusted(self, monkeypatch):
        monkeypatch.delenv(IDENTITY_HEADER_ENV, raising=False)
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
    def _scope(headers: list[tuple[bytes, bytes]]):
        return {"type": "http", "headers": headers}

    async def test_binds_the_identity_and_resets_it_after(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(get_user_context())

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([(b"x-forwarded-email", b"user@example.com")]), None, None)

        assert seen[0] is not None
        assert seen[0].email == "user@example.com"
        assert get_user_context() is None, "the context must not leak past the request"

    async def test_no_header_means_no_identity(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(get_user_context())

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([]), None, None)

        assert seen == [None]

    async def test_a_malformed_identity_is_dropped_not_fatal(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(get_user_context())

        middleware = UserContextMiddleware(app, header="x-forwarded-email")
        await middleware(self._scope([(b"x-forwarded-email", b"not-an-email")]), None, None)

        assert seen == [None]
