# SPDX-License-Identifier: MIT
"""Thin async client for api.higgsfield.ai.

Why not the official `higgsfield-client` SDK: its transport retries *every*
method — the generation POST included — on 408/429/5xx. A 500 on submit can
mean the job was already queued, and a successful generation is charged, so a
blind retry can bill twice for one request (Higgsfield's own docs: "avoid
repeating POST generations without idempotency keys"). It also raises one
exception type with no status code and never closes its httpx clients.

Retry policy — the one design decision in this module:

| call                                   | retried?                                              |
|----------------------------------------|-------------------------------------------------------|
| `submit` (POST /<slug>)                | **Never on an HTTP status.** Only when the request     |
|                                        | provably never left (ConnectError / ConnectTimeout),   |
|                                        | once. A failure after the body was sent (read timeout, |
|                                        | dropped connection) raises `kind="ambiguous_submit"`.  |
| `status`, `models`, `estimate`         | 5xx / 429 / transport errors, `_READ_ATTEMPTS` times,   |
| `cancel`, upload URL, presigned PUT    | backoff 2 s x1.5 (free or idempotent calls)            |
| output download (`stream_output`)      | no — the caller (download) decides                     |

Two httpx clients, never one: `_api` carries the credential and is pinned to
`BASE_URL`; `_bare` carries nothing and serves the presigned upload PUT and the
output download, so the key can never be sent to a storage or CDN host. There
is no base-URL override on purpose — an overridable endpoint is somewhere the
key can be redirected, and tests inject an `httpx.MockTransport` instead.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar, cast
from urllib.parse import urlsplit

import anyio
import httpx

from .errors import HiggsfieldAPIError, error_from_response
from .ids import validate_slug
from .types import CatalogModel, EstimateResponse, RequestStatus, SubmitResponse

BASE_URL = "https://api.higgsfield.ai"

_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=300.0, pool=10.0)
_READ_ATTEMPTS = 3
_BACKOFF_INITIAL_S = 2.0
_BACKOFF_FACTOR = 1.5
_EXPIRED_STATUSES = frozenset({403, 404, 410})

_T = TypeVar("_T")

# A request that never reached the server — safe to resend even for a submit.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout)
# The body may have been delivered and a job created — never resend a submit.
_MAYBE_SENT = (httpx.ReadTimeout, httpx.WriteError, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ReadError)


def _require_https(url: object, what: str) -> str:
    if not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
        raise HiggsfieldAPIError(
            f"Higgsfield returned a non-https {what}; refusing to use it",
            status_code=None,
            detail=str(url)[:200],
            kind="other",
        )
    return url


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:500]
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
            if value is not None:
                return json.dumps(value)[:500]
    return response.text[:500]


def _json_object(response: httpx.Response, what: str) -> dict[str, object]:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        body = None
    if not isinstance(body, dict):
        raise HiggsfieldAPIError(
            f"Higgsfield returned an unexpected {what} response",
            status_code=response.status_code,
            detail=response.text[:200],
            kind="other",
        )
    return cast("dict[str, object]", body)


def _transport_error(exc: httpx.TransportError) -> HiggsfieldAPIError:
    return HiggsfieldAPIError(
        f"Could not reach Higgsfield: {type(exc).__name__}: {exc}",
        status_code=None,
        detail=str(exc)[:200],
        kind="transport",
    )


def _uuid(request_id: str) -> str:
    """The id goes into a URL path; accept only a UUID, whatever the caller validated."""
    try:
        return str(uuid.UUID(request_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"request_id={request_id!r:.80} is not a Higgsfield request id (a UUID)") from None


class HiggsfieldClient:
    """Async client for the Higgsfield API. Build via `config.get_higgsfield_client()`."""

    def __init__(self, key_id: str, key_secret: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._key_id = key_id
        self._api = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Key {key_id}:{key_secret}", "Content-Type": "application/json"},
            timeout=_TIMEOUT,
            transport=transport,
        )
        self._bare = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, transport=transport)

    def __repr__(self) -> str:
        return f"HiggsfieldClient(key_id={self._key_id[:4]}…, secret=<redacted>)"

    async def aclose(self) -> None:
        await self._api.aclose()
        await self._bare.aclose()

    # ---------- retry plumbing ----------

    async def _with_read_retries(self, call: Callable[[], Awaitable[_T]]) -> _T:
        delay = _BACKOFF_INITIAL_S
        for attempt in range(1, _READ_ATTEMPTS + 1):
            try:
                return await call()
            except HiggsfieldAPIError as exc:
                if not exc.transient or attempt == _READ_ATTEMPTS:
                    raise
            await anyio.sleep(delay)
            delay *= _BACKOFF_FACTOR
        raise AssertionError("unreachable")  # pragma: no cover

    async def _request(self, method: str, path: str, *, body: dict[str, object] | None = None) -> httpx.Response:
        try:
            response = await self._api.request(method, path, json=body)
        except httpx.TransportError as exc:
            raise _transport_error(exc) from exc
        if response.is_error:
            raise error_from_response(response.status_code, _detail(response))
        return response

    # ---------- generation ----------

    async def submit(self, slug: str, arguments: dict[str, object]) -> SubmitResponse:
        """Queue a generation. Never retried once the request may have been sent."""
        path = "/" + validate_slug(slug)
        response: httpx.Response | None = None
        for attempt in (1, 2):
            try:
                response = await self._api.post(path, json=arguments)
                break
            except _NOT_SENT as exc:
                if attempt == 2:
                    raise _transport_error(exc) from exc
            except _MAYBE_SENT as exc:
                raise HiggsfieldAPIError(
                    "The submit request failed after it was sent, so the job may have been created. "
                    "Check the Higgsfield dashboard (or wait for it) before resubmitting.",
                    status_code=None,
                    detail=f"{type(exc).__name__}: {exc}"[:200],
                    kind="ambiguous_submit",
                ) from exc
            except httpx.TransportError as exc:
                raise _transport_error(exc) from exc
        assert response is not None
        if response.is_error:
            raise error_from_response(response.status_code, _detail(response))
        body = _json_object(response, "submit")
        if not isinstance(body.get("request_id"), str):
            raise HiggsfieldAPIError(
                "Higgsfield accepted the request but returned no request_id",
                status_code=response.status_code,
                detail=response.text[:200],
                kind="other",
            )
        return cast("SubmitResponse", body)

    async def status(self, request_id: str) -> RequestStatus:
        request_id = _uuid(request_id)

        async def call() -> RequestStatus:
            response = await self._request("GET", f"/requests/{request_id}/status")
            return cast("RequestStatus", _json_object(response, "status"))

        return await self._with_read_retries(call)

    async def cancel(self, request_id: str) -> None:
        request_id = _uuid(request_id)

        async def call() -> None:
            await self._request("POST", f"/requests/{request_id}/cancel")

        await self._with_read_retries(call)

    async def estimate(self, slug: str, arguments: dict[str, object]) -> EstimateResponse:
        path = "/estimate/" + validate_slug(slug)

        async def call() -> EstimateResponse:
            response = await self._request("POST", path, body=arguments)
            return cast("EstimateResponse", _json_object(response, "estimate"))

        return await self._with_read_retries(call)

    async def models(self) -> list[CatalogModel]:
        """The account's catalog (GET /models — undocumented; not on any generation path)."""

        async def call() -> list[CatalogModel]:
            body = _json_object(await self._request("GET", "/models"), "models")
            items = body.get("items")
            return cast("list[CatalogModel]", items if isinstance(items, list) else [])

        return await self._with_read_retries(call)

    # ---------- media ----------

    async def upload(self, data: bytes, content_type: str) -> str:
        """Upload bytes through a presigned URL and return the public https URL to pass as input."""

        async def get_url() -> dict[str, object]:
            response = await self._request("POST", "/files/generate-upload-url", body={"content_type": content_type})
            return _json_object(response, "upload URL")

        grant = await self._with_read_retries(get_url)
        public_url = _require_https(grant.get("public_url"), "public_url")
        upload_url = _require_https(grant.get("upload_url"), "upload_url")
        raw_headers = grant.get("upload_headers")
        headers = {str(k): str(v) for k, v in raw_headers.items()} if isinstance(raw_headers, dict) else {}

        async def put() -> None:
            try:
                response = await self._bare.put(upload_url, content=data, headers=headers)
            except httpx.TransportError as exc:
                raise _transport_error(exc) from exc
            if response.is_error:
                raise error_from_response(response.status_code, _detail(response))

        await self._with_read_retries(put)
        return public_url

    @asynccontextmanager
    async def stream_output(self, url: str) -> AsyncIterator[AsyncIterator[bytes]]:
        """Stream a completed job's output file (https only, no credentials sent)."""
        _require_https(url, "output URL")
        try:
            async with self._bare.stream("GET", url) as response:
                if response.status_code in _EXPIRED_STATUSES:
                    raise HiggsfieldAPIError(
                        "The output is no longer available (Higgsfield keeps outputs at least 7 days) — regenerate it",
                        status_code=response.status_code,
                        detail="output expired or removed",
                        kind="not_found",
                    )
                if response.is_error:
                    await response.aread()
                    raise error_from_response(response.status_code, _detail(response))
                yield response.aiter_bytes()
        except httpx.TransportError as exc:
            raise _transport_error(exc) from exc
