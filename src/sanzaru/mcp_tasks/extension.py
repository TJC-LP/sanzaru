# SPDX-License-Identifier: MIT
"""The `io.modelcontextprotocol/tasks` extension (SEP-2663), on the SDK's generic Extension API.

mcp 2.3 ships no Tasks implementation (python-sdk #2806), but its `Extension`
interface carries everything SEP-2663 needs: `intercept_tool_call` may answer a
`tools/call` with a `CreateTaskResult` instead of running the tool inline, and
`MethodBinding`s serve `tasks/get`, `tasks/update` and `tasks/cancel`. The runner
passes a non-core `resultType` ("task") through untouched on the 2026-07-28 wire.

When a call becomes a task, decided per request and never on a legacy wire:

- **The client declared the extension on this request.** SEP-2663 forbids
  returning a task to a client that did not; such a client gets the ordinary
  `CallToolResult`, exactly as with the extension off.
- **The tool has a route** (`TaskRoute`). There are two kinds:

  *Provider-backed* (`create_video`, `edit_video`, `extend_video`, `create_image`).
  The tool runs inline as it always did. It submits the job and returns at once,
  and the job id becomes a task. `tasks/get` reads the provider's status. On
  completion it downloads the output, the same thing `wait_for(download=true)`
  does, and records the result. No work runs in this process, so these tasks
  survive a restart and are answered by any replica. A dry run or a refused job
  submitted nothing and stays a plain result.

  *Background* (podcasts, `generate_image`, `edit_image`, TTS, transcription).
  These have no provider-side job, so `BackgroundRunner` runs the tool in a task
  group that lives as long as the server, not the request. Heartbeats keep the
  record current; a record whose worker stopped beating is reported failed
  instead of `working` forever.

Results follow SEP-2663's split: a tool that *ran and failed* (`isError`, a
moderation rejection, a failed render) is `completed` with an error result;
`failed` is reserved for JSON-RPC-level errors (a crash, a lost worker).
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import anyio
from anyio.abc import TaskGroup
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolRequestParams, CallToolResult, RequestParams, TextContent
from mcp_types.jsonrpc import INTERNAL_ERROR, INVALID_PARAMS
from pydantic import Field

from ..config import logger
from ..storage.protocol import PathType
from . import store
from .store import TaskRecord, UnknownTaskError, iso

TASKS_EXTENSION = "io.modelcontextprotocol/tasks"
_CLIENT_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"

#: Each process tags the background tasks it runs, so `tasks/get` can tell "mine,
#: still running" from "someone else's, maybe dead".
PROCESS_ID = secrets.token_hex(8)

#: A background task rewrites its record this often, and checks for a cancel request.
HEARTBEAT_SECONDS = 20.0
#: A `working` background record not rewritten for this long is reported lost.
#: Six missed beats: generous enough that a slow storage write is not a death.
STALE_AFTER_SECONDS = 120.0

RouteKind = Literal["video", "image", "background"]


@dataclass(frozen=True)
class TaskRoute:
    """How one tool becomes a task."""

    kind: RouteKind
    path_type: PathType
    poll_interval_ms: int


class TaskIdParams(RequestParams):
    task_id: str = Field(min_length=1, max_length=256)


class TaskUpdateParams(TaskIdParams):
    input_responses: dict[str, object] | None = None


def client_accepts_tasks(ctx: ServerRequestContext[object, object]) -> bool:
    """Whether *this request* declared the tasks extension (SEP-2663: per request, in `_meta`)."""
    meta = ctx.meta
    if not meta:
        return False
    capabilities = meta.get(_CLIENT_CAPABILITIES_META)
    if not isinstance(capabilities, dict):
        return False
    extensions = capabilities.get("extensions")
    return isinstance(extensions, dict) and TASKS_EXTENSION in extensions


# ---------------------------------------------------------------------------
# Wire shapes
# ---------------------------------------------------------------------------


def _task_fields(record: TaskRecord) -> dict[str, object]:
    fields: dict[str, object] = {
        "taskId": record.task_id,
        "status": record.status,
        "createdAt": iso(record.created_at),
        "lastUpdatedAt": iso(record.updated_at),
        "ttlMs": record.ttl_ms,
        "pollIntervalMs": record.poll_interval_ms,
    }
    if record.status_message:
        fields["statusMessage"] = record.status_message
    return fields


def create_task_result(record: TaskRecord) -> dict[str, object]:
    """What `tools/call` returns when the call became a task."""
    return {"resultType": "task", **_task_fields(record)}


def detailed_task(record: TaskRecord) -> dict[str, object]:
    """What `tasks/get` returns: the task, plus its result or error once it has one."""
    fields = _task_fields(record)
    if record.status == "completed" and record.result is not None:
        fields["result"] = record.result
    if record.status == "failed":
        fields["error"] = record.error or {"code": INTERNAL_ERROR, "message": "Task failed"}
    return fields


def _wire_result(result: CallToolResult) -> dict[str, object]:
    return result.model_dump(by_alias=True, mode="json", exclude_none=True)


def _json_result(payload: dict[str, object]) -> dict[str, object]:
    """A successful result in the shape the inline tool call returns: the dict as JSON text.

    The tool wrappers in `server.py` declare no return type, so the SDK advertises
    no output schema and sends no `structuredContent`; matching that keeps a task's
    result indistinguishable from the same call answered inline.
    """
    text = json.dumps(payload, indent=2, default=str)
    return _wire_result(CallToolResult(content=[TextContent(type="text", text=text)]))


def _result_payload(result: CallToolResult) -> dict[str, object] | None:
    """The tool's returned dict: `structuredContent` when present, else its single JSON text block."""
    if isinstance(result.structured_content, dict):
        return result.structured_content
    if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
        return None
    try:
        payload = json.loads(result.content[0].text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _error_result(message: str) -> dict[str, object]:
    return _wire_result(CallToolResult(content=[TextContent(type="text", text=message)], is_error=True))


def _unknown_task() -> MCPError:
    # SEP-2663's wording for the case, and -32602 so a client treats it as final.
    return MCPError(code=INVALID_PARAMS, message="Failed to retrieve task: Task not found")


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Poll:
    status: Literal["working", "done", "tool_error", "cancelled"]
    message: str


async def _poll_video(job_id: str) -> _Poll:
    from ..tools import video

    status = await video.get_video_status(job_id)
    state = status["status"]
    if state == "completed":
        return _Poll("done", f"Higgsfield job {job_id} completed; downloading")
    if state in {"failed", "nsfw"}:
        return _Poll("tool_error", f"Higgsfield job {job_id} {state}: {status['error'] or 'no detail given'}")
    if state == "canceled":
        return _Poll("cancelled", f"Higgsfield job {job_id} canceled")
    return _Poll("working", f"Higgsfield job {job_id}: {state} (renders often take several minutes)")


async def _poll_image(job_id: str) -> _Poll:
    from ..tools import image

    status = await image.get_image_status(job_id)
    state = status["status"]
    if state == "completed":
        return _Poll("done", f"Image job {job_id} completed; downloading")
    if state in {"failed", "incomplete"}:
        return _Poll("tool_error", f"Image job {job_id} {state}")
    if state == "cancelled":
        return _Poll("cancelled", f"Image job {job_id} cancelled")
    return _Poll("working", f"Image job {job_id}: {state}")


async def _download(kind: RouteKind, job_id: str) -> str:
    """Save a finished job's output; returns the stored filename (deterministic, so a re-run overwrites)."""
    if kind == "video":
        from ..tools import video

        return (await video.download_video(job_id))["filename"]
    from ..tools import image

    return (await image.download_image(job_id))["filename"]


async def _cancel_provider(kind: RouteKind, job_id: str) -> None:
    if kind == "video":
        from ..tools import video

        await video.cancel_video(job_id)
        return
    from ..config import get_client
    from ..utils import validate_resource_id

    await get_client().responses.cancel(validate_resource_id(job_id, "response_id"))


_POLLERS: dict[RouteKind, Callable[[str], Awaitable[_Poll]]] = {"video": _poll_video, "image": _poll_image}


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

ToolCall = Callable[[], Awaitable[HandlerResult]]


def _elapsed(start: float) -> str:
    seconds = int(time.monotonic() - start)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


class BackgroundRunner:
    """Runs background tasks in a task group that outlives the request that started them.

    `running()` is entered once, from the server lifespan. Outside it `available`
    is False and the extension runs such calls inline instead. That keeps the
    server correct when it is embedded somewhere that never enters the lifespan.
    """

    def __init__(self) -> None:
        self._tg: TaskGroup | None = None
        self._active: dict[str, anyio.CancelScope] = {}
        self._cancelled: set[str] = set()

    @asynccontextmanager
    async def running(self):  # noqa: ANN201 - an async context manager
        async with anyio.create_task_group() as tg:
            self._tg = tg
            try:
                yield self
            finally:
                self._tg = None
                tg.cancel_scope.cancel()

    @property
    def available(self) -> bool:
        return self._tg is not None

    def owns(self, task_id: str) -> bool:
        return task_id in self._active

    def start(self, record: TaskRecord, call: ToolCall) -> None:
        """Spawn the work. Called from inside the request, so the task inherits its contextvars.

        That inheritance is load-bearing: the caller's `UserContext` is what scopes
        storage to their directory on a multi-tenant backend, for the tool *and*
        for every record write the task makes.
        """
        if self._tg is None:
            raise RuntimeError("background runner is not running")
        self._tg.start_soon(self._run, record, call)

    def cancel(self, task_id: str) -> bool:
        scope = self._active.get(task_id)
        if scope is None:
            return False
        self._cancelled.add(task_id)
        scope.cancel()
        return True

    async def _run(self, record: TaskRecord, call: ToolCall) -> None:
        started = time.monotonic()
        outcome: list[HandlerResult] = []
        errors: list[BaseException] = []
        task_id = record.task_id
        try:
            with anyio.CancelScope() as scope:
                self._active[task_id] = scope
                async with anyio.create_task_group() as inner:

                    async def work() -> None:
                        try:
                            outcome.append(await call())
                        except Exception as exc:  # noqa: BLE001 - recorded as the task's failure below
                            errors.append(exc)
                        inner.cancel_scope.cancel()

                    async def heartbeat() -> None:
                        while True:
                            await anyio.sleep(HEARTBEAT_SECONDS)
                            if await store.cancel_requested(record):
                                self._cancelled.add(task_id)
                                scope.cancel()
                                return
                            record.status_message = f"{record.tool}: running for {_elapsed(started)}"
                            try:
                                await store.save(record)
                            except Exception:  # noqa: BLE001 - a missed beat is not a failed task
                                logger.warning("task %s: heartbeat write failed", record.tool, exc_info=True)

                    inner.start_soon(work)
                    inner.start_soon(heartbeat)
        except anyio.get_cancelled_exc_class():
            # The server task group is being torn down: the work is gone. Say so
            # rather than leave a `working` record for the staleness check to find.
            with anyio.CancelScope(shield=True):
                await self._settle_failed(record, "The server shut down before this task finished; run it again.")
            raise
        finally:
            self._active.pop(task_id, None)

        with anyio.CancelScope(shield=True):
            if outcome:
                await self._settle_result(record, outcome[0], started)
            elif errors:
                exc = errors[0]
                if isinstance(exc, MCPError):
                    record.error = {"code": exc.error.code, "message": exc.error.message}
                else:
                    logger.error("task %s crashed", record.tool, exc_info=exc)
                    record.error = {"code": INTERNAL_ERROR, "message": f"{record.tool} crashed: {exc}"[:500]}
                record.status = "failed"
                record.status_message = None
                await self._save_quietly(record)
            elif task_id in self._cancelled:
                record.status = "cancelled"
                record.status_message = f"Cancelled after {_elapsed(started)}"
                await self._save_quietly(record)
        self._cancelled.discard(task_id)

    async def _settle_result(self, record: TaskRecord, result: HandlerResult, started: float) -> None:
        if isinstance(result, CallToolResult):
            record.status = "completed"
            record.result = _wire_result(result)
            record.status_message = f"Finished in {_elapsed(started)}"
        else:
            # An MRTR interim result: the tool asked the client for input, which a
            # background task cannot relay yet. No sanzaru tool does this today.
            record.status = "failed"
            record.error = {
                "code": INTERNAL_ERROR,
                "message": f"{record.tool} asked for input, which tasks do not relay",
            }
        await self._save_quietly(record)

    async def _settle_failed(self, record: TaskRecord, message: str) -> None:
        record.status = "failed"
        record.error = {"code": INTERNAL_ERROR, "message": message}
        record.status_message = None
        await self._save_quietly(record)

    @staticmethod
    async def _save_quietly(record: TaskRecord) -> None:
        try:
            await store.save(record)
        except Exception:  # noqa: BLE001 - nothing left to report it to; the staleness check covers it
            logger.exception("task %s: could not write its final state", record.tool)


# ---------------------------------------------------------------------------
# The extension
# ---------------------------------------------------------------------------


class Tasks(Extension):
    """Serve SEP-2663 for the routed tools; every other call passes through untouched."""

    identifier = TASKS_EXTENSION

    def __init__(self, routes: Mapping[str, TaskRoute], runner: BackgroundRunner) -> None:
        self._routes = dict(routes)
        self._runner = runner
        self._locks: dict[str, anyio.Lock] = {}

    @property
    def routes(self) -> Mapping[str, TaskRoute]:
        return self._routes

    def methods(self) -> Sequence[MethodBinding]:
        return (
            MethodBinding("tasks/get", TaskIdParams, self._get),
            MethodBinding("tasks/update", TaskUpdateParams, self._update),
            MethodBinding("tasks/cancel", TaskIdParams, self._cancel),
        )

    # -- tools/call ---------------------------------------------------------

    async def intercept_tool_call(
        self,
        params: CallToolRequestParams,
        ctx: ServerRequestContext[object, object],
        call_next: CallNext,
    ) -> HandlerResult:
        route = self._routes.get(params.name)
        if route is None or not client_accepts_tasks(ctx):
            return await call_next(ctx)
        if route.kind == "background":
            if not self._runner.available:
                return await call_next(ctx)
            return await self._start_background(params.name, route, ctx, call_next)
        return await self._track_provider_job(params.name, route, await call_next(ctx))

    async def _start_background(
        self, tool: str, route: TaskRoute, ctx: ServerRequestContext[object, object], call_next: CallNext
    ) -> HandlerResult:
        record = TaskRecord(
            task_id=store.new_task_id(route.path_type),
            kind="background",
            tool=tool,
            path_type=route.path_type,
            owner=store.current_owner(),
            status_message=f"{tool}: started",
            poll_interval_ms=route.poll_interval_ms,
            worker=PROCESS_ID,
        )
        # Durable before returned (SEP-2663): the id must resolve on the first tasks/get.
        await store.save(record)
        self._runner.start(record, lambda: call_next(ctx))
        return create_task_result(record)

    async def _track_provider_job(self, tool: str, route: TaskRoute, result: HandlerResult) -> HandlerResult:
        if not isinstance(result, CallToolResult) or result.is_error:
            return result
        structured = _result_payload(result)
        if structured is None:
            return result
        job_id = structured.get("id")
        if not isinstance(job_id, str) or not job_id or structured.get("status") == "dry_run":
            return result  # a dry run or a refusal submitted nothing: nothing to wait on
        record = TaskRecord(
            task_id=store.new_task_id(route.path_type),
            kind=route.kind,
            tool=tool,
            path_type=route.path_type,
            owner=store.current_owner(),
            job_id=job_id,
            job=dict(structured),
            status_message=f"Submitted {job_id}",
            poll_interval_ms=route.poll_interval_ms,
        )
        await store.save(record)
        return create_task_result(record)

    # -- tasks/* --------------------------------------------------------------

    async def _load(self, task_id: str) -> TaskRecord:
        try:
            return await store.load(task_id)
        except UnknownTaskError:
            raise _unknown_task() from None

    async def _get(self, ctx: ServerRequestContext[object, object], params: TaskIdParams) -> HandlerResult:
        record = await self._load(params.task_id)
        if not record.terminal:
            if record.kind == "background":
                record = await self._check_alive(record)
            else:
                record = await self._refresh_provider(record)
        return detailed_task(record)

    async def _update(self, ctx: ServerRequestContext[object, object], params: TaskUpdateParams) -> HandlerResult:
        # No sanzaru task enters input_required, so nothing is ever outstanding;
        # SEP-2663 says keys that are not outstanding are ignored.
        await self._load(params.task_id)
        return {}

    async def _cancel(self, ctx: ServerRequestContext[object, object], params: TaskIdParams) -> HandlerResult:
        record = await self._load(params.task_id)
        if record.terminal:
            return {}
        if record.kind == "background":
            if not self._runner.cancel(record.task_id):
                # Another process runs it: leave a marker its heartbeat will find.
                await store.request_cancel(record)
            return {}
        assert record.job_id is not None
        try:
            await _cancel_provider(record.kind, record.job_id)
        except Exception as exc:  # noqa: BLE001 - cooperative: the job may simply be past cancelling
            record.status_message = f"Cancel not accepted: {exc}"[:300]
            await store.save(record)
        return {}

    async def _check_alive(self, record: TaskRecord) -> TaskRecord:
        if self._runner.owns(record.task_id):
            return record
        if time.time() - record.updated_at <= STALE_AFTER_SECONDS:
            return record
        record.status = "failed"
        record.status_message = None
        record.error = {
            "code": INTERNAL_ERROR,
            "message": f"{record.tool} stopped reporting progress (the server that ran it restarted or died); "
            "run it again",
        }
        await store.save(record)
        return record

    async def _refresh_provider(self, record: TaskRecord) -> TaskRecord:
        # One refresh per task at a time in this process: completion downloads,
        # and two concurrent polls must not both stream the same output.
        lock = self._locks.setdefault(record.task_id, anyio.Lock())
        async with lock:
            try:
                record = await store.load(record.task_id)
            except UnknownTaskError:
                raise _unknown_task() from None
            job_id = record.job_id
            if record.terminal or job_id is None:
                return record
            try:
                poll = await _POLLERS[record.kind](job_id)
            except Exception as exc:  # noqa: BLE001 - a failed status read is not a failed job
                logger.warning("task %s: status check failed", record.task_id[:8], exc_info=True)
                record.status_message = f"Status check failed, will retry: {exc}"[:300]
                return record
            if poll.status == "working":
                if poll.message != record.status_message:
                    record.status_message = poll.message
                    await store.save(record)
                return record
            if poll.status == "done":
                try:
                    filename = await _download(record.kind, job_id)
                except Exception as exc:  # noqa: BLE001 - retried on the next poll
                    logger.warning("task %s: download failed", record.task_id[:8], exc_info=True)
                    record.status_message = f"Finished; download failed, will retry: {exc}"[:300]
                    return record
                record.status = "completed"
                record.result = _json_result({**(record.job or {}), "status": "completed", "file": filename})
                record.status_message = f"Saved {filename}"
            elif poll.status == "tool_error":
                record.status = "completed"
                record.result = _error_result(poll.message)
                record.status_message = poll.message
            else:
                record.status = "cancelled"
                record.status_message = poll.message
            await store.save(record)
            self._locks.pop(record.task_id, None)
            return record
