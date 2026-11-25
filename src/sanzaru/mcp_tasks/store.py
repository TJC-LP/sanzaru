# SPDX-License-Identifier: MIT
"""Task records: where an MCP task's state lives between requests.

The server is stateless, so a task has to be answerable by any request on any
replica: the record goes in the storage backend, next to the media it is about.
That is the same place `simrun_<id>.json` manifests already live, and on the
Databricks backend it inherits the per-user volume prefix, so one tenant's
records are not even addressable from another's requests.

Three properties carry the security model, and each is checked here rather than
by callers:

- **The task id is a bearer token.** SEP-2663 requires ids that cannot be
  enumerated or guessed, so the id carries 192 random bits. There is no
  `tasks/list`, by design of the spec, and nothing here enumerates records.
- **The record's filename is a digest of the id, never the id.** Storage
  listings (the media resource completion, `/media`) can show filenames; showing
  a digest gives away nothing that `tasks/get` would accept.
- **Every read checks the owner.** The record stores the identity of the request
  that created it, and a read from any other identity is answered exactly like
  an unknown id, so the response does not confirm that a task exists.

What the record does *not* do: hold the work. A provider-backed task (video,
async image) is the provider's job; the record only remembers what was submitted
and, once finished, the result. A background task's work runs in this process
(see `extension.BackgroundRunner`) and the record is how other requests see it.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from datetime import UTC, datetime
from typing import Literal

import anyio
from pydantic import BaseModel, Field

from ..storage.factory import get_storage
from ..storage.protocol import PathType
from ..user_context import get_user_context

#: Bumped when the record shape changes incompatibly; older records read as unknown.
RECORD_VERSION = 1

TaskStatus = Literal["working", "input_required", "completed", "cancelled", "failed"]
TaskKind = Literal["video", "image", "background"]

TERMINAL: frozenset[str] = frozenset({"completed", "cancelled", "failed"})

#: How long a record answers `tasks/get`. Seven days matches how long Higgsfield
#: keeps outputs, which bounds how long a video task is meaningful anyway. The
#: storage protocol has no delete, so an expired record stays on disk and simply
#: stops resolving (SEP-2663 lets a server purge after the TTL).
DEFAULT_TTL_MS = 7 * 24 * 60 * 60 * 1000

_RECORD_PREFIX = "sanzaru-task_"

#: One letter per path type, carried in the id so `tasks/get` knows which
#: directory to read without a lookup table.
_PATH_CODES: dict[PathType, str] = {"video": "v", "reference": "r", "audio": "a"}
_CODE_PATHS: dict[str, PathType] = {code: path for path, code in _PATH_CODES.items()}

_ID_PREFIX = "tk"

#: A record rewrite is not atomic on the storage protocol (truncate, then write),
#: so a concurrent read can land mid-write. Read again this many times in total.
_TORN_READ_ATTEMPTS = 4
_TORN_READ_BACKOFF_SECONDS = 0.05
_TOKEN_BYTES = 24  # 192 bits: well past enumeration, short enough to quote


class UnknownTaskError(LookupError):
    """No record answers this id for this caller: unknown, expired, or someone else's.

    The three are deliberately indistinguishable to the caller.
    """


class TaskRecord(BaseModel):
    """One task, as persisted. Field names are internal; `extension` maps them to the wire."""

    version: int = RECORD_VERSION
    task_id: str
    kind: TaskKind
    tool: str
    path_type: PathType
    owner: str | None
    status: TaskStatus = "working"
    status_message: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    ttl_ms: int = DEFAULT_TTL_MS
    poll_interval_ms: int = 5000
    #: Provider job id (`hf_<uuid>` / `resp_...`) for provider-backed tasks.
    job_id: str | None = None
    #: The tool's structured result at submission; the completed result builds on it.
    job: dict[str, object] | None = None
    #: The final `CallToolResult`, wire-shaped, once `status == "completed"`.
    result: dict[str, object] | None = None
    #: A JSON-RPC error object (`code`, `message`), once `status == "failed"`.
    error: dict[str, object] | None = None
    #: Which process runs a background task; heartbeats keep `updated_at` fresh.
    worker: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def expired(self, now: float | None = None) -> bool:
        return ((now if now is not None else time.time()) - self.created_at) * 1000 > self.ttl_ms

    def touch(self) -> None:
        self.updated_at = time.time()


def iso(ts: float) -> str:
    """ISO 8601 UTC, the timestamp form SEP-2663 uses for `createdAt` / `lastUpdatedAt`."""
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def current_owner() -> str | None:
    """The identity a record is bound to: the request's user, or None on a single-user server."""
    ctx = get_user_context()
    return ctx.email if ctx is not None else None


def new_task_id(path_type: PathType) -> str:
    return f"{_ID_PREFIX}{_PATH_CODES[path_type]}_{secrets.token_urlsafe(_TOKEN_BYTES)}"


def path_type_of(task_id: str) -> PathType:
    """The directory a task's record lives in, read from its id; unknown shapes are unknown tasks."""
    if len(task_id) < 5 or not task_id.startswith(_ID_PREFIX) or task_id[3] != "_":
        raise UnknownTaskError(task_id)
    path_type = _CODE_PATHS.get(task_id[2])
    if path_type is None:
        raise UnknownTaskError(task_id)
    return path_type


def record_name(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:40]
    return f"{_RECORD_PREFIX}{digest}.json"


def cancel_marker_name(task_id: str) -> str:
    """A separate file, so a cancel never races a heartbeat rewriting the record itself."""
    return record_name(task_id).removesuffix(".json") + ".cancel.json"


async def save(record: TaskRecord) -> None:
    record.touch()
    await get_storage().write(record.path_type, record_name(record.task_id), record.model_dump_json().encode("utf-8"))


async def load(task_id: str) -> TaskRecord:
    """Read a task for the current caller, or raise `UnknownTaskError`.

    Owner mismatch, an expired TTL, a record from an incompatible version and a
    file that does not parse all read as unknown: none of them is something the
    caller can act on except by not asking again.
    """
    path_type = path_type_of(task_id)
    storage = get_storage()
    name = record_name(task_id)
    data: dict[str, object] | None = None
    for attempt in range(_TORN_READ_ATTEMPTS):
        try:
            raw = await storage.read(path_type, name)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise UnknownTaskError(task_id) from exc
        try:
            parsed = json.loads(raw)
        except ValueError:
            # Storage writes truncate then fill, so a read racing a heartbeat or
            # the final write can see a partial file. That is a torn read, not a
            # missing task: wait a beat and read again before giving up.
            if attempt + 1 < _TORN_READ_ATTEMPTS:
                await anyio.sleep(_TORN_READ_BACKOFF_SECONDS * (attempt + 1))
            continue
        data = parsed if isinstance(parsed, dict) else None
        break
    if data is None or data.get("version") != RECORD_VERSION:
        raise UnknownTaskError(task_id)
    try:
        record = TaskRecord.model_validate(data)
    except ValueError as exc:
        raise UnknownTaskError(task_id) from exc
    # The stored id must be the one asked for: a record copied over another
    # task's filename (same tenant, same directory) does not become that task.
    if not secrets.compare_digest(record.task_id.encode(), task_id.encode()):
        raise UnknownTaskError(task_id)
    if record.owner != current_owner() or record.expired():
        raise UnknownTaskError(task_id)
    return record


async def request_cancel(record: TaskRecord) -> None:
    await get_storage().write(record.path_type, cancel_marker_name(record.task_id), b"{}")


async def cancel_requested(record: TaskRecord) -> bool:
    try:
        return await get_storage().exists(record.path_type, cancel_marker_name(record.task_id))
    except (OSError, ValueError):
        return False
