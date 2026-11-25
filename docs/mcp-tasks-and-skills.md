# MCP Tasks and Skills

Two MCP extensions that Claude supports as of 2026-11-24, and how sanzaru serves them.

| | Tasks (`io.modelcontextprotocol/tasks`, SEP-2663) | Skills (`io.modelcontextprotocol/skills`, SEP-2640) |
|---|---|---|
| Default | **Off.** Opt in with `SANZARU_TASKS=1` | **On** |
| What it changes | Slow tool calls may return a task; the client polls `tasks/get` and delivers the result | Adds `skills/list`, `skills/get`, and `skill://` resources |
| Wire | 2026-07-28 only (HTTP, or stdio opened with the modern envelope) | Both wires; `ttlMs`/`cacheScope` only on 2026-07-28 |
| Clients without it | Unchanged: the ordinary result, inline | Unchanged: a few extra resources in `resources/list` |
| Code | `src/sanzaru/mcp_tasks/` | `src/sanzaru/mcp_skills.py`, `src/sanzaru/skills/` |

Neither extension is in a released `mcp` SDK. 2.3.0 has no Tasks implementation (python-sdk #2806),
and the SDK's `Skills` class landed on main after 2.3.0. Both are built on the SDK's generic
`Extension` / `MethodBinding` API, the same API the SDK's own versions use, so swapping them in
later is local to those files.

## Tasks

### What becomes a task

A call becomes a task only when **both** of these hold:

1. **The request declared the extension.** The check is per request, on
   `_meta["io.modelcontextprotocol/clientCapabilities"].extensions`. SEP-2663 forbids returning
   a task to a client that did not declare it.
2. **The tool has a route** (`TASK_ROUTES` in `server.py`).

| Kind | Tools | Who runs the work | Poll interval |
|---|---|---|---|
| Provider-backed | `create_video`, `edit_video`, `extend_video` | Higgsfield | 10 s |
| Provider-backed | `create_image` | OpenAI (Responses, `background=True`) | 3 s |
| Background | `generate_podcast` | this server | 5 s |
| Background | `simulate_podcast` | this server | 10 s |
| Background | `generate_image`, `edit_image`, `create_audio`, `transcribe_audio`, `transcribe_with_enhancement`, `chat_with_audio` | this server | 3 s |

Some calls stay ordinary results even with Tasks on:

- **Provider-backed calls that submitted nothing.** That covers a dry run, a refusal over
  `max_cost_usd`, and any error. There is no job to wait on.
- **Background calls when the server lifespan is not running**, for example when the server is
  embedded somewhere that never enters it. These run inline as before.

### Lifecycle

| Event | Provider-backed | Background |
|---|---|---|
| `tools/call` | Runs the tool inline. It submits and returns at once. The returned job is recorded and the call answers `resultType: "task"`. | Writes the record, starts the work in the server-lifetime task group, and answers `resultType: "task"` at once. |
| `tasks/get` while working | Reads the provider's status and reports it in `statusMessage`. | Answers from the record. A heartbeat rewrites it every 20 s. |
| Finished | Downloads the output, as `wait_for(download=true)` would, then records the result. The result is the submitted job plus `status: "completed"` and `file`. | Records the tool's `CallToolResult`. |
| The tool ran and failed | `completed`, with an `isError` result (e.g. a moderation rejection). | `completed`, with the tool's `isError` result. |
| JSON-RPC-level failure | Not applicable: a failed status read keeps the task `working` and is retried on the next poll. | `failed`, for a crash, a server shutdown mid-task, or a worker silent for 120 s. |
| `tasks/cancel` | Asks the provider to cancel. The next poll that sees the cancellation reports `cancelled`. | Cancels in place if this process runs it. Otherwise writes a marker that the owning process's heartbeat picks up. |
| `tasks/update` | Accepted and ignored. No sanzaru task enters `input_required`. | Same. |

`completed` versus `failed` follows SEP-2663 exactly. `failed` is reserved for JSON-RPC
errors. A tool that ran and reported failure is `completed` with `isError: true`, the same thing
the inline call would have returned.

### Records

Every task, of either kind, has one small JSON record in the storage backend, in the directory
of the tool's media type. It sits next to the `simrun_<id>.json` manifests.

- **The task id is a bearer token.** It has the form `tk<v|r|a>_<192 random bits>`; the letter
  is the path type. There is no `tasks/list`, by design of the spec.
- **The file name is a digest of the id**, `sanzaru-task_<sha256[:40]>.json`, so a storage
  listing reveals nothing that `tasks/get` accepts. Caller-chosen output names that match the
  pattern are refused (`reject_reserved_name`).
- **Every read checks the owner.** The record stores the identity of the request that created
  it. A different identity, an expired TTL (7 days), or an unparseable record all answer
  `-32602 "Failed to retrieve task: Task not found"`, so the response does not confirm that a
  task exists.
- **Multi-tenant storage applies unchanged.** Background work inherits the request's
  contextvars, so on Databricks the record and the tool's output both land under the caller's
  volume prefix.
- **Storage writes are not atomic** (truncate, then write), so a read that races a heartbeat
  can see a partial file. `store.load` re-reads a record that does not parse, up to 4 times,
  before calling it unknown.
- **Records are never deleted.** The storage protocol has no delete operation, the same as
  run checkpoints. An expired record stops resolving.

What this means operationally:

- **Provider-backed tasks survive a restart** and are answered by any replica, because the
  provider holds the job and the record holds the rest.
- **Background tasks die with their process.** The next `tasks/get` after the heartbeat goes
  stale reports `failed`, so a task never stays `working` forever. A simulated podcast keeps its
  own checkpoints and `resume` path regardless.
- **Cancelling a task on another replica is cooperative**: the owner process sees the cancel
  marker within one heartbeat.

### Why opt-in

The extension is new. sanzaru implements it on the SDK's generic API rather than through the
SDK itself, and Claude made Tasks generally available only on 2026-11-24. Turning it on is a
deployment decision, not an upgrade side effect. The switch is a strict boolean, like
`SANZARU_REQUIRE_USER_CONTEXT`: `1/true/yes/on` or `0/false/no/off`, and anything else refuses
to start.

### Client support (as documented 2026-11-24)

| Surface | Tasks |
|---|---|
| Claude Code (terminal, VS Code, desktop app local sessions) | Yes, v2.1.304+, main conversation |
| claude.ai chats and cloud sessions | Yes, through claude.ai connectors (not Projects, incognito or the Cowork option) |
| `claude -p`, Desktop `claude_desktop_config.json` servers, `.mcp.json`/plugin servers in cloud sessions | No: the ordinary result |

Claude Code clamps the poll interval to 250–30000 ms. Cancelling a task in `/tasks` (key `x`)
sends `tasks/cancel`.

## Skills

The bundled skills are package data under `src/sanzaru/skills/<name>/`:

| Skill | Served over MCP | In the plugin |
|---|---|---|
| `sanzaru-mcp` | yes | yes |
| `prompt-guidance` | yes | yes |
| `podcast-production` | yes | yes |
| `sanzaru-cli` | **no**: it teaches a shell command, which a hosted connector cannot run | yes |

Every file is served twice. It is a plain resource at `skill://<name>/<path>` (`text/markdown`,
audience `assistant`), which any client can read. Through the extension, `skills/list` and
`skills/get` describe each skill as its frontmatter plus a complete manifest of
`{uri, digest: "sha256:…", size}` entries, `SKILL.md` included.

- **The catalog is static**, computed once at import, so responses carry
  `cacheScope: "public"` and an hour of `ttlMs`.
- **`directoryRead` is not offered**, because the manifests are complete.
- **Validation happens at import.** The frontmatter is single-line `key: value` with a
  non-empty `name` and `description`. The name must equal the directory name, which SEP-2640
  requires of the URI's final segment. The spec's per-skill limits are enforced.
- **The plugin carries byte-identical copies.** A marketplace-installed plugin cannot read
  outside its own directory. Edit one side and copy the directory to the other;
  `test_the_bundled_skills_match_the_plugin_copies_byte_for_byte` fails until both agree.

Claude Code loads MCP skills as `<server>:<skill>` (a claude.ai connector shows as
`claude_ai_<Name>:<skill>`), and only when a task needs one. Its documented limits are 100
skills per server, 100 KB per file and 400 KB per skill. sanzaru's largest file is about 14 KB.
Skills over MCP do not load in cloud sessions, `claude -p`, the Agent SDK, or for Desktop
connectors, so `SERVER_INSTRUCTIONS` stays enough on its own.

## Sync vs. async image generation

| Tool | API | Blocking inline? | As a task |
|------|-----|-------------------|-----------|
| `create_image` | Responses API (`background=True`) | No: returns a `response_id` at once | Provider-backed |
| `generate_image` | Images API | Yes: the call returns the finished image | Background |
| `edit_image` | Images API | Yes | Background |

The Images API has no job mode, so without Tasks the non-blocking path is `create_image` →
`wait_for`.
