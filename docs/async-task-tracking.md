# Async Task Tracking — Status & Decisions

This note records where Sanzaru stands on long-running task handling and why, so the
question doesn't have to be re-investigated from scratch next time.

## TL;DR

- Sanzaru already implements a "call-now, fetch-later" pattern for long jobs
  (`create_*` → `get_*_status` → `download_*`). It works on **every** MCP client today.
- We are **not** adopting MCP's native Tasks protocol yet — it's experimental, was removed
  from the spec, and the experimental API is slated for removal in the `mcp` 2.0 SDK.
- Direct image generation (`generate_image` / `edit_image`) is **synchronous by design**
  because the OpenAI Images API has no background/job mode. The non-blocking path already
  exists as `create_image` (Responses API, `background=True`).

## MCP native Tasks (SEP-1686): deferred

The Model Context Protocol added a general-purpose **Tasks** capability (SEP-1686) in the
`2025-11-25` spec revision: a request can be augmented with a `task` field, the server
returns a task handle immediately, and the client polls `tasks/get` / `tasks/result`.

Why we're not using it (as of this writing):

- **Experimental and unstable.** Tasks were subsequently **removed from the MCP spec**, and
  the experimental Tasks API is **scheduled for removal in the `mcp` 2.0 Python SDK**. It is
  expected to return later as a separate MCP extension.
- **Not in the SDK we use.** Sanzaru runs on the **official `mcp` SDK** (`mcp.server.fastmcp`).
  The ergonomic `@tool(task=True)` decorator lives only in the *standalone* `fastmcp` package
  (v2.14+/v3), which would be a framework migration, not a drop-in.
- **No client benefit yet.** Native tasks only help when the consuming client also implements
  them. Our current polling pattern already gives the same "start now, fetch later" behavior
  with universal client support.

**Revisit trigger:** when the official `mcp` SDK ships a *stable* tasks/extension API **and**
the target clients (Claude Desktop / Claude Code / claude.ai) advertise support for it.

## Sync vs. async image generation

| Tool | API | Blocking? | Async job + poll? |
|------|-----|-----------|-------------------|
| `create_image` | Responses API (`background=True`) | No — returns a `response_id` immediately | Yes: `get_image_status` → `download_image` |
| `generate_image` | Images API (`images.generate`) | Yes — call returns the finished image | No (Images API has no job mode) |
| `edit_image` | Images API (`images.edit`) | Yes — call returns the finished image | No (Images API has no job mode) |

`generate_image` / `edit_image` are already `async def` (non-blocking *I/O* — they don't stall
the event loop), but the *workflow* is one-shot: a single API call blocks until the image is
rendered. The OpenAI **Images API has no background/job parameter**, so they cannot be turned
into a job-and-poll flow without switching to the Responses API. For non-blocking generation,
use `create_image`.

### Deferred enhancement options

If a blocking `generate_image` ever causes client timeouts on large/slow renders, two stable
options exist (neither depends on SEP-1686):

1. **Images API streaming** — `images.generate(stream=True, partial_images=1-3)` yields partial
   images as they render, giving incremental output instead of one long wait.
2. **MCP progress notifications** — inject `ctx: Context` into the tool and call
   `ctx.report_progress(...)`; clients that render progress bars get live updates during the
   call. This is part of MCP core (not Tasks) and is supported by the official SDK.

Both are deferred until there's a concrete need.
