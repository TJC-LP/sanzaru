# Repository Guidelines

This document helps contributors work effectively in this repository.

## Project Structure & Module Organization
- `src/sora_mcp_server/` — MCP server (entrypoint `server.py`, CLI `sora-mcp-server`).
- `sora-videos/`, `sora-references/` — Downloaded videos and reference images (git-ignored).
- `README.md` — Setup and tool usage; `setup.sh` — interactive setup; `.mcp.json` — sample MCP config.

## Build, Test, and Development Commands
- Install deps: `uv sync` (use `uv venv` first if needed).
- Run server: `uv run sora-mcp-server` (loads `.env`).
- Quick start: `./setup.sh` (prompts for `OPENAI_API_KEY`, creates folders, installs deps).
- Codex integration (from repo root):
  - `codex mcp add sora-mcp-server -- uv run --directory "$(pwd)" sora-mcp-server`
- Smoke test (via MCP tools): create a video/image, poll status, then download (see README for tool list).

## Sora Toolkit & Prompting
- Tools: `sora_create_video`, `sora_get_status`, `sora_download`, `sora_list`, `sora_delete`, `sora_remix`; images: `image_create`, `image_get_status`, `image_download`; refs: `sora_list_references`, `sora_prepare_reference`.
- Flows: Video only (create → poll → download) or Image → Video (generate image → download → use as `input_reference_filename`).
- Prompting: see `docs/sora2-prompting-guide.md`. Think “action + subject + scene + style + camera + lighting + mood”.
- Example: `sora_create_video(prompt="wide tracking shot of a neon-lit rainy alley, cinematic, 35mm", size="1280x720", seconds="8")`

## Coding Style & Naming Conventions
- Python 3.10+, typed (prefer `TypedDict`, explicit return types).
- `snake_case` for functions/vars, `PascalCase` for classes, constants UPPER_SNAKE.
- Lint/format with `ruff` (line length 120, py310). Run: `uv run ruff check .` and `uv run ruff format .`.

## Testing Guidelines
- No formal tests yet. Do manual smoke tests: create → status → download for videos/images.
- Verify files land in `sora-videos/` and `sora-references/`; include exact steps in PRs.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope first (e.g., `feat: add image_create tool`).
- Group related changes; avoid drive-by refactors.
- PRs: include purpose, screenshots/paths of created files, and testing steps (commands + observed output).
- Link issues when applicable; note any follow-ups.

## Security & Configuration Tips
- Required env: `OPENAI_API_KEY`, `SORA_VIDEO_PATH`, `SORA_REFERENCE_PATH` (no defaults; must be explicitly set).
- Folders must exist before starting the server; use `./setup.sh` for interactive configuration.
- Do not commit secrets or downloaded assets; `sora-videos/` and `sora-references/` are git-ignored.
- The server is stateless and communicates over stdio via FastMCP; rely on polling, not long-lived state.
- Paths are validated lazily at runtime, supporting both `uv run` and `mcp run` invocations.
