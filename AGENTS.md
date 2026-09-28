# Repository Guidelines

This document helps contributors work effectively in this repository.

## Project Structure & Module Organization
- `src/sanzaru/` — MCP server (entrypoint `server.py`, CLI `sanzaru`).
- `videos/`, `images/` — Downloaded videos and reference images (git-ignored).
- `README.md` — Setup and tool usage; `setup.sh` — interactive setup; `.mcp.json` — sample MCP config.

## Build, Test, and Development Commands
- Install deps: `uv sync` (use `uv venv` first if needed).
- Run server: `uv run sanzaru` (loads `.env`).
- Quick start: `./setup.sh` (prompts for `OPENAI_API_KEY`, creates folders, installs deps). Add `HF_KEY` (Higgsfield API key, `key_id:key_secret`) for video.
- Codex integration (from repo root):
  - `codex mcp add sanzaru -- uv run --directory "$(pwd)" sanzaru`
- Smoke test (via MCP tools): `create_video(..., dry_run=true)` to price a job for free, then create a video/image and `wait_for([id], download=true)` (see README for the tool list).

## Video Toolkit & Prompting (Higgsfield; Sora was removed by OpenAI on 2026-09-24)
- Tools: `create_video`, `edit_video`, `extend_video`, `get_video_status`, `download_video`, `cancel_video`, `list_local_videos`, `inspect_video_frame`; images: `create_image`, `generate_image`, `edit_image`, `get_image_status`, `download_image`; refs: `list_reference_images`, `prepare_reference_image`; waiting: `wait_for`.
- Flows: Video only (create → `wait_for(download=true)`) or Image → Video (generate image → `prepare_reference_image(aspect_ratio=...)` → `create_video(reference_image=...)`). Every video job is priced first; pass `max_cost_usd`.
- Prompting: see `docs/video-prompting-guide.md`. Think “action + subject + scene + style + camera + lighting + mood”.
- Example: `create_video(prompt="wide tracking shot of a neon-lit rainy alley, cinematic, 35mm", aspect_ratio="16:9", duration=8, max_cost_usd=5)`

## Coding Style & Naming Conventions
- Python 3.10+, typed (prefer `TypedDict`, explicit return types).
- `snake_case` for functions/vars, `PascalCase` for classes, constants UPPER_SNAKE.
- Lint/format with `ruff` (line length 120, py310). Run: `uv run ruff check .` and `uv run ruff format .`.

## Testing Guidelines
- Tests: `uv run pytest` (unit + integration with mocked clients); for live checks, create → `wait_for` → download for videos/images.
- Verify files land in `videos/` and `images/`; include exact steps in PRs.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope first (e.g., `feat: add create_image tool`).
- Group related changes; avoid drive-by refactors.
- PRs: include purpose, screenshots/paths of created files, and testing steps (commands + observed output).
- Link issues when applicable; note any follow-ups.

## Security & Configuration Tips
- Required env: `OPENAI_API_KEY` and a media path (`SANZARU_MEDIA_PATH`, or `VIDEO_PATH`/`IMAGE_PATH`/`AUDIO_PATH`); `HF_KEY` enables video generation (billed separately from a Higgsfield app subscription).
- Folders must exist before starting the server; use `./setup.sh` for interactive configuration.
- Do not commit secrets or downloaded assets; `videos/` and `images/` are git-ignored.
- The server is stateless (stdio or HTTP, MCP SDK 2.x); job state lives with the providers — wait with `wait_for`, not long-lived state.
- Paths are validated lazily at runtime, supporting both `uv run` and `mcp run` invocations.
