# sora-mcp

A **stateless**, lightweight **FastMCP** server that wraps the **OpenAI Sora Video API** via the OpenAI Python SDK.

## Features
- Create Sora jobs (`sora-2` / `sora-2-pro`), optional image reference, optional remix
- Get status, wait until completion (polling), download assets (`video|thumbnail|spritesheet`)
- List and delete videos
- Stateless by default; no DB, no in-memory job tracking

> **Note:** Content guardrails are enforced by OpenAI. This server does not run local moderation.

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` in env

## Install with `uv`
```bash
uv venv
uv sync
```

## Run
```bash
export OPENAI_API_KEY=sk-...
uv run sora-mcp
```
This runs an MCP server over stdio that exposes these tools:
- `sora_create_video(prompt, model="sora-2", seconds?, size?, input_reference_path?)`
  - Note: `seconds` must be a string: `"4"`, `"8"`, or `"12"` (not an integer)
  - Note: `size` must be one of: `"720x1280"`, `"1280x720"`, `"1024x1792"`, `"1792x1024"`
  - Note: `model` must be one of: `"sora-2"`, `"sora-2-pro"`
- `sora_get_status(video_id)` - Returns Video object
- `sora_wait(video_id, poll_every=5, timeout=600)` - Returns Video object
- `sora_download(video_id, variant="video", path?)` - Downloads to disk
- `sora_list(limit=20, after?, order="desc")` - Returns paginated list
- `sora_delete(video_id)` - Deletes video
- `sora_remix(previous_video_id, prompt)` - Creates a remix

## Download variants
- `variant="video"` → `mp4`
- `variant="thumbnail"` → `webp`
- `variant="spritesheet"` → `jpg`

## Notes
- If providing an `input_reference_path`, ensure the image **matches** the requested `size` (e.g., `1280x720`).
- Download URLs are time-limited—copy outputs to your own storage promptly if needed.

## License
MIT
