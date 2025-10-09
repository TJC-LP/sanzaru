# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A stateless FastMCP server wrapping OpenAI's Sora Video API and Responses API (image generation). Runs over stdio and exposes MCP tools for async video/image generation with polling-based workflows.

**Key Architecture Principles:**
- **Stateless**: No database, no in-memory job tracking. All state lives in OpenAI's cloud.
- **Async polling pattern**: Create → Poll → Download workflow for both videos and images
- **Security sandbox**: Reference images restricted to `SORA_REFERENCE_PATH` with path traversal protection
- **Type-safe**: Extensive use of TypedDict and Literal types from OpenAI SDK

## Development Commands

```bash
# Install dependencies
uv sync

# Run the MCP server (stdio mode)
uv run sora-mcp-server

# Lint and format code
ruff check .
ruff format .

# Test the server locally (requires Claude Code or MCP client)
claude  # in this directory with .mcp.json configured
```

## Core Architecture

### Modular Server Design
The server is organized into focused modules for maintainability and code reuse:

```
src/sora_mcp_server/
├── server.py           # FastMCP initialization & tool registration (~320 lines)
├── types.py            # TypedDict definitions (~70 lines)
├── config.py           # OpenAI client + path configuration (~110 lines)
├── security.py         # File security utilities (~90 lines)
├── utils.py            # Shared helpers (~30 lines)
└── tools/              # Tool implementations
    ├── __init__.py
    ├── video.py        # 6 video tools (~200 lines)
    ├── reference.py    # 2 reference image tools (~230 lines)
    └── image.py        # 3 image generation tools (~180 lines)
```

**server.py** registers all tools with FastMCP decorators and delegates to tool implementations
**types.py** defines all return types (DownloadResult, VideoSummary, etc.)
**config.py** provides `get_client()` and `get_path()` with validation
**security.py** provides reusable functions: `validate_safe_path()`, `check_not_symlink()`, `safe_open_file()`
**utils.py** provides helpers: `suffix_for_variant()`, `generate_filename()`
**tools/*.py** contain the actual tool implementations as plain async functions

### Runtime Path Configuration
Paths are validated lazily via the `get_path()` function when tools are called:
- `get_path("video")`: Returns validated SORA_VIDEO_PATH for video downloads
- `get_path("reference")`: Returns validated SORA_REFERENCE_PATH for reference images

Both environment variables are required (no defaults). Paths are cached with `@lru_cache` for performance.

### Two API Integration Patterns

**1. Sora Video API (client.videos.*)**
- Async jobs with polling: `create()` → `retrieve()` → `download_content()`
- Status progression: `queued` → `in_progress` → `completed` or `failed`
- Progress tracking: 0-100 integer
- Uses OpenAI SDK types: `Video`, `VideoModel`, `VideoSize`, `VideoSeconds`
- Download supports optional custom filenames with path traversal protection

**2. Responses API (client.responses.*)**
- Background image generation: `create(background=True)` with `tools=[{"type": "image_generation"}]`
- Iterative refinement via `previous_response_id` parameter
- Returns `Response` object with `output` array containing `ImageGenerationCall` items
- Base64-encoded image in `ImageGenerationCall.result`

### Security Model
All file operations use centralized security utilities from `security.py`:

**`validate_safe_path(base_path, filename, allow_create=False)`**
- Prevents path traversal attacks (e.g., `../../etc/passwd`)
- Ensures resolved path stays within base_path
- Optionally validates file existence

**`check_not_symlink(path, error_context)`**
- Prevents symlink exploitation
- Raises ValueError if path is a symbolic link

**`safe_open_file(path, mode, error_context, check_symlink=True)`**
- Context manager for safe file I/O
- Standardized error handling (FileNotFoundError, PermissionError, OSError)
- Optional symlink checking

Example usage:
```python
from security import validate_safe_path, safe_open_file
from config import get_path

base_path = get_path("reference")
file_path = validate_safe_path(base_path, user_filename)

with safe_open_file(file_path, "rb", "reference image") as f:
    data = f.read()
```

Additional security:
- Symlinks rejected in environment variable paths (`get_path()` validation)
- Empty/whitespace-only env vars rejected
- User filenames validated against allowed extensions where applicable

## Prompting Sora with Reference Images

**CRITICAL**: When using `input_reference_filename`, keep prompts simple and focused on motion/action ONLY.

❌ **Bad**: Re-describing what's already in the image
```python
sora_create_video(
    prompt="A pilot in orange suit sitting in cockpit with instruments glowing...",
    input_reference_filename="pilot.png"
)
```

✅ **Good**: Describing only the action/transformation
```python
sora_create_video(
    prompt="The pilot glances up, takes a breath, then returns focus to the instruments.",
    input_reference_filename="pilot.png"
)
```

The reference image already contains: character, setting, framing, style, lighting.
The prompt should only describe: what happens next, motion, camera movement.

See `docs/sora-prompting-guide.md` and `docs/sora2_prompting_guide.ipynb` for complete prompting guidelines.

## Typical Workflows

### Generate Reference Image → Animate with Sora
```python
# 1. Generate reference image
resp = image_create(prompt="futuristic pilot in mech cockpit", size="1536x1024")
image_get_status(resp.id)  # poll until completed
image_download(resp.id, filename="pilot.png")

# 2. Resize for Sora if needed
sora_prepare_reference("pilot.png", target_size="1280x720", resize_mode="crop")

# 3. Create video with simple motion prompt
sora_create_video(
    prompt="The pilot looks up and smiles.",
    input_reference_filename="pilot_1280x720.png",
    size="1280x720",
    seconds="8"
)
```

### Iterative Image Refinement
```python
# Generate initial concept
resp1 = image_create(prompt="a cyberpunk character")

# Refine with previous_response_id
resp2 = image_create(
    prompt="add more neon details and a cityscape background",
    previous_response_id=resp1.id
)

# Continue refining
resp3 = image_create(
    prompt="change camera angle to show profile",
    previous_response_id=resp2.id
)
```

## Image Resize Modes

Three modes available in `sora_prepare_reference`:
- **crop**: Preserve aspect ratio, scale to cover target, center crop excess (no distortion, may lose edges)
- **pad**: Preserve aspect ratio, scale to fit, add black letterbox bars (no distortion, full image preserved)
- **rescale**: Stretch/squash to exact dimensions (may distort, no cropping/padding)

## Environment Configuration

Required environment variables (loaded via python-dotenv):
```bash
OPENAI_API_KEY="sk-..."
SORA_VIDEO_PATH="/absolute/path/to/videos"
SORA_REFERENCE_PATH="/absolute/path/to/references"
```

Use `./setup.sh` for interactive setup, or manually copy `.env.example` to `.env`.

## Model Selection Guidelines

**sora-2**: Faster, cheaper, good for iteration and testing
**sora-2-pro**: Slower, higher quality, for final production (supports larger resolutions)

**Supported video sizes:**
- Both models: `720x1280`, `1280x720`
- Pro only: `1024x1792`, `1792x1024`

**Image generation:**
- Use GPT-5 for best results
- Supported sizes: `1024x1024`, `1024x1536`, `1536x1024`, `auto`

## Type Safety Notes

- `VideoSeconds` must be string literal: `"4"`, `"8"`, or `"12"` (NOT integers)
- `VideoSize` and image sizes are string Literals enforced by type system
- Use `omit` from `openai._types` when converting `None` to SDK parameters
- All async functions use `AsyncOpenAI` client

## Code Style

- Line length: 120 characters
- Format with ruff (double quotes, space indentation)
- Tool descriptions support E501 ignore for readability
- Comprehensive docstrings on all MCP tools
- Security-first: path traversal protection on all file operations