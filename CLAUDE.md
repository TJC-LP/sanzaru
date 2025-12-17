# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A stateless FastMCP server wrapping OpenAI's Sora Video API and Responses API (image generation). Supports both stdio (for MCP clients) and HTTP streaming (for web clients) transports. Exposes MCP tools for async video/image generation with polling-based workflows.

**Key Architecture Principles:**
- **Stateless**: No database, no in-memory job tracking. All state lives in OpenAI's cloud.
- **Async polling pattern**: Create → Poll → Download workflow for both videos and images
- **Security sandbox**: Reference images restricted to `IMAGE_PATH` with path traversal protection
- **Type-safe**: Extensive use of TypedDict and Literal types from OpenAI SDK
- **Dual transport**: stdio (default) for Claude Desktop, HTTP for web clients and remote access

## Development Commands

```bash
# Install dependencies
uv sync

# Run the MCP server (stdio mode - default)
uv run sanzaru

# Run the MCP server (HTTP mode - stateless)
uv run sanzaru --transport http
uv run sanzaru --transport http --port 3000
uv run sanzaru --transport http --host 0.0.0.0 --port 8080

# Lint and format code
ruff check .
ruff format .

# Test the server locally (requires Claude Code or MCP client)
claude  # in this directory with .mcp.json configured
```

## Core Architecture

**Async Architecture:**
- Fully non-blocking I/O with `aiofiles` and `anyio`
- CPU-bound operations (PIL, base64) run in thread pools
- Streaming downloads with async iteration
- See [`docs/async-optimizations.md`](docs/async-optimizations.md) for details

### Modular Server Design
The server is organized into focused modules for maintainability and code reuse:

```
src/sanzaru/
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
- `get_path("video")`: Returns validated VIDEO_PATH for video downloads
- `get_path("reference")`: Returns validated IMAGE_PATH for reference images

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
create_video(
    prompt="A pilot in orange suit sitting in cockpit with instruments glowing...",
    input_reference_filename="pilot.png"
)
```

✅ **Good**: Describing only the action/transformation
```python
create_video(
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
resp = create_image(prompt="futuristic pilot in mech cockpit", size="1536x1024")
get_image_status(resp.id)  # poll until completed
download_image(resp.id, filename="pilot.png")

# 2. Resize for Sora if needed
prepare_reference_image("pilot.png", target_size="1280x720", resize_mode="crop")

# 3. Create video with simple motion prompt
create_video(
    prompt="The pilot looks up and smiles.",
    input_reference_filename="pilot_1280x720.png",
    size="1280x720",
    seconds="8"
)
```

### Iterative Image Refinement
```python
# Generate initial concept
resp1 = create_image(prompt="a cyberpunk character")

# Refine with previous_response_id
resp2 = create_image(
    prompt="add more neon details and a cityscape background",
    previous_response_id=resp1.id
)

# Continue refining
resp3 = create_image(
    prompt="change camera angle to show profile",
    previous_response_id=resp2.id
)
```

## Image Resize Modes

Three modes available in `prepare_reference_image`:
- **crop**: Preserve aspect ratio, scale to cover target, center crop excess (no distortion, may lose edges)
- **pad**: Preserve aspect ratio, scale to fit, add black letterbox bars (no distortion, full image preserved)
- **rescale**: Stretch/squash to exact dimensions (may distort, no cropping/padding)

## Environment Configuration

Required environment variables:
```bash
OPENAI_API_KEY="sk-..."
VIDEO_PATH="/absolute/path/to/videos"
IMAGE_PATH="/absolute/path/to/references"
AUDIO_PATH="/absolute/path/to/audio"
```

Optional environment variables:
```bash
LOG_LEVEL="INFO"  # Optional: DEBUG, INFO, WARNING, ERROR (defaults to INFO)
```

**For MCP servers (Claude Desktop):**
Set environment variables explicitly in `.mcp.json` using template variables:
```json
{
  "mcpServers": {
    "sanzaru": {
      "command": "uv",
      "args": ["run", "sanzaru"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "VIDEO_PATH": "${VIDEO_PATH}",
        "IMAGE_PATH": "${IMAGE_PATH}",
        "AUDIO_PATH": "${AUDIO_PATH}"
      }
    }
  }
}
```

**Note:** `LOG_LEVEL` can be optionally added to the `env` object if you want to override the default (INFO).

**For local development with .env files:**
1. Run `./setup.sh` for interactive setup, or manually copy `.env.example` to `.env`
2. Install dotenv: `uv add --dev python-dotenv`
3. Run Claude with dotenv-cli to inject env vars: `npx dotenv-cli -- claude` (or `bunx dotenv-cli -- claude`)

This approach makes environment configuration explicit and avoids confusion from implicit `.env` loading.

## Transport Modes

Sanzaru supports two transport modes for different deployment scenarios:

### stdio (Default)
Standard I/O transport for MCP clients like Claude Desktop:
```bash
uv run sanzaru
```

**Use cases:**
- Claude Desktop integration
- Local MCP client connections
- Development and testing

**Configuration:** Set environment variables in `.mcp.json` or via dotenv-cli

### http (Stateless HTTP Streaming)
HTTP streaming transport for web clients and remote access:
```bash
# Local HTTP server
uv run sanzaru --transport http

# Custom host/port
uv run sanzaru --transport http --host 0.0.0.0 --port 3000
```

**Use cases:**
- Web-based MCP clients
- Remote server deployments
- Multi-user access (stateless, no session management)
- Browser-based integrations

**Endpoints:** MCP tools available at `http://{host}:{port}/mcp`

**Key features:**
- **Stateless by design:** No session IDs required (all state lives in OpenAI's cloud)
- **SSE streaming:** Server-Sent Events for real-time communication
- **CORS support:** Can be configured via Starlette middleware (see Python MCP SDK docs)

**Production deployment:**
For advanced deployments with CORS, multiple servers, or custom middleware, mount the server in a Starlette app:
```python
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware
from sanzaru.server import mcp

app = Starlette(routes=[Mount("/mcp", mcp.streamable_http_app())])
app = CORSMiddleware(app, allow_origins=["*"], expose_headers=["Mcp-Session-Id"])
```

## Model Selection Guidelines

### Video Generation (Sora)
**sora-2**: Faster, cheaper, good for iteration and testing
**sora-2-pro**: Slower, higher quality, for final production (supports larger resolutions)

**Supported video sizes:**
- Both models: `720x1280`, `1280x720`
- Pro only: `1024x1792`, `1792x1024`

### Image Generation

**Two APIs available:**

| Tool | API | Best For |
|------|-----|----------|
| `generate_image` | Images API | New generation with gpt-image-1.5 (RECOMMENDED) |
| `edit_image` | Images API | Editing existing images with gpt-image-1.5 |
| `create_image` | Responses API | Iterative refinement with `previous_response_id` |

**Image generation models:**
- **gpt-image-1.5**: STATE-OF-THE-ART (RECOMMENDED) - best quality, better instruction following, improved text rendering, 20% cheaper
- **gpt-image-1**: High quality image generation
- **gpt-image-1-mini**: Fast, cost-effective generation
- **dall-e-3**: Legacy DALL-E 3
- **dall-e-2**: Legacy DALL-E 2

**Supported image sizes:** `1024x1024`, `1024x1536`, `1536x1024`, `auto`

**Example with generate_image (recommended):**
```python
# Direct Images API - synchronous, returns immediately
generate_image(
    prompt="a futuristic cityscape at sunset",
    model="gpt-image-1.5",
    size="1536x1024",
    quality="high"
)
```

**Example with create_image (for iterative refinement):**
```python
# Responses API - async polling, supports previous_response_id
resp = create_image(prompt="a cyberpunk character")
# ... poll and download ...
resp2 = create_image(
    prompt="add neon details",
    previous_response_id=resp.id
)
```

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

### Naming Conventions

**Function Names: Verb-First (Predicate-First)**
- Internal functions use `verb_noun` pattern (action comes first)
- Examples:
  - ✅ `create_video()` - verb first
  - ✅ `download_image()` - verb first
  - ✅ `list_reference_images()` - verb first
  - ✅ `get_video_status()` - verb first

**MCP Tool Names (Public API): Keep "sora" prefix for branding**
- Server wrapper functions can keep descriptive names for MCP tools
- Example: MCP tool `create_video` → calls internal `create_video()`

**Description Constants: Match tool names**
- `CREATE_VIDEO`, `DOWNLOAD_IMAGE`, `LIST_REFERENCE_IMAGES`
- ALL_CAPS with underscores
- Verb comes first, matches function structure

## Testing

**Test Structure:**
```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests for pure functions (46 tests)
│   ├── test_utils.py
│   ├── test_security.py
│   └── test_image_processing.py
└── integration/         # Integration tests with mocked clients (12 tests)
    ├── test_video_tools.py
    ├── test_image_tools.py
    └── test_reference_tools.py
```

**Style Notes:**
- **NO `__init__.py` files in test directories** - tests are not a package
- Use `@pytest.mark.unit` for unit tests (pure functions, no mocking)
- Use `@pytest.mark.integration` for integration tests (mocked OpenAI client)
- Use `pytest-mock` (mocker fixture) for all mocking
- Ignore SIM117 in tests (nested `with` intentional for pytest.raises)

**Running Tests:**
```bash
pytest                        # All tests
pytest tests/unit -m unit     # Unit tests only (fast)
pytest tests/integration      # Integration tests only
pytest --cov=src              # With coverage report
```

**Coverage Goals:**
- Pure functions: 100% (achieved)
- Tools (business logic): 80%+ (achieved: 82-88%)
- Overall: 65%+ (achieved)
- Always remember to read `.venv` files for type information about external libraries. NEVER use `typing.Any`.
