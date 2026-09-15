# sanzaru

<div align="center">
  <img src="https://raw.githubusercontent.com/TJC-LP/sanzaru/main/assets/logo.png" alt="sanzaru logo" width="400">

  [![PyPI version](https://img.shields.io/pypi/v/sanzaru)](https://pypi.org/project/sanzaru/)
  [![Python versions](https://img.shields.io/pypi/pyversions/sanzaru)](https://pypi.org/project/sanzaru/)
  [![License](https://img.shields.io/pypi/l/sanzaru)](https://github.com/TJC-LP/sanzaru/blob/main/LICENSE)
  [![CI](https://github.com/TJC-LP/sanzaru/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/TJC-LP/sanzaru/actions/workflows/ci-cd.yml)
  [![PyPI downloads](https://img.shields.io/pypi/dm/sanzaru)](https://pypi.org/project/sanzaru/)
</div>

A **stateless**, lightweight **MCP** server **and agent CLI** that wraps **OpenAI's Sora Video API, Whisper, GPT-4o Audio, and TTS APIs** via the OpenAI Python SDK.

## Features

### Video Generation (Sora)
- Create videos with `sora-2` or `sora-2-pro` models
- Use reference images to guide generation
- Remix and refine existing videos
- Download variants (video, thumbnail, spritesheet)

### Image Generation
- Generate images with gpt-image-2.5 (sunburst/flare, recommended), gpt-image-2, gpt-image-1.5, or GPT-5
- Edit and compose images with up to 16 inputs
- Iterative refinement via Responses API
- Automatic resizing for Sora compatibility

### Audio Processing
- **Transcription**: Whisper and GPT-4o models
- **Audio Chat**: Interactive analysis with GPT-4o
- **Text-to-Speech**: Multi-voice TTS generation
- **Processing**: Format conversion, compression, file management

### Podcast Generation
- Multi-voice podcasts with up to 4 speakers and 10 TTS voices
- Parallel segment generation with configurable pacing
- MP3/WAV output with loudness normalization
- ElevenLabs `dialogue` render mode: consecutive turns go out together so the model paces them
- `--verify` transcribes the rendered audio and re-renders segments the TTS silently dropped

### Simulated Podcasts
- **The conversation is generated, not read** — N `gpt-realtime` agents actually talk to each other
- Each host hears the others' audio, so they respond to delivery and timing, not to a transcript
- Pre-production plans acts that record **in parallel**: a 30-minute episode in ~1 minute
- Checkpointed per act and resumable; cost ceiling, dry-run projection, per-host stems
- QC transcribes the rendered audio and judges it against the plan
- See [`docs/audio/simulated-podcasts.md`](docs/audio/simulated-podcasts.md)

### Agent CLI
- Every capability as a shell command: `sanzaru video create`, `sanzaru image generate`, ...
- One-shot async workflows: `create ... -o out.mp4` submits, polls, downloads in one command
- JSON envelopes on stdout, progress on stderr, deterministic exit codes, resumable waits
- Concurrent fan-out (multi-prompt image batches, multi-job `wait`) and arbitrary `-o` output paths
- See [`docs/cli.md`](docs/cli.md) — bare `sanzaru` still runs the MCP server (nothing breaks)

> **Note:** Content guardrails are enforced by OpenAI. This server does not run local moderation.

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` environment variable

**Media storage** (choose one):
```bash
# Recommended: unified path (auto-creates videos/, images/, audio/ subdirs)
SANZARU_MEDIA_PATH="/path/to/media"

# Or individual paths (legacy, still supported)
VIDEO_PATH="/path/to/videos"
IMAGE_PATH="/path/to/images"
AUDIO_PATH="/path/to/audio"
```

Features are auto-detected based on configured paths. Set only what you need.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TJC-LP/sanzaru.git
   cd sanzaru
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   The script will:
   - Prompt for your OpenAI API key
   - Create directories and `.env` configuration
   - Install dependencies with `uv sync --all-extras --dev`

3. **Start using:**
   ```bash
   claude
   ```

That's it! Claude Code will automatically connect and you can start generating videos, images, and processing audio.

### Or skip MCP entirely — the agent CLI

```bash
uv tool install sanzaru && export OPENAI_API_KEY=sk-...

# One command: submit Sora job → poll → download → print the file path
sanzaru video create "a tabby cat stretches on a windowsill" --seconds 4 -o ./cat.mp4 | jq -r .result.file.path

# Synchronous image generation (gpt-image-2.5), batch fan-out, JSONL output
sanzaru image generate "app icon" "hero banner" --quality high -o ./art/

sanzaru capabilities   # machine-readable: what's enabled here
```

JSON envelopes on stdout, progress on stderr, exit 4 = still-running-and-resumable. Full
reference: [`docs/cli.md`](docs/cli.md).

## Installation

### Claude Code Plugin (Recommended)

Install as a plugin — auto-configures the MCP server + includes prompting guidance:

```bash
/plugin marketplace add TJC-LP/sanzaru
```

Requires `OPENAI_API_KEY` and `SANZARU_MEDIA_PATH` environment variables to be set.

### Quick Install
```bash
# All features
uv add "sanzaru[all]"

# Specific features
uv add "sanzaru[audio]"       # With audio support
uv add "sanzaru[elevenlabs]"  # ElevenLabs as a second TTS provider
uv add sanzaru                # Base (video + image only)
```

<details>
<summary><strong>Alternative Installation Methods</strong></summary>

### From Source
```bash
git clone https://github.com/TJC-LP/sanzaru.git
cd sanzaru
uv sync --all-extras
```

### Claude Desktop
Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sanzaru": {
      "command": "uvx",
      "args": ["sanzaru[all]"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here",
        "SANZARU_MEDIA_PATH": "/absolute/path/to/media"
      }
    }
  }
}
```

Or from source:
```json
{
  "mcpServers": {
    "sanzaru": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/sanzaru", "sanzaru"]
    }
  }
}
```

### Codex MCP
```bash
# Using uvx (from PyPI)
codex mcp add sanzaru \
  --env OPENAI_API_KEY="sk-..." \
  --env SANZARU_MEDIA_PATH="$HOME/sanzaru-media" \
  -- uvx "sanzaru[all]"
```

### Manual Setup
```bash
uv venv
uv sync

# Set required environment variables
export OPENAI_API_KEY=sk-...
export SANZARU_MEDIA_PATH=~/sanzaru-media

# Run server (stdio for MCP clients)
uv run sanzaru

# Or HTTP mode (for remote access)
uv run sanzaru --transport http --port 8000
```

</details>

## Available Tools

| Category | Tools | Description |
|----------|-------|-------------|
| **Video** | `create_video`, `get_video_status`, `download_video`, `list_videos`, `list_local_videos`, `delete_video`, `remix_video` | Generate and manage Sora videos with optional reference images |
| **Jobs** | `wait_for` | Block server-side on any mix of `video_*`/`resp_*` ids until they finish (progress on every poll, optional download); replaces model-driven polling |
| **Image** | `generate_image`, `edit_image`, `create_image`, `get_image_status`, `download_image` | Generate with gpt-image-2.5 (default, sync) or GPT-5 (polling) |
| **Reference** | `list_reference_images`, `prepare_reference_image` | Manage and resize images for Sora compatibility |
| **Audio** | `transcribe_audio`, `chat_with_audio`, `create_audio`, `convert_audio`, `compress_audio`, `list_audio_files`, `get_latest_audio`, `transcribe_with_enhancement` | Transcription, analysis, TTS (OpenAI or ElevenLabs), and file management |
| **Podcast** | `generate_podcast` | Multi-voice podcast generation with parallel TTS and audio stitching; speakers may mix TTS providers |
| **Simulated Podcast** | `simulate_podcast` | Realtime agents converse from a rundown — parallel acts, checkpointing, cost ceiling, QC |
| **Media** | `view_media` | Interactive media player via MCP App protocol |

> **Full API documentation**: See [docs/api-reference.md](docs/api-reference.md)

## Basic Workflows

### Generate a Video
```python
# Create video from text
video = create_video(
    prompt="A serene mountain landscape at sunrise",
    model="sora-2",
    seconds="8",
    size="1280x720"
)

# Poll for completion
status = get_video_status(video.id)

# Download when ready
download_video(video.id, filename="mountain_sunrise.mp4")
```

### Generate with Reference Image
```python
# 1. Generate reference image (gpt-image-2.5-flare, synchronous)
generate_image(
    prompt="futuristic pilot in mech cockpit",
    size="1536x1024",
    filename="pilot.png"
)

# 2. Prepare for video (resize to Sora dimensions)
prepare_reference_image("pilot.png", "1280x720", resize_mode="crop")

# 3. Animate
video = create_video(
    prompt="The pilot looks up and smiles",
    size="1280x720",
    input_reference_filename="pilot_1280x720.png"
)
```

### Audio Transcription
```python
# List available audio files
files = list_audio_files(format="mp3")

# Transcribe — long files are auto-windowed, so nothing truncates silently
result = transcribe_audio("interview.mp3")

# Or analyze with GPT-4o
analysis = chat_with_audio(
    "meeting.mp3",
    user_prompt="Summarize key decisions and action items"
)
```

### Generate a Podcast
```python
generate_podcast(script={
    "title": "AI Weekly",
    "speakers": [
        {"id": "host", "name": "Alex", "voice": "nova"},
        {"id": "guest", "name": "Sam", "voice": "echo"}
    ],
    "segments": [
        {"speaker": "host", "text": "Welcome to AI Weekly!"},
        {"speaker": "guest", "text": "Thanks for having me."}
    ]
})
```

### Simulate a Podcast (no script — the agents talk)
```bash
# 1. Plan it. Cheap, and the JSON is yours to edit.
sanzaru podcast rundown "why TTS providers drop sentence tails" \
  --acts 3 -m 6 --host "Avery" --host "Rory:cedar:You chased the bug." \
  -o rundown.json

# 2. See what it would cost. Records nothing.
sanzaru podcast simulate @rundown.json --dry-run

# 3. Record it. Acts run in parallel and each is checkpointed as it lands.
sanzaru podcast simulate @rundown.json --model gpt-realtime-2.1-mini \
  --max-cost 2.00 --stems -o ep1.mp3

# Interrupted? The run id is on stderr from the start.
sanzaru podcast simulate --resume 6f1a9c02
```

## Documentation

- **[API Reference](docs/api-reference.md)** - Complete tool documentation with parameters and examples
- **[Reference Images Guide](docs/reference-images.md)** - Working with reference images and resizing
- **[Image Generation Guide](docs/image-generation.md)** - Generating and editing reference images
- **[Sora Prompting Guide](docs/sora2-prompting-guide.md)** - Crafting effective video prompts
- **[Audio Features](docs/audio/README.md)** - Audio transcription, chat, and TTS
- **[Simulated Podcasts](docs/audio/simulated-podcasts.md)** - Realtime agents in conversation: producer model, act chunking, cost, QC
- **[Performance & Architecture](docs/async-optimizations.md)** - Technical details and benchmarks

## Transport Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **stdio** (default) | `uv run sanzaru` | Claude Desktop, Claude Code, local MCP clients |
| **HTTP** | `uv run sanzaru --transport http` | Remote access, Databricks Apps, web clients |

### Authenticating HTTP mode

HTTP mode exposes the full toolset — paid generation, `delete_video`, and every
stored media file — to whoever can reach the port. Set a token and send it as
`Authorization: Bearer <token>` on both `/mcp` and `/media`:

```bash
export SANZARU_HTTP_TOKEN="$(openssl rand -hex 32)"
uv run sanzaru --transport http --host 0.0.0.0
```

Binding to anything other than loopback **requires** the token: sanzaru refuses
to start otherwise (exit 3). Set `SANZARU_ALLOW_UNAUTHENTICATED_HTTP=1` only when
something in front of it already authenticates every request — and name that
proxy's hostnames in `SANZARU_ALLOWED_HOSTS` (comma-separated, `host` or `host:*`),
which the hatch requires: it keeps the SDK's DNS-rebinding check on, Origin
included, so the unauthenticated path is never also the least-protected one. With
a token and no allowlist the Host/Origin check is switched off and the token is the
control. On a loopback bind the token is optional but still recommended.

**Databricks Apps** authenticates in front of the app, so run it with
`SANZARU_ALLOW_UNAUTHENTICATED_HTTP=1` and `SANZARU_ALLOWED_HOSTS` set to the Host
value the platform proxy forwards (include whatever its health probe sends). Include
the loopback names too if the probe uses them, e.g.
`SANZARU_ALLOWED_HOSTS=myapp.example.com,localhost:*`.

**Upgrading from 0.10.x:** `sanzaru --transport http --host 0.0.0.0` used to start
with no credential. It now refuses unless `SANZARU_HTTP_TOKEN` or the hatch above
is set — deliberate, and worth a line in your deployment notes.

`/media/{type}/{name}` requires the same `Authorization` header, and every response
is `Content-Disposition: attachment`, so it is for programmatic clients; a browser
media element cannot send the header and should use the viewer's `_get_media_data`
path (the bundled MCP App already does).

Embedding the server in your own ASGI stack? Use `sanzaru.server.build_http_app()` —
the same authenticated app the CLI serves — never the bare `mcp.streamable_http_app()`,
which carries none of the middleware. See CLAUDE.md, *Transport Modes*.

### What a `.env` file can (and cannot) configure

The `sanzaru` command — every subcommand, `sanzaru serve` included — autoloads a
`.env` for local development. `python-dotenv` is a runtime dependency, so this
works in any install, not only under `uv sync`. Two deliberate limits apply,
because a file found on disk must not be able to redirect credentials, relax
transport security, or change what a run is allowed to cost:

- **Only `./.env` is read** — the directory you run sanzaru in. There is no
  search of parent directories, so a `.env` at your project root is not found
  when you run from a subdirectory.
- **Only sanzaru's documented configuration keys load** (API keys, media paths,
  storage backend credentials, tuning knobs), matched exactly and
  case-sensitively. Everything else in the file is ignored with a warning naming
  the keys. Notably ignored on purpose: `DATABRICKS_HOST` and any
  `*_BASE_URL`/proxy variable (they decide *where* credentials are sent); every
  HTTP-security variable — `SANZARU_HTTP_TOKEN`, `SANZARU_ALLOW_UNAUTHENTICATED_HTTP`,
  `SANZARU_ALLOWED_HOSTS`, `SANZARU_ALLOWED_ORIGINS`, `SANZARU_IDENTITY_HEADER`,
  `SANZARU_REQUIRE_USER_CONTEXT` (a planted file must not weaken or satisfy
  transport auth, nor pick whose identity is trusted); `SANZARU_RUN_SECRET` (the
  signing key); `SANZARU_REALTIME_PRICE_*` (the price table is what `--max-cost` is
  enforced against — a planted `0,0,0,0,0,0` would make every turn free); and
  `DATABRICKS_VIDEO_DIR`/`_IMAGE_DIR`/`_AUDIO_DIR` (joined into the volume path
  unsanitized, so `..` in one walks into another tenant's files).

Anything the allowlist skips still works exported in the real environment, or
injected explicitly with `npx dotenv-cli -- <command>` — both are deliberate
operator actions rather than a file discovered on disk.

## Storage Backends

| Backend | Config | Use Case |
|---------|--------|----------|
| **Local** (default) | `SANZARU_MEDIA_PATH=/path/to/media` | Development, local deployments |
| **Databricks** | `STORAGE_BACKEND=databricks` | Databricks Apps with Unity Catalog Volumes |

The Databricks backend supports per-user storage isolation via the `user_context` module, enabling multi-tenant deployments where each user's media is stored under their own volume prefix (`<local-part>-<hash>`, injective over email addresses; the prefix format changed after 0.10.0 — see CLAUDE.md for the migration note). In HTTP mode the identity comes from a proxy-injected header, and trusting one is **opt-in**: set `SANZARU_IDENTITY_HEADER` (e.g. `x-forwarded-email` on Databricks Apps) only when a proxy in front of sanzaru both injects that header and **strips** any client-supplied copy. When it is unset, no header is trusted and every request resolves to the shared volume root. A request carrying the header twice (an appending proxy forwards the client's copy first) or malformed is refused with 400 rather than binding either copy. Set `SANZARU_REQUIRE_USER_CONTEXT=1` on a shared deployment so a request with no identity is refused (403 on `/media`) instead of silently served out of the shared root.

Multi-tenant deployments should also set `SANZARU_RUN_SECRET`, which signs simulated-podcast run manifests and act checkpoints so `--resume` refuses bookkeeping this installation did not write.

See [CLAUDE.md](CLAUDE.md) for full configuration details.

## Performance

Fully asynchronous architecture with proven scalability:
- ✅ 32+ concurrent operations verified
- ✅ 8-10x speedup for parallel tasks
- ✅ Non-blocking I/O with `aiofiles` + `anyio`
- ✅ Python 3.14 free-threading ready

See [docs/async-optimizations.md](docs/async-optimizations.md) for technical details.

## License

[MIT](LICENSE)
