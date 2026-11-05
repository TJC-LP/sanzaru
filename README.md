# sanzaru

<div align="center">
  <img src="https://raw.githubusercontent.com/TJC-LP/sanzaru/main/assets/logo.png" alt="sanzaru logo" width="400">

  [![PyPI version](https://img.shields.io/pypi/v/sanzaru)](https://pypi.org/project/sanzaru/)
  [![Python versions](https://img.shields.io/pypi/pyversions/sanzaru)](https://pypi.org/project/sanzaru/)
  [![License](https://img.shields.io/pypi/l/sanzaru)](https://github.com/TJC-LP/sanzaru/blob/main/LICENSE)
  [![CI](https://github.com/TJC-LP/sanzaru/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/TJC-LP/sanzaru/actions/workflows/ci-cd.yml)
  [![PyPI downloads](https://img.shields.io/pypi/dm/sanzaru)](https://pypi.org/project/sanzaru/)
</div>

A **stateless**, lightweight **MCP** server **and agent CLI** that wraps **Higgsfield video generation** (Seedance 2.5, Kling 3.0) and **OpenAI's image, transcription, audio and TTS APIs**, plus ElevenLabs for podcasts.

> **0.14.0:** the base install is the agent CLI only. Anything that launches the MCP server (`.mcp.json`, Claude Desktop, `sanzaru serve`) must install `sanzaru[mcp]` or `sanzaru[all]`. `HIGGSFIELD_BASE_URL` can point video at a credential proxy.
>
> **0.13.0:** OpenAI removed the Sora Videos API and every Sora model on 2026-09-24. sanzaru now generates video through the [Higgsfield API](https://higgsfield.ai/higgsfield-api) — set `HF_KEY`. Old `video_…` ids and the `list_videos` / `delete_video` / `remix_video` tools are gone.

## Features

### Video Generation (Higgsfield)
- Text-to-video and image-to-video with **Seedance 2.5** (default; 4–30 s, native audio) or **Kling 3.0**
  (std / pro / 4K / turbo), or any Higgsfield catalog model by id
- **Edit** and **extend** existing clips (Seedance 2.5) — including a job you just rendered, by its `hf_…` id
- **Every job is priced before it is submitted**: the cost comes back with the job, `max_cost_usd`
  refuses anything over budget (nothing uploaded or charged), `dry_run` prices for free
- One `wait_for(download=true)` call waits server-side and saves the file into your media directory

### Image Generation
- Generate images with gpt-image-2.5 (sunburst/flare, recommended), gpt-image-2, gpt-image-1.5, or GPT-5
- Edit and compose images with up to 16 inputs
- Iterative refinement via Responses API
- Crop a reference image to a video aspect ratio (image-to-video framing follows the image)

### Audio Processing
- **Transcription**: Whisper and GPT-4o models
- **Audio Chat**: Interactive analysis with GPT-4o
- **Text-to-Speech**: Multi-voice TTS generation
- **Processing**: Format conversion, compression, file management

### Podcast Generation
- Multi-voice podcasts with up to 4 speakers and 10 TTS voices
- Parallel segment generation with configurable pacing
- MP3/WAV output with loudness normalization
- **The recommended way to make a podcast** — from a script, or from a topic (the model writes the script first): ElevenLabs `eleven_v4` (the ElevenLabs default) in `dialogue` render mode with `--verify`. OpenAI is the zero-setup fallback
- ElevenLabs `dialogue` render mode: consecutive turns go out together so the model paces them
- `--verify` transcribes the rendered audio and re-renders segments the TTS silently dropped

### Simulated Podcasts (experimental)
- Experimental, and the most expensive tool in sanzaru — for when you explicitly want an unscripted conversation. For a podcast, prefer the scripted ElevenLabs path above
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
- Python 3.11+ (tested through 3.15 and 3.15t)
- `OPENAI_API_KEY` for images, audio and podcasts
- `HF_KEY` for video — a Higgsfield **API** key (`key_id:key_secret`) from the API console. The API is
  prepaid and billed separately from a Higgsfield app/CLI subscription; the video tools register only when it is set

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

# One command: price → submit → wait → download → print the file path
sanzaru video create "a tabby cat stretches on a windowsill" --duration 4 --resolution 480p --max-cost 1 -o ./cat.mp4 | jq -r .result.file.path

# Synchronous image generation (gpt-image-2.5), batch fan-out, JSONL output
sanzaru image generate "app icon" "hero banner" --quality high -o ./art/

sanzaru capabilities   # machine-readable: what's enabled here
```

JSON envelopes on stdout, progress on stderr, exit 4 = still-running-and-resumable. Full
reference: [`docs/cli.md`](docs/cli.md).

## Installation

### Claude Code Plugin (Recommended)

Install as a plugin — auto-configures the MCP server and ships three skills: `sanzaru-mcp` (the tool surface, incl. `wait_for`), `sanzaru-cli` (the shell surface), `prompt-guidance` (how to write video/image prompts):

```bash
/plugin marketplace add TJC-LP/sanzaru
```

Requires `OPENAI_API_KEY` and `SANZARU_MEDIA_PATH` environment variables to be set (and `HF_KEY` for video).

### Quick Install
```bash
# All features
uv add "sanzaru[all]"

# Specific features
uv add "sanzaru[audio]"       # With audio support
uv add "sanzaru[elevenlabs]"  # ElevenLabs as a second TTS provider
uv add sanzaru                # Base: the agent CLI only (video via HF_KEY; no MCP dependencies)
uv add "sanzaru[mcp]"         # + the MCP server (bare `sanzaru` / `sanzaru serve`)
```

The base install is the CLI and carries nothing from the MCP stack (`mcp`, `starlette`,
`uvicorn`). Anything that launches the server — a `.mcp.json`, Claude Desktop, `codex mcp add` —
must install the `mcp` extra; `sanzaru[all]` includes it. Running bare `sanzaru` without it exits 3
with the install command.

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
        "HF_KEY": "your-higgsfield-key-id:secret",
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
| **Video** | `create_video`, `edit_video`, `extend_video`, `get_video_status`, `download_video`, `cancel_video`, `list_local_videos`, `inspect_video_frame` | Generate, edit and extend video on Higgsfield (priced before submit, optional cost cap); inspect framess with optional reference images |
| **Jobs** | `wait_for` | Block server-side on any mix of `hf_*`/`resp_*` ids until they finish (progress on every poll, optional download); replaces model-driven polling |
| **Image** | `generate_image`, `edit_image`, `create_image`, `get_image_status`, `download_image`, `inspect_image` | `generate_image` is synchronous — one call, finished file (the default). `create_image` starts a background job for batches and refinement chains; collect with `wait_for`. `edit_image` edits; `inspect_image` shows the model what it rendered |
| **Reference** | `list_reference_images`, `prepare_reference_image` | Manage reference images; crop to a video aspect ratio or exact size |
| **Audio** | `transcribe_audio`, `chat_with_audio`, `create_audio`, `convert_audio`, `compress_audio`, `list_audio_files`, `get_latest_audio`, `transcribe_with_enhancement` | Transcription, analysis, TTS (OpenAI or ElevenLabs), and file management |
| **Podcast** | `generate_podcast` | The recommended podcast tool: multi-voice episodes with parallel TTS and stitching (ElevenLabs v4 + dialogue + verify recommended); speakers may mix TTS providers |
| **Simulated Podcast** (experimental) | `simulate_podcast` | Unscripted: realtime agents converse from a rundown — parallel acts, checkpointing, cost ceiling, QC |
| **Media** | `view_media` | Interactive media player via MCP App protocol |

> **Full API documentation**: See [docs/api-reference.md](docs/api-reference.md)

## Basic Workflows

### Generate a Video
```python
# Create video from text
video = create_video(
    prompt="A serene mountain landscape at sunrise, slow aerial push-in",
    duration=8,                 # Seedance 2.5 (default): 4-30 s
    aspect_ratio="16:9",
    resolution="720p",
    max_cost_usd=5,             # refused (and nothing charged) if the estimate is higher
)                               # returns id "hf_...", status, and the estimated cost

# Wait server-side and save it — one call, no polling loop
wait_for([video.id], download=True)
```

### Generate with Reference Image
```python
# 1. Generate reference image (gpt-image-2.5-flare, synchronous)
generate_image(
    prompt="futuristic pilot in mech cockpit",
    size="1536x1024",
    filename="pilot.png"
)

# 2. Crop to the frame shape you want (image-to-video framing follows the image)
prepare_reference_image("pilot.png", aspect_ratio="16:9", resize_mode="crop")

# 3. Animate — prompt only the motion; the image already fixes subject and style
video = create_video(
    prompt="The pilot looks up and smiles",
    reference_image="pilot_1280x720.png",
    duration=5, resolution="480p", max_cost_usd=2,
)
wait_for([video.id], download=True)

# 4. Keep going from the finished job — no download/re-upload
extend_video("He reaches for the throttle.", source_video=video.id, duration=5, max_cost_usd=3)
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

### Simulate a Podcast (experimental — no script, the agents talk)
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
- **[Video Prompting Guide](docs/video-prompting-guide.md)** - Motion-first prompts, image-to-video framing, edit/extend, cost
- **[Audio Features](docs/audio/README.md)** - Audio transcription, chat, and TTS
- **[Simulated Podcasts](docs/audio/simulated-podcasts.md)** (experimental) - Realtime agents in conversation: producer model, act chunking, cost, QC
- **[Performance & Architecture](docs/async-optimizations.md)** - Technical details and benchmarks

## Transport Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **stdio** (default) | `uv run sanzaru` | Claude Desktop, Claude Code, local MCP clients |
| **HTTP** | `uv run sanzaru --transport http` | Remote access, Databricks Apps, web clients |

### Authenticating HTTP mode

HTTP mode exposes the full toolset — paid generation, `cancel_video`, and every
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
- ✅ Free-threaded Python ready (3.15t tested in CI)

See [docs/async-optimizations.md](docs/async-optimizations.md) for technical details.

## License

[MIT](LICENSE)
