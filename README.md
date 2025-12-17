# sanzaru

<div align="center">
  <img src="https://raw.githubusercontent.com/TJC-LP/sanzaru/main/assets/logo.png" alt="sanzaru logo" width="400">
</div>

A **stateless**, lightweight **MCP** server that wraps **OpenAI's Sora Video API, Whisper, and GPT-4o Audio APIs** via the OpenAI Python SDK.

## Features

### Video Generation (Sora)
- Create videos with `sora-2` or `sora-2-pro` models
- Use reference images to guide generation
- Remix and refine existing videos
- Download variants (video, thumbnail, spritesheet)

### Image Generation
- Generate images with gpt-image-1.5 (recommended) or GPT-5
- Edit and compose images with up to 16 inputs
- Iterative refinement via Responses API
- Automatic resizing for Sora compatibility

### Audio Processing
- **Transcription**: Whisper and GPT-4o models
- **Audio Chat**: Interactive analysis with GPT-4o
- **Text-to-Speech**: Multi-voice TTS generation
- **Processing**: Format conversion, compression, file management

> **Note:** Content guardrails are enforced by OpenAI. This server does not run local moderation.

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` environment variable

**Feature-specific paths** (set only what you need):
- `VIDEO_PATH` - Enables video generation features
- `IMAGE_PATH` - Enables image generation features
- `AUDIO_PATH` - Enables audio processing features

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

## Installation

### Quick Install
```bash
# All features
uv add "sanzaru[all]"

# Specific features
uv add "sanzaru[audio]"  # With audio support
uv add sanzaru           # Base (video + image only)
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
        "VIDEO_PATH": "/absolute/path/to/videos",
        "IMAGE_PATH": "/absolute/path/to/images",
        "AUDIO_PATH": "/absolute/path/to/audio"
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
  --env VIDEO_PATH="$HOME/sanzaru-videos" \
  --env IMAGE_PATH="$HOME/sanzaru-images" \
  --env AUDIO_PATH="$HOME/sanzaru-audio" \
  -- uvx "sanzaru[all]"

# Or from source
cd /path/to/sanzaru
set -a; source .env; set +a
codex mcp add sanzaru \
  --env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --env VIDEO_PATH="$VIDEO_PATH" \
  --env IMAGE_PATH="$IMAGE_PATH" \
  --env AUDIO_PATH="$AUDIO_PATH" \
  -- uv run --directory "$(pwd)" sanzaru
```

### Manual Setup
```bash
uv venv
uv sync

# Set required environment variables
export OPENAI_API_KEY=sk-...
export VIDEO_PATH=~/videos
export IMAGE_PATH=~/images
export AUDIO_PATH=~/audio

# Run server
uv run sanzaru
```

**Feature Auto-Detection:** Features are automatically enabled based on configured paths. Set only the paths you need.

</details>

## Available Tools

| Category | Tools | Description |
|----------|-------|-------------|
| **Video** | `create_video`, `get_video_status`, `download_video`, `list_videos`, `delete_video`, `remix_video` | Generate and manage Sora videos with optional reference images |
| **Image** | `generate_image`, `edit_image`, `create_image`, `get_image_status`, `download_image` | Generate with gpt-image-1.5 (sync) or GPT-5 (polling) |
| **Reference** | `list_reference_images`, `prepare_reference_image` | Manage and resize images for Sora compatibility |
| **Audio** | `transcribe_audio`, `chat_with_audio`, `create_audio`, `convert_audio`, `compress_audio`, `list_audio_files`, `get_latest_audio`, `transcribe_with_enhancement` | Transcription, analysis, TTS, and file management |

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
# 1. Generate reference image (gpt-image-1.5, synchronous)
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

# Transcribe
result = transcribe_audio("interview.mp3")

# Or analyze with GPT-4o
analysis = chat_with_audio(
    "meeting.mp3",
    user_prompt="Summarize key decisions and action items"
)
```

## Documentation

- **[API Reference](docs/api-reference.md)** - Complete tool documentation with parameters and examples
- **[Reference Images Guide](docs/reference-images.md)** - Working with reference images and resizing
- **[Image Generation Guide](docs/image-generation.md)** - Generating and editing reference images
- **[Sora Prompting Guide](docs/sora2-prompting-guide.md)** - Crafting effective video prompts
- **[Audio Features](docs/audio/README.md)** - Audio transcription, chat, and TTS
- **[Performance & Architecture](docs/async-optimizations.md)** - Technical details and benchmarks

## Performance

Fully asynchronous architecture with proven scalability:
- ✅ 32+ concurrent operations verified
- ✅ 8-10x speedup for parallel tasks
- ✅ Non-blocking I/O with `aiofiles` + `anyio`
- ✅ Python 3.14 free-threading ready

See [docs/async-optimizations.md](docs/async-optimizations.md) for technical details.

## License

[MIT](LICENSE)
