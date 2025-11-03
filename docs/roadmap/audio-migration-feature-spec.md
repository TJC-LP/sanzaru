# Sanzaru Multimodal Migration: Feature Specification

## Vision

Transform **sanzaru** from a Sora-focused MCP server into a unified multimodal OpenAI integrations hub. Users can install subsets of features based on their needs via optional dependencies: `sanzaru[audio]`, `sanzaru[video]`, `sanzaru[image]`, or `sanzaru[all]`.

## Goals

1. **Modular Architecture**: Clean separation of concerns with optional feature groups
2. **Zero Breaking Changes**: Existing sanzaru users continue working without modification
3. **Graceful Degradation**: Missing optional dependencies don't break the server
4. **Code Quality**: Maintain high standards - type safety, tests, documentation
5. **Performance**: Preserve async architecture and parallel processing capabilities
6. **Developer Experience**: Clear structure, consistent patterns, excellent docs

## Feature Matrix

| Feature Group | What It Provides | Optional Deps | Env Vars Required |
|---------------|------------------|---------------|-------------------|
| **video** (core) | Sora video generation, remix, status polling | None (in base) | `VIDEO_PATH` |
| **image** (core) | GPT-5/4.1 image generation, reference management | `pillow` | `IMAGE_PATH` |
| **audio** (NEW) | Whisper transcription, GPT-4o audio chat, TTS | `pydub`, `ffmpeg-python`, `audioop-lts` | `AUDIO_FILES_PATH` |

**Core principle:** Base package includes video. Image and audio are optional extras.

## Architecture Overview

### Directory Structure

```
src/sanzaru/
├── __init__.py              # Core exports
├── __main__.py
├── server.py                # Main server with conditional tool registration
├── config.py                # Unified config for all features
├── security.py              # Shared security utilities
├── utils.py                 # Shared helpers
├── types.py                 # Shared type definitions
├── descriptions.py          # Tool descriptions for all features
├── tools/
│   ├── __init__.py
│   ├── video.py             # Existing: Sora tools
│   ├── reference.py         # Existing: Image reference tools
│   ├── image.py             # Existing: GPT image generation
│   └── audio.py             # NEW: Whisper/TTS tools
├── audio/                   # NEW: Audio feature domain
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── models.py
│   ├── processor.py
│   ├── file_filter.py
│   └── services/
│       ├── __init__.py
│       ├── audio_service.py
│       ├── file_service.py
│       ├── transcription_service.py
│       └── tts_service.py
└── infrastructure/          # NEW: Shared infrastructure
    ├── __init__.py
    ├── openai_client.py
    ├── cache.py
    └── file_system.py
```

### Testing Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (pure functions)
│   ├── test_utils.py
│   ├── test_security.py
│   ├── test_image_processing.py
│   └── test_config.py
├── integration/             # Integration tests (mocked APIs)
│   ├── test_video_tools.py
│   ├── test_image_tools.py
│   └── test_reference_tools.py
└── audio/                   # NEW: Audio feature tests
    ├── test_audio_processor.py
    ├── test_file_filter.py
    ├── test_audio_service.py
    ├── test_file_service.py
    ├── test_transcription_service.py
    ├── test_tts_service.py
    └── test_audio_tools.py
```

### Documentation Structure

```
docs/
├── async-optimizations.md          # Existing
├── sora-prompting-guide.md         # Existing
├── sora2_prompting_guide.ipynb     # Existing
├── audio/                          # NEW: Audio documentation
│   ├── README.md                   # Audio feature guide
│   ├── architecture.md
│   ├── openai-audio.md
│   └── openai-realtime.md
└── roadmap/                        # Migration roadmap
    └── [this directory]
```

## Audio Feature Tools

### Tool Inventory (from mcp-server-whisper)

#### File Management
- `list_audio_files(pattern?, min_size?, max_size?, min_duration?, max_duration?, format?, sort_by?, order?, limit?)`
- `get_latest_audio()` - Get most recent audio file

#### Audio Processing
- `convert_audio(input_path, output_format)` - Convert between formats
- `compress_audio(input_path, target_size?)` - Compress oversized files

#### Transcription
- `transcribe_audio(file_path, model?, prompt?, timestamp_granularities?, response_format?)`
- `chat_with_audio(file_path, model?, system_prompt?, user_prompt?)` - GPT-4o audio chat
- `transcribe_with_enhancement(file_path, enhancement_type, model?)` - Enhanced transcription with templates

#### Text-to-Speech
- `create_audio(text_prompt, voice?, model?, speed?, instructions?, output_filename?)` - Generate TTS audio

### Supported Models

**Transcription:**
- `whisper-1` - Standard Whisper
- `gpt-4o-transcribe` - Enhanced transcription
- `gpt-4o-mini-transcribe` - Fast transcription

**Audio Chat:**
- `gpt-4o-audio-preview` (recommended)

**Text-to-Speech:**
- `gpt-4o-mini-tts` (preferred)
- Voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar

### Audio Format Support

| Model Type | Supported Formats |
|------------|-------------------|
| Transcription | flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm |
| Chat | mp3, wav |
| TTS Output | mp3, opus, aac, flac, wav, pcm |

## Configuration Design

### Environment Variables

```bash
# Core (always required)
OPENAI_API_KEY=sk-...

# Video feature (required if using video tools)
VIDEO_PATH=/path/to/videos

# Image feature (required if using image tools)
IMAGE_PATH=/path/to/images

# Audio feature (required if using audio tools)
AUDIO_FILES_PATH=/path/to/audio
```

### Unified Config Pattern

Extend existing `config.py` pattern:

```python
@lru_cache(maxsize=3)
def get_path(path_type: Literal["video", "reference", "audio"]) -> pathlib.Path:
    """Get and validate a configured path from environment.

    Args:
        path_type: "video", "reference", or "audio"

    Returns:
        Validated absolute path

    Raises:
        RuntimeError: If environment variable not set or invalid
    """
```

### Conditional Tool Registration

```python
# server.py
def _check_audio_available() -> bool:
    """Check if audio dependencies are installed."""
    try:
        import pydub
        import ffmpeg
        return True
    except ImportError:
        logger.info("Audio dependencies not installed - audio tools disabled")
        return False

def main():
    """Run the MCP server."""
    logger.info("Starting sanzaru MCP server")
    load_dotenv()

    # Always register video and image tools (core features)
    # ... existing tool registration ...

    # Conditionally register audio tools
    if _check_audio_available():
        from .tools.audio import register_audio_tools
        register_audio_tools(mcp)
        logger.info("Audio tools registered")

    mcp.run()
```

## Package Configuration

### pyproject.toml Structure

```toml
[project]
name = "sanzaru"
version = "0.2.0"
description = "Unified MCP server for OpenAI multimodal APIs (Sora, Whisper, GPT Vision)"
requires-python = ">=3.10"
dependencies = [
    "openai>=2.6.0",
    "mcp>=1.20.0",
    "httpx>=0.27.0",
    "anyio>=4.0.0",
    "aiofiles>=24.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.11.0",
    "python-dotenv",
]

[project.optional-dependencies]
audio = [
    "pydub",
    "ffmpeg-python",
    "async-lru>=2.0.5",
    "audioop-lts; python_version >= '3.13'",
]
image = [
    "pillow>=12.0.0",
]
video = []  # No extra deps - video is always available
all = [
    "sanzaru[audio,image]",
]

[dependency-groups]
dev = [
    "ruff>=0.14.3",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "pytest-cov>=6.0.0",
    "pre-commit>=3.6.0",
    "mypy>=1.18.0",
    "types-aiofiles",
]

[project.scripts]
sanzaru = "sanzaru.server:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
asyncio_mode = "auto"
markers = [
    "unit: Unit tests for pure functions (no external dependencies)",
    "integration: Integration tests with mocked external APIs",
    "audio: Tests for audio feature (requires audio dependencies)",
]
```

### Installation Examples

```bash
# Base package (video + image capabilities)
uv add sanzaru

# With audio support
uv add "sanzaru[audio]"

# With all features
uv add "sanzaru[all]"

# Development installation
cd sanzaru
uv sync --all-extras
```

## Migration Source

### From: mcp-server-whisper

**Repository:** `/Users/rcaputo3/git/mcp-server-whisper`

**Key Stats:**
- ~2,600 lines of Python code
- 12 test files with comprehensive coverage
- 5 documentation files
- Well-structured domain-driven architecture

**Migration Approach:**
- Clean copy (no git history preservation)
- Adapt imports to sanzaru structure
- Maintain all functionality and tests
- Update documentation for integrated experience

## Code Quality Standards

### Must Preserve

1. **Type Safety**
   - All type hints from both codebases
   - Strict mypy validation
   - Pydantic models for data validation

2. **Async Architecture**
   - Fully non-blocking I/O
   - Structured concurrency with anyio
   - Thread pools for CPU-bound operations

3. **Testing**
   - Unit tests for pure functions (100% coverage target)
   - Integration tests with mocked APIs (80%+ coverage)
   - All existing tests migrated and passing

4. **Security**
   - Path traversal protection
   - Symlink validation
   - Sandboxed file access

5. **Documentation**
   - Comprehensive docstrings
   - Usage examples
   - Architecture documentation
   - Migration attribution

### Code Style Consistency

- Line length: 120 characters
- Ruff formatting (double quotes, space indentation)
- Import order: stdlib → typing → third-party → local
- Function naming: `verb_noun` (predicate-first)
- Class naming: PascalCase
- Constants: UPPER_SNAKE_CASE

## Success Criteria

### Functional Requirements

- [ ] All audio tools from mcp-server-whisper work in sanzaru
- [ ] All existing video/image tools continue working
- [ ] Optional dependencies work as documented
- [ ] Server starts successfully with/without audio deps
- [ ] Environment variables validated correctly
- [ ] Graceful error messages for missing deps/config

### Technical Requirements

- [ ] All tests pass (unit, integration, audio)
- [ ] Type checking passes (mypy --strict)
- [ ] Linting passes (ruff check)
- [ ] Test coverage ≥ 80% for new code
- [ ] No performance regression
- [ ] Documentation complete and accurate

### User Experience

- [ ] Installation instructions clear and tested
- [ ] Feature discovery intuitive (which tools are available?)
- [ ] Error messages helpful (missing dep vs. config vs. runtime error)
- [ ] Migration attribution proper (credit mcp-server-whisper)
- [ ] README updated with feature matrix
- [ ] CLAUDE.md updated with audio patterns

## Timeline

| Phase | Duration | Work Type |
|-------|----------|-----------|
| Phase 0: Foundation | ~5 mins | Sequential (1 agent) |
| Phase 1: Parallel Migration | ~25 mins | Parallel (4 agents) |
| Phase 2: Integration | ~10 mins | Sequential (1 agent) |
| **Total Elapsed Time** | **~40 mins** | 3x speedup vs sequential |

## Risk Mitigation

1. **Merge Conflicts**: Eliminated by design - parallel tracks work in separate directories
2. **Import Errors**: Each track has self-validation steps before integration
3. **Test Failures**: Full test suite runs in Phase 2 before final commit
4. **Missing Dependencies**: Conditional imports with helpful logging
5. **Breaking Changes**: Existing functionality untouched, new features additive
6. **Rollback**: Each track on separate branch, can discard/redo individually

## Attribution

This migration incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) by Richie Caputo, licensed under MIT. The original project provided production-ready audio processing capabilities that are being integrated into sanzaru's unified multimodal architecture.
