# Track C: Infrastructure & Tools Migration

## Agent Role

**You are Track C Agent** - responsible for migrating shared infrastructure and creating audio MCP tools for sanzaru.

**Duration:** ~25 minutes
**Dependencies:** Phase 0 foundation must be complete
**Works in parallel with:** Tracks A, B, D (zero conflicts!)

## Objective

1. Migrate shared infrastructure (OpenAI client, caching, file system) from mcp-server-whisper
2. Create `tools/audio.py` following sanzaru's patterns to register audio MCP tools
3. Add audio tool descriptions to `descriptions.py`

## Prerequisites

- [ ] Phase 0 foundation complete (`src/sanzaru/infrastructure/` directory exists)
- [ ] Working directory: `/Users/rcaputo3/git/sanzaru`
- [ ] Source available: `/Users/rcaputo3/git/mcp-server-whisper`
- [ ] Feature branch: `migration/audio-feature` exists

## Your Directory Scope

You **only** work in these locations:
- `src/sanzaru/infrastructure/` (shared infrastructure)
- `src/sanzaru/tools/audio.py` (NEW - audio tools)
- `src/sanzaru/descriptions.py` (ADD audio tool descriptions)

**You do NOT touch:**
- `src/sanzaru/audio/` ← Track A's territory
- `src/sanzaru/audio/services/` ← Track B's territory
- `tests/` ← Track D's territory
- `docs/` ← Track D's territory

This isolation **prevents merge conflicts**!

## Setup

```bash
cd /Users/rcaputo3/git/sanzaru

# Ensure you're on the foundation branch
git checkout migration/audio-feature

# Create your track branch
git checkout -b migration/track-c-infrastructure

# Verify foundation is in place
ls -la src/sanzaru/infrastructure/
ls -la src/sanzaru/tools/
```

## Part 1: Infrastructure Migration

### Source Files to Migrate

From `/Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/infrastructure/`:

1. **OpenAI Client** → `src/sanzaru/infrastructure/openai_client.py`
2. **Cache** → `src/sanzaru/infrastructure/cache.py`
3. **File System** → `src/sanzaru/infrastructure/file_system.py`
4. **Path Resolver** → `src/sanzaru/infrastructure/path_resolver.py`
5. **Utils** → Check if `utils/text_utils.py` should go here

### Step 1: Migrate OpenAI Client Infrastructure

First, check what's in the source:
```bash
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/infrastructure/openai_client.py
```

**Sanzaru already has** `config.get_client()` - check if we need to merge patterns:
```bash
cat src/sanzaru/config.py | grep -A 10 "def get_client"
```

**Decision:**
- If mcp-server-whisper's `openai_client.py` just wraps `AsyncOpenAI()`, **skip it** - use sanzaru's existing `get_client()`
- If it has additional logic (retries, custom config), migrate to `infrastructure/openai_client.py`

**Most likely:** Skip this file and use sanzaru's `config.get_client()` everywhere.

### Step 2: Migrate Cache Infrastructure

```bash
# Copy cache
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/infrastructure/cache.py \
   src/sanzaru/infrastructure/cache.py
```

**Adapt if needed:**
- Check for any mcp-server-whisper specific imports
- Update docstring to mention sanzaru

**Verify:**
```bash
python -m py_compile src/sanzaru/infrastructure/cache.py
python -c "from sanzaru.infrastructure.cache import cached_function"  # Or whatever exports exist
```

### Step 3: Migrate File System Utilities

```bash
# Copy file system utils
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/infrastructure/file_system.py \
   src/sanzaru/infrastructure/file_system.py
```

**Check for overlap** with sanzaru's existing `security.py`:
```bash
cat src/sanzaru/security.py
```

**If there's overlap:**
- Use sanzaru's `security.py` functions where possible
- Only migrate unique functionality from `file_system.py`
- Or merge them into `security.py`

**Adapt imports:**
```python
# Update any relative imports
from ..exceptions import SomeError  # Should point to sanzaru exceptions
```

**Verify:**
```bash
python -m py_compile src/sanzaru/infrastructure/file_system.py
```

### Step 4: Migrate Path Resolver

```bash
# Copy path resolver
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/infrastructure/path_resolver.py \
   src/sanzaru/infrastructure/path_resolver.py
```

**Check for overlap** with sanzaru's existing `security.validate_safe_path()`:
```bash
grep "validate_safe_path" src/sanzaru/security.py
```

**Likely:** Path resolver and `validate_safe_path` do similar things. Consider:
- Updating PathResolver to use sanzaru's security functions
- Or using sanzaru's pattern directly in audio services

**Adapt imports:**
```python
# Use sanzaru's security utilities
from ..security import validate_safe_path, check_not_symlink
```

**Verify:**
```bash
python -m py_compile src/sanzaru/infrastructure/path_resolver.py
```

### Step 5: Migrate Text Utils (if needed)

Check if text utils exist and are needed:
```bash
ls /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/utils/
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/utils/text_utils.py
```

**If text_utils.py exists:**
```bash
# Copy to infrastructure
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/utils/text_utils.py \
   src/sanzaru/infrastructure/text_utils.py
```

**Verify:**
```bash
python -m py_compile src/sanzaru/infrastructure/text_utils.py
```

### Step 6: Update Infrastructure `__init__.py`

Create `src/sanzaru/infrastructure/__init__.py`:

```python
"""Shared infrastructure for sanzaru - caching, file system, OpenAI client utilities.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

# Only export what was migrated and is useful
from .cache import *  # Adjust based on what's in cache.py
from .file_system import *  # Adjust based on what's in file_system.py
from .path_resolver import PathResolver

__all__ = [
    # Add specific exports
    "PathResolver",
]
```

## Part 2: Audio Tools Creation

### Step 7: Study Existing Sanzaru Tool Pattern

First, understand how sanzaru structures tools:
```bash
cat src/sanzaru/tools/video.py | head -50
cat src/sanzaru/server.py | grep "@mcp.tool" | head -5
```

**Pattern:**
- Tools are async functions in `tools/*.py`
- They delegate to service/domain logic
- Server.py registers them with `@mcp.tool()` decorator
- Descriptions come from `descriptions.py`

### Step 8: Study mcp-server-whisper's Tool Registration

Check how whisper registers tools:
```bash
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/tools/__init__.py
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/tools/audio_tools.py | head -30
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/tools/file_tools.py | head -30
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/tools/transcription_tools.py | head -30
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/tools/tts_tools.py | head -30
```

### Step 9: Create `tools/audio.py`

Create a new file following sanzaru's pattern:

**`src/sanzaru/tools/audio.py`:**

```python
"""Audio tools for sanzaru - Whisper transcription, GPT-4o audio chat, and TTS.

These tools provide MCP interfaces to audio processing capabilities.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from typing import Literal

from ..audio.models import (
    AudioProcessingResult,
    ChatResult,
    FilePathSupportParams,
    TranscriptionResult,
    TTSResult,
)
from ..audio.services import (
    AudioService,
    FileService,
    TranscriptionService,
    TTSService,
)


# ==================== FILE MANAGEMENT TOOLS ====================
async def list_audio_files(
    pattern: str | None = None,
    file_type: str | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    sort_by: Literal["name", "size", "modified", "duration"] = "modified",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
) -> list[FilePathSupportParams]:
    """List audio files with filtering and sorting options."""
    service = FileService()
    return await service.list_files(
        pattern=pattern,
        file_type=file_type,
        min_size=min_size,
        max_size=max_size,
        min_duration=min_duration,
        max_duration=max_duration,
        sort_by=sort_by,
        order=order,
        limit=limit,
    )


async def get_latest_audio() -> FilePathSupportParams:
    """Get the most recently modified audio file."""
    service = FileService()
    return await service.get_latest()


# ==================== AUDIO PROCESSING TOOLS ====================
async def convert_audio(
    input_path: str,
    output_format: Literal["mp3", "wav"],
) -> AudioProcessingResult:
    """Convert audio file to specified format."""
    service = AudioService()
    return await service.convert(input_path, output_format)


async def compress_audio(
    input_path: str,
    target_size_mb: int | None = None,
) -> AudioProcessingResult:
    """Compress audio file to reduce size."""
    service = AudioService()
    return await service.compress(input_path, target_size_mb)


# ==================== TRANSCRIPTION TOOLS ====================
async def transcribe_audio(
    file_path: str,
    model: Literal["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] = "whisper-1",
    prompt: str | None = None,
    timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
) -> TranscriptionResult:
    """Transcribe audio file using OpenAI Whisper or GPT-4o models."""
    service = TranscriptionService()
    return await service.transcribe(
        file_path=file_path,
        model=model,
        prompt=prompt,
        timestamp_granularities=timestamp_granularities,
        response_format=response_format,
    )


async def chat_with_audio(
    file_path: str,
    model: str = "gpt-4o-audio-preview",
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> ChatResult:
    """Interactive audio analysis using GPT-4o audio models."""
    service = TranscriptionService()
    return await service.chat(
        file_path=file_path,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


async def transcribe_with_enhancement(
    file_path: str,
    enhancement_type: Literal["detailed", "storytelling", "professional", "analytical"],
    model: str = "whisper-1",
) -> TranscriptionResult:
    """Enhanced transcription with specialized templates."""
    service = TranscriptionService()
    return await service.transcribe_enhanced(
        file_path=file_path,
        enhancement_type=enhancement_type,
        model=model,
    )


# ==================== TEXT-TO-SPEECH TOOLS ====================
async def create_audio(
    text_prompt: str,
    voice: Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"] = "shimmer",
    model: str = "gpt-4o-mini-tts",
    speed: float = 1.0,
    instructions: str | None = None,
    output_filename: str | None = None,
) -> TTSResult:
    """Generate text-to-speech audio using OpenAI TTS API."""
    service = TTSService()
    return await service.create(
        text=text_prompt,
        voice=voice,
        model=model,
        speed=speed,
        instructions=instructions,
        output_filename=output_filename,
    )
```

**Note:** Adjust parameter names and types based on actual service signatures in Track B's work.

**Verify:**
```bash
python -m py_compile src/sanzaru/tools/audio.py
```

### Step 10: Add Audio Tool Descriptions

Update `src/sanzaru/descriptions.py` to include audio tool descriptions:

```python
# At the end of descriptions.py, add:

# ==================== AUDIO TOOL DESCRIPTIONS ====================

LIST_AUDIO_FILES = """List audio files with comprehensive filtering and sorting options.

Supports:
- Pattern matching on filenames (regex)
- File type filtering (mp3, wav, etc.)
- Size filtering (min/max bytes)
- Duration filtering (min/max seconds)
- Sorting by name, size, modified time, or duration
- Result limiting for performance

Returns list of FilePathSupportParams with full metadata.
"""

GET_LATEST_AUDIO = """Get the most recently modified audio file.

Returns the latest audio file with model support information.
Useful for quick access to recent recordings.
"""

CONVERT_AUDIO = """Convert audio files between supported formats.

Supports conversion to:
- mp3: Compressed audio (smaller file size)
- wav: Uncompressed audio (higher quality)

Returns AudioProcessingResult with output path.
"""

COMPRESS_AUDIO = """Compress audio files that exceed size limits.

Automatically adjusts bitrate to meet target size.
Useful for preparing files for API upload (25MB limit).

Returns AudioProcessingResult with compressed file path.
"""

TRANSCRIBE_AUDIO = """Transcribe audio using OpenAI Whisper or GPT-4o models.

Supports:
- whisper-1: Standard Whisper transcription
- gpt-4o-transcribe: Enhanced accuracy with GPT-4o
- gpt-4o-mini-transcribe: Fast transcription

Features:
- Custom prompts for guided transcription
- Timestamp granularities (word/segment level)
- Multiple response formats (json, text, srt, vtt)

Returns TranscriptionResult with text and metadata.
"""

CHAT_WITH_AUDIO = """Interactive audio analysis using GPT-4o audio models.

Have a conversation about audio content:
- Ask questions about what's said
- Analyze tone and emotion
- Summarize or extract insights

Recommended model: gpt-4o-audio-preview

Returns ChatResult with conversational response.
"""

TRANSCRIBE_WITH_ENHANCEMENT = """Enhanced transcription with specialized templates.

Enhancement types:
- detailed: Includes tone, emotion, and background details
- storytelling: Transforms transcript into narrative form
- professional: Creates formal, business-appropriate output
- analytical: Adds analysis of speech patterns and key points

Returns TranscriptionResult with enhanced output.
"""

CREATE_AUDIO = """Generate text-to-speech audio using OpenAI TTS API.

Features:
- 8+ voice options (alloy, ash, ballad, coral, echo, sage, shimmer, verse)
- Speed adjustment (0.25x to 4.0x)
- Custom instructions for voice style
- Automatic text splitting for long inputs

Recommended model: gpt-4o-mini-tts

Returns TTSResult with output file path.
"""
```

**Verify:**
```bash
python -m py_compile src/sanzaru/descriptions.py
```

## Self-Validation

Run these checks before committing:

```bash
cd /Users/rcaputo3/git/sanzaru

# 1. Check infrastructure files
ls -la src/sanzaru/infrastructure/
# Should see: __init__.py, cache.py, file_system.py, path_resolver.py, (maybe text_utils.py)

# 2. Check audio tools created
test -f src/sanzaru/tools/audio.py && echo "✓ audio tools created"

# 3. Syntax validation
python -m py_compile src/sanzaru/infrastructure/*.py
python -m py_compile src/sanzaru/tools/audio.py
python -m py_compile src/sanzaru/descriptions.py

# 4. Import validation
python -c "from sanzaru.infrastructure import cache"
python -c "from sanzaru.tools import audio"
python -c "from sanzaru import descriptions"

# 5. Check no old package references
grep -r "mcp_server_whisper" src/sanzaru/infrastructure/ && echo "ERROR!" || echo "✓ Clean"
grep -r "mcp_server_whisper" src/sanzaru/tools/audio.py && echo "ERROR!" || echo "✓ Clean"

# 6. Verify description constants exist
python -c "from sanzaru.descriptions import LIST_AUDIO_FILES, TRANSCRIBE_AUDIO, CREATE_AUDIO"
```

## Git Commit

```bash
cd /Users/rcaputo3/git/sanzaru

# Stage your changes
git add src/sanzaru/infrastructure/
git add src/sanzaru/tools/audio.py
git add src/sanzaru/descriptions.py

# Commit with descriptive message
git commit -m "migration: infrastructure and audio tools (Track C)

Migrate shared infrastructure and create audio MCP tools:

Infrastructure added:
- infrastructure/cache.py: Caching utilities for performance
- infrastructure/file_system.py: Safe file operations
- infrastructure/path_resolver.py: Path validation and resolution
- infrastructure/text_utils.py: Text processing utilities
- infrastructure/__init__.py: Infrastructure exports

Audio tools added:
- tools/audio.py: 9 MCP tools for audio operations
  - File management: list_audio_files, get_latest_audio
  - Processing: convert_audio, compress_audio
  - Transcription: transcribe_audio, chat_with_audio, transcribe_with_enhancement
  - TTS: create_audio

Tool descriptions added:
- 8 audio tool descriptions in descriptions.py

All tools follow sanzaru patterns (async, type-safe, delegating to services).
Ready for server.py registration in Phase 2 integration.

Source: mcp-server-whisper v1.1.0 by Richie Caputo (MIT license)
Track: C (Infrastructure & Tools)
"

# Verify commit
git log -1 --stat
git show --name-status
```

## Success Criteria

- [ ] Infrastructure files migrated (cache, file_system, path_resolver, text_utils)
- [ ] `infrastructure/__init__.py` exports key components
- [ ] `tools/audio.py` created with all audio MCP tools
- [ ] Audio tool descriptions added to `descriptions.py`
- [ ] All Python files have valid syntax
- [ ] No references to `mcp_server_whisper` package
- [ ] Imports use sanzaru structure
- [ ] Tools follow sanzaru's async patterns
- [ ] All self-validation checks pass
- [ ] Committed to `migration/track-c-infrastructure` branch
- [ ] Commit message is descriptive

## Common Issues

### Issue: Duplicate Functionality with Sanzaru

```python
# If infrastructure overlaps with existing sanzaru code:
# Prefer sanzaru's implementation
# Update audio services to use sanzaru's utilities
```

### Issue: Service Signatures Unknown

```python
# Track B might still be in progress
# Check mcp-server-whisper source for actual service method signatures
# Update tools/audio.py parameters to match
```

## Time Estimate

- Step 1-6: ~10 minutes (infrastructure migration)
- Step 7-9: ~10 minutes (create audio tools)
- Step 10: ~3 minutes (add descriptions)
- Validation & commit: ~2 minutes

**Total: ~25 minutes**

## Notes

- You're working in isolated directories (infrastructure/ and tools/audio.py)
- No conflicts with other tracks
- If Track B isn't done, check source for service signatures
- Phase 2 will register your tools in server.py
- Keep tool functions thin - delegate to services!

## Questions?

Refer to:
- `audio-migration-feature-spec.md` - Overall architecture
- `src/sanzaru/tools/video.py` - Tool pattern examples
- `src/sanzaru/server.py` - How tools are registered
- `/Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/tools/` - Source tool implementations
