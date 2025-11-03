# Phase 2: Integration & Wiring

## Agent Role

**You are the Integration Agent.** Your job is to wire everything together after all parallel tracks complete.

**Duration:** ~10 minutes
**Dependencies:** ALL Phase 1 tracks (A, B, C, D) must be complete
**Next Phase:** Done! Ready to merge to main.

## Objective

Wire up the audio feature into sanzaru's server:
1. Update `server.py` to conditionally register audio tools
2. Update `config.py` to support `AUDIO_FILES_PATH`
3. Create integration tests
4. Run full test suite
5. Update `CLAUDE.md` with audio patterns
6. Final validation

## Prerequisites

- [ ] Phase 1 complete: All tracks merged to `migration/audio-feature` branch
- [ ] Working directory: `/Users/rcaputo3/git/sanzaru`
- [ ] On branch: `migration/audio-feature`
- [ ] All four track branches merged

## Setup

```bash
cd /Users/rcaputo3/git/sanzaru

# Ensure all tracks are merged
git checkout migration/audio-feature
git log --oneline --graph | head -20

# Should see merge commits from tracks A, B, C, D

# Verify structure is complete
ls -la src/sanzaru/audio/
ls -la src/sanzaru/audio/services/
ls -la src/sanzaru/infrastructure/
ls -la src/sanzaru/tools/audio.py
ls -la tests/audio/
ls -la docs/audio/
```

## Integration Tasks

### Task 1: Update `config.py` for Audio Path Support

Add support for `AUDIO_FILES_PATH` alongside existing `VIDEO_PATH` and `IMAGE_PATH`.

**Edit `src/sanzaru/config.py`:**

```python
# Update the Literal type annotation
@lru_cache(maxsize=3)  # Increase cache size from 2 to 3
def get_path(path_type: Literal["video", "reference", "audio"]) -> pathlib.Path:
    """Get and validate a configured path from environment.

    Requires explicit environment variable configuration - no defaults.
    Creates paths lazily at runtime, so this works with both `uv run` and `mcp run`.

    Security: Rejects symlinks in environment variable paths to prevent directory traversal.

    Args:
        path_type: Either "video" for VIDEO_PATH, "reference" for IMAGE_PATH, or "audio" for AUDIO_FILES_PATH

    Returns:
        Validated absolute path

    Raises:
        RuntimeError: If environment variable not set, malformed, path doesn't exist, isn't a directory, or is a symlink
    """
    if path_type == "video":
        path_str = os.getenv("VIDEO_PATH")
        env_var = "VIDEO_PATH"
        error_name = "Video download directory"
    elif path_type == "reference":  # Changed from 'else' to 'elif'
        path_str = os.getenv("IMAGE_PATH")
        env_var = "IMAGE_PATH"
        error_name = "Image directory"
    else:  # audio
        path_str = os.getenv("AUDIO_FILES_PATH")
        env_var = "AUDIO_FILES_PATH"
        error_name = "Audio files directory"

    # Rest of the function stays the same
    # ... (existing validation logic)
```

**Verify:**
```bash
python -c "from sanzaru.config import get_path; print(get_path.__doc__)"
```

### Task 2: Update `server.py` to Register Audio Tools

Add conditional audio tool registration to `server.py`.

**Edit `src/sanzaru/server.py`:**

Add import at the top:
```python
from typing import Literal

# ... existing imports ...
```

Add helper function after imports, before tool definitions:
```python
# ==================== FEATURE DETECTION ====================
def _check_audio_available() -> bool:
    """Check if audio dependencies are installed.

    Returns:
        True if audio feature can be enabled, False otherwise.
    """
    try:
        import pydub  # Core audio dependency
        import ffmpeg  # Audio processing
        logger.info("Audio dependencies detected - audio tools will be enabled")
        return True
    except ImportError as e:
        logger.info(f"Audio dependencies not available - audio tools disabled: {e}")
        return False
```

Add audio tool registration at the end, before `# ==================== SERVER ENTRYPOINT ====================`:

```python
# ==================== AUDIO TOOLS (CONDITIONAL) ====================
if _check_audio_available():
    # Import audio tool descriptions
    from .descriptions import (
        LIST_AUDIO_FILES,
        GET_LATEST_AUDIO,
        CONVERT_AUDIO,
        COMPRESS_AUDIO,
        TRANSCRIBE_AUDIO,
        CHAT_WITH_AUDIO,
        TRANSCRIBE_WITH_ENHANCEMENT,
        CREATE_AUDIO,
    )
    # Import audio tool functions
    from .tools import audio

    @mcp.tool(description=LIST_AUDIO_FILES)
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
    ):
        return await audio.list_audio_files(
            pattern, file_type, min_size, max_size, min_duration, max_duration, sort_by, order, limit
        )

    @mcp.tool(description=GET_LATEST_AUDIO)
    async def get_latest_audio():
        return await audio.get_latest_audio()

    @mcp.tool(description=CONVERT_AUDIO)
    async def convert_audio(input_path: str, output_format: Literal["mp3", "wav"]):
        return await audio.convert_audio(input_path, output_format)

    @mcp.tool(description=COMPRESS_AUDIO)
    async def compress_audio(input_path: str, target_size_mb: int | None = None):
        return await audio.compress_audio(input_path, target_size_mb)

    @mcp.tool(description=TRANSCRIBE_AUDIO)
    async def transcribe_audio(
        file_path: str,
        model: Literal["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] = "whisper-1",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
    ):
        return await audio.transcribe_audio(file_path, model, prompt, timestamp_granularities, response_format)

    @mcp.tool(description=CHAT_WITH_AUDIO)
    async def chat_with_audio(
        file_path: str,
        model: str = "gpt-4o-audio-preview",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        return await audio.chat_with_audio(file_path, model, system_prompt, user_prompt)

    @mcp.tool(description=TRANSCRIBE_WITH_ENHANCEMENT)
    async def transcribe_with_enhancement(
        file_path: str,
        enhancement_type: Literal["detailed", "storytelling", "professional", "analytical"],
        model: str = "whisper-1",
    ):
        return await audio.transcribe_with_enhancement(file_path, enhancement_type, model)

    @mcp.tool(description=CREATE_AUDIO)
    async def create_audio(
        text_prompt: str,
        voice: Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"] = "shimmer",
        model: str = "gpt-4o-mini-tts",
        speed: float = 1.0,
        instructions: str | None = None,
        output_filename: str | None = None,
    ):
        return await audio.create_audio(text_prompt, voice, model, speed, instructions, output_filename)

    logger.info("Audio tools registered successfully")
```

**Verify syntax:**
```bash
python -m py_compile src/sanzaru/server.py
```

### Task 3: Update `CLAUDE.md` with Audio Patterns

Add audio feature documentation to `CLAUDE.md`.

**Edit `CLAUDE.md`** - add after the Sora section:

```markdown
## Audio Feature (Optional)

The audio feature provides Whisper transcription, GPT-4o audio chat, and text-to-speech capabilities.

### Installation

```bash
uv sync --extra audio
```

### Configuration

```bash
AUDIO_FILES_PATH=/path/to/audio/files
```

### Available Tools

- `list_audio_files()` - List and filter audio files
- `get_latest_audio()` - Get most recent audio file
- `convert_audio()` - Convert between mp3 and wav
- `compress_audio()` - Compress oversized files
- `transcribe_audio()` - Whisper/GPT-4o transcription
- `chat_with_audio()` - Interactive audio analysis
- `transcribe_with_enhancement()` - Enhanced transcription with templates
- `create_audio()` - Text-to-speech generation

### Example Usage

```python
# List recent audio files
files = list_audio_files(limit=5, sort_by="modified", order="desc")

# Transcribe latest
latest = get_latest_audio()
result = transcribe_audio(latest.path, model="gpt-4o-transcribe")

# Generate TTS
audio_file = create_audio(
    text_prompt="Hello from sanzaru!",
    voice="shimmer",
    model="gpt-4o-mini-tts"
)
```

### Graceful Degradation

If audio dependencies are not installed, audio tools will not be registered. The server will log:
```
Audio dependencies not available - audio tools disabled
```

Install with: `uv sync --extra audio`
```

### Task 4: Create Integration Test

Create a simple integration test to verify audio feature registration:

**Create `tests/integration/test_audio_integration.py`:**

```python
"""Integration test for audio feature registration."""

import pytest


@pytest.mark.integration
@pytest.mark.audio
def test_audio_tools_register_when_available():
    """Test that audio tools are registered when dependencies are available."""
    try:
        import pydub  # noqa: F401

        # If we can import pydub, audio should be available
        from sanzaru.tools import audio

        # Verify key functions exist
        assert hasattr(audio, "list_audio_files")
        assert hasattr(audio, "transcribe_audio")
        assert hasattr(audio, "create_audio")
    except ImportError:
        pytest.skip("Audio dependencies not installed")


@pytest.mark.integration
def test_config_supports_audio_path():
    """Test that config supports audio path type."""
    from sanzaru.config import get_path

    # Should accept 'audio' as a path type (will fail if env var not set, that's okay)
    try:
        path = get_path("audio")
        assert path is not None
    except RuntimeError as e:
        # Expected if AUDIO_FILES_PATH not set
        assert "AUDIO_FILES_PATH" in str(e)
```

**Verify:**
```bash
python -m py_compile tests/integration/test_audio_integration.py
```

### Task 5: Run Full Test Suite

Run all tests to ensure nothing broke:

```bash
cd /Users/rcaputo3/git/sanzaru

# Run unit tests (should pass without audio deps)
pytest tests/unit -m unit -v

# Run integration tests (may skip audio if deps missing)
pytest tests/integration -v

# Run audio tests (will fail/skip if audio deps not installed)
pytest tests/audio -m audio -v

# Type checking
mypy --strict src/sanzaru/ || echo "Check mypy errors"

# Linting
ruff check src/sanzaru/

# Format check
ruff format --check src/sanzaru/
```

**Expected outcomes:**
- Unit tests: PASS (no audio deps needed)
- Integration tests: PASS (audio tests may skip)
- Audio tests: PASS if deps installed, SKIP if not
- Mypy: Should pass (fix any type errors)
- Ruff: Should pass (fix any lint errors)

### Task 6: Test Installation Scenarios

Verify optional dependencies work:

```bash
# Test base install (video + image only)
uv sync
uv run python -c "from sanzaru.tools import video, image; print('✓ Base features work')"

# Test with audio
uv sync --extra audio
uv run python -c "from sanzaru.tools import audio; print('✓ Audio feature works')"

# Test all extras
uv sync --all-extras
uv run python -c "from sanzaru.tools import video, image, audio; print('✓ All features work')"
```

### Task 7: Update Migration Status

Update `docs/audio/MIGRATION_STATUS.md`:

```markdown
# Audio Feature Migration Status

**Start Date:** [Original start date]
**Completion Date:** [Today's date]
**Source:** mcp-server-whisper v1.1.0
**Target:** sanzaru v0.2.0

## Progress

- [x] Phase 0: Foundation
- [x] Phase 1: Parallel Migration
  - [x] Track A: Audio Domain Logic
  - [x] Track B: Audio Services
  - [x] Track C: Infrastructure & Tools
  - [x] Track D: Tests & Documentation
- [x] Phase 2: Integration

## Integration Details

- server.py: Audio tools registered conditionally
- config.py: AUDIO_FILES_PATH support added
- CLAUDE.md: Audio patterns documented
- Tests: Full suite passing
- Documentation: Complete

## Final Validation

- [x] All tests pass
- [x] Type checking passes
- [x] Linting passes
- [x] Installation scenarios verified
- [x] Documentation complete

## Attribution

This migration incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) by Richie Caputo, licensed under MIT.

**Migration completed successfully! 🎉**
```

## Final Validation Checklist

Run through this checklist:

```bash
cd /Users/rcaputo3/git/sanzaru

# 1. All source code compiles
python -m py_compile src/sanzaru/**/*.py

# 2. Server imports successfully
python -c "from sanzaru import server; print('✓ Server imports')"

# 3. Config supports audio path
python -c "from sanzaru.config import get_path; print(get_path.__annotations__)"

# 4. Audio tools exist
python -c "from sanzaru.tools import audio; print('✓ Audio tools exist')"

# 5. Test suite structure is valid
pytest --collect-only tests/

# 6. Documentation exists
ls docs/audio/README.md
grep "Audio Processing" README.md

# 7. Version is updated
grep "version = \"0.2.0\"" pyproject.toml

# 8. CLAUDE.md mentions audio
grep "Audio Feature" CLAUDE.md
```

## Git Commit

```bash
cd /Users/rcaputo3/git/sanzaru

# Stage integration changes
git add src/sanzaru/server.py
git add src/sanzaru/config.py
git add CLAUDE.md
git add tests/integration/test_audio_integration.py
git add docs/audio/MIGRATION_STATUS.md

# Commit with comprehensive message
git commit -m "migration: Phase 2 integration complete

Wire up audio feature into sanzaru server:

Changes:
- server.py: Conditional audio tool registration with _check_audio_available()
- config.py: Added 'audio' path type for AUDIO_FILES_PATH support
- CLAUDE.md: Added audio feature documentation and usage patterns
- tests/integration/test_audio_integration.py: Integration tests for audio
- docs/audio/MIGRATION_STATUS.md: Updated with completion status

Audio tools registered:
✓ list_audio_files, get_latest_audio
✓ convert_audio, compress_audio
✓ transcribe_audio, chat_with_audio, transcribe_with_enhancement
✓ create_audio

Validation:
✓ All tests pass (unit, integration, audio)
✓ Type checking passes
✓ Linting passes
✓ Installation scenarios verified (base, audio, all)
✓ Graceful degradation when audio deps missing

The audio feature is now fully integrated and ready for use!

Source: mcp-server-whisper v1.1.0 by Richie Caputo (MIT license)
Phase: 2 (Integration)
"

# Verify commit
git log -1 --stat
```

## Merge to Main

Once everything is validated:

```bash
cd /Users/rcaputo3/git/sanzaru

# Ensure on feature branch
git checkout migration/audio-feature

# Final check
git log --oneline --graph | head -30

# Merge to main
git checkout main
git merge migration/audio-feature --no-ff -m "feat: Add audio processing feature (Whisper, GPT-4o, TTS)

Integrate comprehensive audio capabilities from mcp-server-whisper v1.1.0:

Features:
- Whisper and GPT-4o transcription
- Interactive audio chat with GPT-4o
- Text-to-speech generation
- Audio format conversion and compression
- Advanced file filtering and management

Architecture:
- Optional dependency: install with sanzaru[audio]
- Graceful degradation when deps not available
- Fully async, type-safe, tested
- Follows sanzaru patterns and conventions

Attribution:
Incorporates code from mcp-server-whisper by Richie Caputo (MIT license)
https://github.com/arcaputo3/mcp-server-whisper

Migration completed via 4-track parallel workflow:
- Track A: Domain logic (models, processor, filters)
- Track B: Services (audio, file, transcription, TTS)
- Track C: Infrastructure and tools
- Track D: Tests and documentation

Co-authored-by: Richie Caputo <rcaputo3@tjclp.com>
"

# Push to remote
git push origin main

# Clean up feature branches (optional)
git branch -d migration/audio-feature
git branch -d migration/track-a-audio-domain
git branch -d migration/track-b-audio-services
git branch -d migration/track-c-infrastructure
git branch -d migration/track-d-tests-docs
```

## Success Criteria

- [ ] `config.py` supports `audio` path type
- [ ] `server.py` conditionally registers audio tools
- [ ] Audio feature detection works (`_check_audio_available()`)
- [ ] All 8 audio tools registered
- [ ] `CLAUDE.md` documents audio feature
- [ ] Integration test created
- [ ] Full test suite passes
- [ ] Type checking passes
- [ ] Linting passes
- [ ] All installation scenarios verified
- [ ] Migration status updated to complete
- [ ] Changes committed
- [ ] Merged to main
- [ ] Feature branches cleaned up

## Post-Integration Tasks

After merging to main:

1. **Tag release:**
   ```bash
   git tag v0.2.0 -m "Release v0.2.0: Audio feature"
   git push origin v0.2.0
   ```

2. **Update changelog:**
   ```bash
   echo "## v0.2.0 - Audio Feature Release

   ### Added
   - Audio transcription with Whisper and GPT-4o
   - Interactive audio chat with GPT-4o
   - Text-to-speech generation
   - Audio format conversion and compression
   - Optional audio dependency group

   ### Attribution
   Incorporates mcp-server-whisper v1.1.0 by Richie Caputo (MIT)
   " >> CHANGELOG.md
   ```

3. **Test end-to-end:**
   ```bash
   # Clean install and test
   cd /tmp
   git clone /Users/rcaputo3/git/sanzaru sanzaru-test
   cd sanzaru-test
   uv sync --extra audio
   export OPENAI_API_KEY=sk-...
   export AUDIO_FILES_PATH=/path/to/audio
   uv run sanzaru
   ```

4. **Update documentation site** (if applicable)

5. **Announce release:**
   - Update README badges
   - Post to relevant communities
   - Update MCP server listings

## Time Estimate

- Task 1: ~2 minutes (config.py)
- Task 2: ~4 minutes (server.py)
- Task 3: ~2 minutes (CLAUDE.md)
- Task 4: ~1 minute (integration test)
- Task 5: ~3 minutes (test suite)
- Task 6: ~2 minutes (installation scenarios)
- Task 7: ~1 minute (migration status)
- Validation & commit: ~2 minutes
- Merge to main: ~1 minute

**Total: ~10 minutes**

## Common Issues

### Issue: Type Errors in server.py

```python
# If mypy complains about tool signatures:
# Ensure Literal types are imported
# Match signatures exactly with tools/audio.py
```

### Issue: Tests Fail

```bash
# Check which tests fail:
pytest tests/ -v --tb=short

# Common fixes:
# - Import paths incorrect
# - Missing fixtures
# - Audio deps not installed (expected for audio tests)
```

### Issue: Circular Imports

```python
# If you get circular import errors:
# Check import order in server.py
# Audio tools should import AFTER other tools
```

## Notes

- This is the final integration step!
- Take time to test thoroughly
- Validate all installation scenarios
- The migration should feel seamless to users
- Audio feature is optional and gracefully degrades

## Celebration! 🎉

**You've successfully migrated mcp-server-whisper into sanzaru!**

The audio feature is now:
- ✅ Fully integrated
- ✅ Optional and modular
- ✅ Well-tested and documented
- ✅ Ready for production use

**Total migration time:** ~40 minutes (with parallel execution)
**Code quality:** Maintained at highest standards
**Attribution:** Properly credited
**Architecture:** Clean and extensible

Great work! 🚀
