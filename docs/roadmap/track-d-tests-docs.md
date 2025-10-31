# Track D: Tests & Documentation Migration

## Agent Role

**You are Track D Agent** - responsible for migrating tests and documentation from mcp-server-whisper.

**Duration:** ~25 minutes
**Dependencies:** Phase 0 foundation must be complete
**Works in parallel with:** Tracks A, B, C (zero conflicts!)

## Objective

1. Migrate all test files from mcp-server-whisper to `tests/audio/`
2. Migrate documentation from `docs/` to `docs/audio/`
3. Update main README.md with audio feature section
4. Add pytest markers for audio tests

## Prerequisites

- [ ] Phase 0 foundation complete (`tests/audio/` and `docs/audio/` directories exist)
- [ ] Working directory: `/Users/rcaputo3/git/sanzaru`
- [ ] Source available: `/Users/rcaputo3/git/mcp-server-whisper`
- [ ] Feature branch: `migration/audio-feature` exists

## Your Directory Scope

You **only** work in these directories:
- `tests/audio/` (all audio tests)
- `docs/audio/` (audio documentation)
- `README.md` (audio feature section)

**You do NOT touch:**
- `src/sanzaru/audio/` ← Track A's territory
- `src/sanzaru/audio/services/` ← Track B's territory
- `src/sanzaru/infrastructure/` ← Track C's territory
- `src/sanzaru/tools/` ← Track C's territory
- `tests/unit/` or `tests/integration/` (existing tests)

This isolation **prevents merge conflicts**!

## Setup

```bash
cd /Users/rcaputo3/git/sanzaru

# Ensure you're on the foundation branch
git checkout migration/audio-feature

# Create your track branch
git checkout -b migration/track-d-tests-docs

# Verify foundation is in place
ls -la tests/audio/
ls -la docs/audio/
```

## Part 1: Test Migration

### Source Test Files

From `/Users/rcaputo3/git/mcp-server-whisper/tests/`:

List all test files:
```bash
find /Users/rcaputo3/git/mcp-server-whisper/tests -name "test_*.py" -o -name "conftest.py"
```

### Step 1: Migrate conftest.py (if audio-specific)

Check if conftest has audio-specific fixtures:
```bash
cat /Users/rcaputo3/git/mcp-server-whisper/tests/conftest.py
```

**If it has audio-specific fixtures:**
```bash
cp /Users/rcaputo3/git/mcp-server-whisper/tests/conftest.py \
   tests/audio/conftest.py
```

**Adapt imports:**
```python
# Old imports
from mcp_server_whisper.models import AudioInfo

# New imports
from sanzaru.audio.models import AudioInfo
```

**If it only has generic fixtures:**
- Merge useful fixtures into sanzaru's main `tests/conftest.py`
- Or keep a minimal audio-specific conftest

### Step 2: Migrate All Test Files

Copy all test files:
```bash
# Find and copy all test files
cd /Users/rcaputo3/git/mcp-server-whisper
find tests -name "test_*.py" -exec cp {} /Users/rcaputo3/git/sanzaru/tests/audio/ \;

# Return to sanzaru
cd /Users/rcaputo3/git/sanzaru
```

**List of test files to migrate** (based on typical mcp-server-whisper structure):
- `test_audio_processor.py`
- `test_file_filter.py`
- `test_audio_service.py`
- `test_file_service.py`
- `test_transcription_service.py`
- `test_tts_service.py`
- `test_audio_tools.py`
- `test_file_tools.py`
- `test_transcription_tools.py`
- `test_openai_client.py`
- `test_cache.py`
- `test_path_resolver.py`
- `test_file_system.py`
- `test_text_utils.py`
- `test_server.py` (may need adaptation)

### Step 3: Update Test Imports

For **each test file**, update imports:

```python
# Old imports
from mcp_server_whisper.domain.audio_processor import AudioProcessor
from mcp_server_whisper.models.audio import AudioInfo
from mcp_server_whisper.services.audio_service import AudioService

# New imports
from sanzaru.audio.processor import AudioProcessor
from sanzaru.audio.models import AudioInfo
from sanzaru.audio.services import AudioService
```

**Systematic approach:**
```bash
cd tests/audio

# For each test file:
# 1. Open in editor
# 2. Replace 'mcp_server_whisper' → 'sanzaru.audio' or 'sanzaru.infrastructure'
# 3. Update relative imports
# 4. Verify syntax

# Quick check with sed (review before applying):
sed -i.bak 's/from mcp_server_whisper\./from sanzaru./g' test_*.py
sed -i.bak 's/import mcp_server_whisper/import sanzaru/g' test_*.py

# Manual review needed for:
# - from mcp_server_whisper.domain → from sanzaru.audio
# - from mcp_server_whisper.services → from sanzaru.audio.services
# - from mcp_server_whisper.infrastructure → from sanzaru.infrastructure
# - from mcp_server_whisper.models → from sanzaru.audio.models
```

### Step 4: Add Pytest Markers to Audio Tests

Add `@pytest.mark.audio` to each test function:

```python
import pytest

@pytest.mark.audio
def test_transcribe_audio():
    """Test audio transcription."""
    # test code
```

**Or add at module level** (easier):
```python
import pytest

pytestmark = pytest.mark.audio  # Marks all tests in this file

def test_transcribe_audio():
    # All tests in this file are marked as 'audio'
```

Add this line near the top of each test file (after imports).

### Step 5: Verify Test Syntax

```bash
cd /Users/rcaputo3/git/sanzaru

# Syntax check all test files
python -m py_compile tests/audio/*.py

# Check imports (may fail if deps not installed - that's okay)
python -c "import tests.audio.test_audio_processor"
```

## Part 2: Documentation Migration

### Source Documentation Files

From `/Users/rcaputo3/git/mcp-server-whisper/docs/`:

```bash
ls /Users/rcaputo3/git/mcp-server-whisper/docs/
# Typically: architecture.md, mcp-readme.md, mcp-overview.md, openai-audio.md, openai-realtime.md
```

### Step 6: Migrate Documentation Files

Copy all documentation:
```bash
cp /Users/rcaputo3/git/mcp-server-whisper/docs/* \
   docs/audio/
```

### Step 7: Create Audio Feature README

Create `docs/audio/README.md` as the main audio feature guide:

```markdown
# Sanzaru Audio Feature

Audio processing capabilities for sanzaru via OpenAI's Whisper and GPT-4o Audio APIs.

## Installation

```bash
# Install sanzaru with audio support
uv add "sanzaru[audio]"

# Or install all features
uv add "sanzaru[all]"
```

## Configuration

Set the audio files directory:

```bash
export AUDIO_FILES_PATH=/path/to/your/audio/files
export OPENAI_API_KEY=sk-...
```

## Available Tools

### File Management
- `list_audio_files`: List and filter audio files
- `get_latest_audio`: Get most recent audio file

### Audio Processing
- `convert_audio`: Convert between formats (mp3, wav)
- `compress_audio`: Compress oversized files

### Transcription
- `transcribe_audio`: Standard Whisper transcription
- `chat_with_audio`: Interactive audio analysis with GPT-4o
- `transcribe_with_enhancement`: Enhanced transcription with templates

### Text-to-Speech
- `create_audio`: Generate TTS audio

## Supported Formats

**Transcription:** flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
**Audio Chat:** mp3, wav
**TTS Output:** mp3, opus, aac, flac, wav, pcm

## Example Usage

```python
# With Claude Code
claude

# Then in Claude:
"List my audio files and transcribe the latest one with detailed enhancement"
```

## Documentation

- [Architecture](architecture.md) - Technical architecture details
- [OpenAI Audio APIs](openai-audio.md) - API reference and capabilities
- [OpenAI Realtime](openai-realtime.md) - Realtime audio features

## Attribution

This feature incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) v1.1.0 by Richie Caputo (MIT license).
```

### Step 8: Update Main README.md

Add audio feature section to sanzaru's main README.md:

**Find the Features section and add:**

```markdown
## Features

### Video Generation (Sora)
- **Video Generation**: Create Sora jobs (`sora-2` / `sora-2-pro`), optional image reference, optional remix
- Get status, wait until completion (polling), download assets
- List and delete videos
- Remix existing videos

### Image Generation
- **Image Generation**: Create reference images using GPT-5/GPT-4.1 with iterative refinement
- List and prepare reference images
- Reference image management with automatic resizing

### Audio Processing (NEW)
- **Audio Transcription**: Whisper and GPT-4o transcription models
- **Audio Chat**: Interactive audio analysis with GPT-4o
- **Text-to-Speech**: High-quality TTS with multiple voices
- **Audio Processing**: Format conversion, compression, file management
- **Enhanced Transcription**: Specialized templates for detailed, storytelling, professional, or analytical output

> **Note:** Content guardrails are enforced by OpenAI. This server does not run local moderation.
```

**Add installation instructions:**

```markdown
## Installation

### Base Installation (Video + Image)
```bash
uv add sanzaru
```

### With Audio Support
```bash
uv add "sanzaru[audio]"
```

### All Features
```bash
uv add "sanzaru[all]"
```

### Development Installation
```bash
git clone https://github.com/TJC-LP/sanzaru.git
cd sanzaru
uv sync --all-extras
```
```

**Add configuration section:**

```markdown
## Configuration

### Required Environment Variables

**Core:**
- `OPENAI_API_KEY`: Your OpenAI API key

**Video feature:**
- `VIDEO_PATH`: Directory for downloaded videos

**Image feature:**
- `IMAGE_PATH`: Directory for reference images

**Audio feature:**
- `AUDIO_FILES_PATH`: Directory containing audio files

### Quick Setup

```bash
# Run the setup script
./setup.sh

# Or manually create .env
cat > .env << EOF
OPENAI_API_KEY=your_api_key
VIDEO_PATH=/path/to/videos
IMAGE_PATH=/path/to/images
AUDIO_FILES_PATH=/path/to/audio
EOF
```
```

## Self-Validation

Run these checks before committing:

```bash
cd /Users/rcaputo3/git/sanzaru

# 1. Check all test files migrated
ls -la tests/audio/
wc -l tests/audio/test_*.py
# Should have ~12 test files

# 2. Check documentation migrated
ls -la docs/audio/
# Should see: README.md, architecture.md, openai-audio.md, etc.

# 3. Syntax validation for tests
python -m py_compile tests/audio/*.py

# 4. Check no old package references in tests
grep -r "mcp_server_whisper" tests/audio/ && echo "ERROR: Old refs found!" || echo "✓ Clean"

# 5. Check pytest markers added
grep -r "pytest.mark.audio" tests/audio/ || grep -r "pytestmark.*audio" tests/audio/

# 6. Check README updated
grep "Audio Processing" README.md || echo "WARNING: README not updated"
grep "sanzaru\[audio\]" README.md || echo "WARNING: Installation instructions missing"
```

## Git Commit

```bash
cd /Users/rcaputo3/git/sanzaru

# Stage your changes
git add tests/audio/
git add docs/audio/
git add README.md

# Commit with descriptive message
git commit -m "migration: tests and documentation (Track D)

Migrate comprehensive test suite and documentation from mcp-server-whisper:

Tests migrated (tests/audio/):
- 12+ test files covering all audio functionality
- Test coverage for domain, services, infrastructure, tools
- conftest.py with audio-specific fixtures
- All tests marked with @pytest.mark.audio
- All imports updated to sanzaru structure

Documentation migrated (docs/audio/):
- README.md: Audio feature guide
- architecture.md: Technical architecture
- openai-audio.md: API reference
- openai-realtime.md: Realtime features
- Other audio-specific documentation

README.md updated:
- Audio feature section added
- Installation instructions for optional extras
- Configuration guide expanded

All tests have valid syntax.
Ready to run after Phase 2 integration.

Source: mcp-server-whisper v1.1.0 by Richie Caputo (MIT license)
Track: D (Tests & Documentation)
"

# Verify commit
git log -1 --stat
git show --name-status
```

## Success Criteria

- [ ] All test files migrated to `tests/audio/`
- [ ] Test imports updated to sanzaru structure
- [ ] All tests marked with `@pytest.mark.audio`
- [ ] `conftest.py` adapted (if needed)
- [ ] All documentation migrated to `docs/audio/`
- [ ] `docs/audio/README.md` created
- [ ] Main `README.md` updated with audio feature
- [ ] Installation instructions include optional deps
- [ ] All Python test files have valid syntax
- [ ] No references to `mcp_server_whisper` package
- [ ] All self-validation checks pass
- [ ] Committed to `migration/track-d-tests-docs` branch
- [ ] Commit message is descriptive

## Common Issues

### Issue: Test Fixtures Reference Missing Code

```python
# If fixtures reference Track A/B/C code that isn't migrated yet:
# That's okay - tests will be run in Phase 2 after everything is integrated
# Just ensure syntax is valid
```

### Issue: Circular Imports in Tests

```python
# If tests import from each other:
# Reorganize imports to avoid circles
# Usually tests should only import from src/, not from other tests
```

### Issue: Too Many Test Files

```python
# If there are 20+ test files:
# That's fine! Copy them all.
# Phase 2 integration will run the full suite
```

## Time Estimate

- Step 1-2: ~5 minutes (copy test files)
- Step 3-4: ~8 minutes (update imports and markers)
- Step 5: ~1 minute (syntax validation)
- Step 6-7: ~5 minutes (copy docs and create README)
- Step 8: ~4 minutes (update main README)
- Validation & commit: ~2 minutes

**Total: ~25 minutes**

## Notes

- You're working in **complete isolation** in `tests/audio/` and `docs/audio/`
- No conflicts with other tracks
- Don't worry if tests can't run yet (missing imports from other tracks)
- Focus on clean migration and import updates
- Phase 2 will validate everything works together
- Documentation is crucial - make it excellent!

## Questions?

Refer to:
- `audio-migration-feature-spec.md` - Overall architecture
- `/Users/rcaputo3/git/mcp-server-whisper/tests/` - Source tests
- `/Users/rcaputo3/git/mcp-server-whisper/docs/` - Source documentation
- `tests/conftest.py` - Sanzaru's existing test fixtures
- `README.md` - Sanzaru's existing README for style reference
