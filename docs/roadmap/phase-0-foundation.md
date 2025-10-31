# Phase 0: Foundation Setup

## Agent Role

**You are the Foundation Agent.** Your job is to create the scaffolding for the audio feature migration before parallel work begins.

**Duration:** ~5 minutes
**Dependencies:** None
**Next Phase:** All Phase 1 tracks depend on your work

## Objective

Create the directory structure and package configuration skeleton that all parallel migration tracks will build upon. This must be done **first** before any parallel work can start.

## Prerequisites

- [ ] Working directory: `/Users/rcaputo3/git/sanzaru`
- [ ] Clean git status (commit or stash changes)
- [ ] Read `audio-migration-feature-spec.md` for context
- [ ] Have access to `/Users/rcaputo3/git/mcp-server-whisper` for reference

## Tasks

### Task 1: Create Audio Feature Directory Structure

Create the following directory tree:

```bash
cd /Users/rcaputo3/git/sanzaru

# Create audio feature directories
mkdir -p src/sanzaru/audio/services
mkdir -p src/sanzaru/infrastructure

# Create test directories
mkdir -p tests/audio

# Create documentation directories
mkdir -p docs/audio
```

**Verification:**
```bash
ls -la src/sanzaru/audio/
ls -la src/sanzaru/audio/services/
ls -la src/sanzaru/infrastructure/
ls -la tests/audio/
ls -la docs/audio/
```

### Task 2: Create Placeholder `__init__.py` Files

Create empty `__init__.py` files to mark directories as Python packages:

```bash
# Audio package markers
touch src/sanzaru/audio/__init__.py
touch src/sanzaru/audio/services/__init__.py
touch src/sanzaru/infrastructure/__init__.py
```

Add a comment to each indicating which track will populate it:

**`src/sanzaru/audio/__init__.py`:**
```python
"""Audio feature for sanzaru - Whisper transcription, GPT-4o audio chat, and TTS.

This module will be populated by Track A (audio domain logic).
"""
```

**`src/sanzaru/audio/services/__init__.py`:**
```python
"""Audio services - transcription, file management, TTS.

This module will be populated by Track B (audio services).
"""
```

**`src/sanzaru/infrastructure/__init__.py`:**
```python
"""Shared infrastructure for sanzaru - OpenAI client, caching, file system.

This module will be populated by Track C (infrastructure).
"""
```

**Verification:**
```bash
cat src/sanzaru/audio/__init__.py
cat src/sanzaru/audio/services/__init__.py
cat src/sanzaru/infrastructure/__init__.py
```

### Task 3: Update `pyproject.toml` with Optional Dependencies

Add the audio optional dependency group to `pyproject.toml`.

**Current state:** The project has `[project]` with `dependencies` list.

**Add this section after the main `dependencies` list:**

```toml
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
all = [
    "sanzaru[audio,image]",
]
```

**Note:** Move `pillow` from main `dependencies` to `image` optional group since it's only needed for image processing.

**Updated main dependencies should look like:**
```toml
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
```

**Add pydantic if not present** (needed for audio models).

**Verification:**
```bash
cat pyproject.toml | grep -A 15 "optional-dependencies"
```

### Task 4: Update `pytest.ini_options` with Audio Marker

Add the `audio` pytest marker to `pyproject.toml`:

**Find the existing `[tool.pytest.ini_options]` section and update markers:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
markers = [
  "unit: Unit tests for pure functions (no external dependencies)",
  "integration: Integration tests with mocked external APIs",
  "audio: Tests for audio feature (requires audio dependencies)",
]
```

**Verification:**
```bash
cat pyproject.toml | grep -A 10 "pytest.ini_options"
```

### Task 5: Update Project Version and Description

Update the version number to reflect this major feature addition:

**In `pyproject.toml`:**
```toml
[project]
name = "sanzaru"
version = "0.2.0"  # Changed from 0.1.0
description = "Unified MCP server for OpenAI multimodal APIs (Sora, Whisper, GPT Vision)"
```

**Also update in `src/sanzaru/__init__.py` if version is defined there.**

**Verification:**
```bash
grep "version" pyproject.toml
grep "description" pyproject.toml
```

### Task 6: Create Migration Tracking Document

Create a file to track migration progress:

**`docs/audio/MIGRATION_STATUS.md`:**
```markdown
# Audio Feature Migration Status

**Start Date:** [Today's date]
**Source:** mcp-server-whisper v1.1.0
**Target:** sanzaru v0.2.0

## Progress

- [ ] Phase 0: Foundation (this file created means in progress!)
- [ ] Phase 1: Parallel Migration
  - [ ] Track A: Audio Domain Logic
  - [ ] Track B: Audio Services
  - [ ] Track C: Infrastructure & Tools
  - [ ] Track D: Tests & Documentation
- [ ] Phase 2: Integration

## Notes

Foundation scaffolding completed by Phase 0 agent.

## Attribution

This migration incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) by Richie Caputo, licensed under MIT.
```

**Verification:**
```bash
cat docs/audio/MIGRATION_STATUS.md
```

## Self-Validation

Before committing, verify everything is in place:

```bash
# Check directory structure
ls -la src/sanzaru/audio/
ls -la src/sanzaru/audio/services/
ls -la src/sanzaru/infrastructure/
ls -la tests/audio/
ls -la docs/audio/

# Check __init__.py files exist
test -f src/sanzaru/audio/__init__.py && echo "✓ audio __init__.py"
test -f src/sanzaru/audio/services/__init__.py && echo "✓ services __init__.py"
test -f src/sanzaru/infrastructure/__init__.py && echo "✓ infrastructure __init__.py"

# Check pyproject.toml changes
grep "optional-dependencies" pyproject.toml || echo "ERROR: optional-dependencies not found"
grep "audio" pyproject.toml || echo "ERROR: audio extra not found"
grep "version = \"0.2.0\"" pyproject.toml || echo "ERROR: version not updated"

# Check migration status doc
test -f docs/audio/MIGRATION_STATUS.md && echo "✓ migration status created"

# Verify Python syntax (should have no errors)
python -c "import ast; ast.parse(open('src/sanzaru/audio/__init__.py').read())"
python -c "import ast; ast.parse(open('src/sanzaru/audio/services/__init__.py').read())"
python -c "import ast; ast.parse(open('src/sanzaru/infrastructure/__init__.py').read())"
```

All checks should pass!

## Git Commit

Create a feature branch and commit the foundation:

```bash
cd /Users/rcaputo3/git/sanzaru

# Create feature branch
git checkout -b migration/audio-feature

# Stage all changes
git add src/sanzaru/audio/
git add src/sanzaru/infrastructure/
git add tests/audio/
git add docs/audio/
git add pyproject.toml

# Commit with descriptive message
git commit -m "migration: audio feature foundation scaffolding

- Create audio/ directory structure for domain logic and services
- Create infrastructure/ for shared components
- Add audio optional dependency group to pyproject.toml
- Move pillow to image optional dependency
- Update version to 0.2.0 for major feature addition
- Add pytest marker for audio tests
- Create migration tracking document

This foundation enables 4 parallel migration tracks:
- Track A: Audio domain logic (models, processor, filters)
- Track B: Audio services (transcription, file mgmt, TTS)
- Track C: Infrastructure and MCP tools
- Track D: Tests and documentation

Source: mcp-server-whisper v1.1.0 by Richie Caputo (MIT license)
"

# Verify commit
git log -1 --stat
git show --name-only
```

## Success Criteria

- [ ] All directories created
- [ ] All `__init__.py` files present with comments
- [ ] `pyproject.toml` has `[project.optional-dependencies]` section
- [ ] `audio`, `image`, and `all` extras defined
- [ ] `pillow` moved to image extras
- [ ] `pydantic` and `pydantic-settings` in main dependencies
- [ ] Version updated to 0.2.0
- [ ] Description updated to mention multimodal
- [ ] Pytest marker for `audio` added
- [ ] Migration status doc created
- [ ] All self-validation checks pass
- [ ] Committed to `migration/audio-feature` branch
- [ ] Commit message is descriptive

## Handoff to Phase 1

Once your commit is complete, Phase 1 parallel tracks can begin:

1. **Track A** can checkout `migration/audio-feature` and branch to `migration/track-a-audio-domain`
2. **Track B** can checkout `migration/audio-feature` and branch to `migration/track-b-audio-services`
3. **Track C** can checkout `migration/audio-feature` and branch to `migration/track-c-infrastructure`
4. **Track D** can checkout `migration/audio-feature` and branch to `migration/track-d-tests-docs`

All tracks work independently with **zero dependencies** on each other.

## Time Estimate

- Task 1-2: ~2 minutes (directory and file creation)
- Task 3-5: ~2 minutes (pyproject.toml edits)
- Task 6: ~1 minute (migration status doc)
- Validation & commit: ~1 minute

**Total: ~5 minutes**

## Notes

- Keep this phase lightweight - just scaffolding
- Don't copy any actual code yet (that's Phase 1)
- The goal is to create empty containers for parallel tracks to fill
- Clear comments help parallel workers know what goes where

## Questions?

Refer to:
- `audio-migration-feature-spec.md` - Complete technical specification
- `audio-migration-README.md` - Orchestration guide
- `/Users/rcaputo3/git/mcp-server-whisper/pyproject.toml` - Source dependencies reference
