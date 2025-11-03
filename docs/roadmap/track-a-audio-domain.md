# Track A: Audio Domain Logic Migration

## Agent Role

**You are Track A Agent** - responsible for migrating audio domain models, constants, and core processing logic from mcp-server-whisper.

**Duration:** ~25 minutes
**Dependencies:** Phase 0 foundation must be complete
**Works in parallel with:** Tracks B, C, D (zero conflicts!)

## Objective

Copy and adapt the core audio domain logic from mcp-server-whisper into sanzaru's `audio/` directory. This includes Pydantic models, constants, audio processing, and file filtering logic.

## Prerequisites

- [ ] Phase 0 foundation complete (`src/sanzaru/audio/` directory exists)
- [ ] Working directory: `/Users/rcaputo3/git/sanzaru`
- [ ] Source available: `/Users/rcaputo3/git/mcp-server-whisper`
- [ ] Feature branch: `migration/audio-feature` exists

## Your Directory Scope

You **only** work in these directories:
- `src/sanzaru/audio/` (your main workspace)

**You do NOT touch:**
- `src/sanzaru/audio/services/` ← Track B's territory
- `src/sanzaru/infrastructure/` ← Track C's territory
- `src/sanzaru/tools/` ← Track C's territory
- `tests/` ← Track D's territory
- `docs/` (except MIGRATION_STATUS.md) ← Track D's territory

This isolation **prevents merge conflicts**!

## Setup

```bash
cd /Users/rcaputo3/git/sanzaru

# Ensure you're on the foundation branch
git checkout migration/audio-feature

# Create your track branch
git checkout -b migration/track-a-audio-domain

# Verify foundation is in place
ls -la src/sanzaru/audio/
```

## Source Files to Migrate

From `/Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/`:

### 1. **Constants** → `src/sanzaru/audio/constants.py`
**Source:** `constants.py`
**Contains:** Audio format constants, model names, size limits

### 2. **Models** → `src/sanzaru/audio/models.py`
**Source:** `models/` directory
- `models/base.py`
- `models/audio.py`
- `models/transcription.py`
- `models/responses.py`
- `models/tts.py`
- `models/__init__.py`

**Merge these into a single file** for simplicity (unless they're very large).

### 3. **Audio Processor** → `src/sanzaru/audio/processor.py`
**Source:** `domain/audio_processor.py`
**Contains:** Audio file processing, conversion, compression logic

### 4. **File Filter** → `src/sanzaru/audio/file_filter.py`
**Source:** `domain/file_filter.py`
**Contains:** File filtering and sorting logic

### 5. **Audio Config** → `src/sanzaru/audio/config.py`
**Source:** `config.py` (adapt for sanzaru patterns)
**Contains:** Audio-specific configuration (AUDIO_FILES_PATH validation)

### 6. **Exceptions** (if audio-specific) → `src/sanzaru/audio/exceptions.py`
**Source:** `exceptions.py`
**Only if:** Exceptions are audio-specific; otherwise, use sanzaru's global exceptions

## Migration Steps

### Step 1: Migrate Constants

```bash
# Copy constants file
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/constants.py \
   src/sanzaru/audio/constants.py
```

**Adapt the file:**
- Update docstring to mention sanzaru
- No import changes needed (should be pure constants)

**Verify:**
```bash
python -c "from sanzaru.audio.constants import SUPPORTED_FORMATS_TRANSCRIBE"
```

### Step 2: Migrate Models

Read all model files from source:
```bash
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/models/base.py
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/models/audio.py
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/models/transcription.py
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/models/responses.py
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/models/tts.py
```

**Create:** `src/sanzaru/audio/models.py`

Merge all models into a single comprehensive file:

```python
"""Audio domain models for sanzaru.

This module contains Pydantic models for audio processing, transcription,
text-to-speech, and related functionality.

Migrated from mcp-server-whisper v1.1.0 (MIT license).
"""

from enum import Enum
from pathlib import Path
from typing import Literal, Any

from pydantic import BaseModel, Field

# [Copy all model classes from source files here]
# - Base models from base.py
# - Audio models from audio.py
# - Transcription models from transcription.py
# - Response models from responses.py
# - TTS models from tts.py

__all__ = [
    # Export all model classes
]
```

**Update imports if needed:**
- Change `from ..constants import` → `from .constants import`
- Change `from .base import` → `# (now in same file)`

**Verify:**
```bash
python -c "from sanzaru.audio.models import TranscriptionResult, TTSResult"
```

### Step 3: Migrate Audio Processor

```bash
# Copy processor
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/domain/audio_processor.py \
   src/sanzaru/audio/processor.py
```

**Adapt imports:**

```python
# Old imports
from ..constants import SUPPORTED_FORMATS_TRANSCRIBE
from ..models.audio import AudioInfo

# New imports
from .constants import SUPPORTED_FORMATS_TRANSCRIBE
from .models import AudioInfo
```

**Verify:**
```bash
python -c "from sanzaru.audio.processor import AudioProcessor"
```

### Step 4: Migrate File Filter

```bash
# Copy file filter
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/domain/file_filter.py \
   src/sanzaru/audio/file_filter.py
```

**Adapt imports:**

```python
# Old imports
from ..models.audio import FilePathSupportParams

# New imports
from .models import FilePathSupportParams
```

**Verify:**
```bash
python -c "from sanzaru.audio.file_filter import FileFilter"
```

### Step 5: Create Audio Config

Read the source config:
```bash
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/config.py
```

**Create:** `src/sanzaru/audio/config.py`

**Adapt to sanzaru patterns** (similar to main `config.py`):

```python
"""Audio feature configuration for sanzaru.

Manages AUDIO_FILES_PATH validation and audio-specific settings.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from ..exceptions import ConfigurationError  # Use sanzaru's exceptions


class AudioConfig(BaseSettings):
    """Configuration for audio feature.

    Loads AUDIO_FILES_PATH from environment with validation.
    """

    audio_files_path: Path = Field(
        ...,
        description="Path to the directory containing audio files",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "arbitrary_types_allowed": True,
    }

    @field_validator("audio_files_path")
    @classmethod
    def validate_audio_path(cls, v: Path) -> Path:
        """Validate that the audio path exists and is a directory."""
        resolved_path = v.resolve()
        if not resolved_path.exists():
            raise ConfigurationError(f"Audio path does not exist: {resolved_path}")
        if not resolved_path.is_dir():
            raise ConfigurationError(f"Audio path is not a directory: {resolved_path}")
        return resolved_path


@lru_cache
def get_audio_config() -> AudioConfig:
    """Get the audio configuration (cached singleton).

    Returns:
        AudioConfig: The validated configuration object.

    Raises:
        ConfigurationError: If configuration is invalid or missing.
    """
    try:
        return AudioConfig()  # type: ignore
    except Exception as e:
        raise ConfigurationError(f"Failed to load audio configuration: {e}") from e
```

**Note:** Use `ConfigurationError` from sanzaru's main exceptions, or create if needed.

**Verify:**
```bash
# Will fail without AUDIO_FILES_PATH set (expected)
python -c "from sanzaru.audio.config import get_audio_config"
```

### Step 6: Handle Exceptions

Check if audio needs specific exceptions:
```bash
grep "class.*Error" /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/exceptions.py
```

**If exceptions are generic (ConfigurationError, ValidationError):**
- Create `src/sanzaru/exceptions.py` if it doesn't exist
- Or use existing sanzaru exception patterns

**If exceptions are audio-specific:**
- Create `src/sanzaru/audio/exceptions.py`

**For now, assume generic exceptions are sufficient.** Audio services (Track B) will reference them.

### Step 7: Update `__init__.py`

Update `src/sanzaru/audio/__init__.py` to export key components:

```python
"""Audio feature for sanzaru - Whisper transcription, GPT-4o audio chat, and TTS.

This module provides comprehensive audio processing capabilities via OpenAI APIs.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from .config import AudioConfig, get_audio_config
from .constants import (
    SUPPORTED_FORMATS_TRANSCRIBE,
    SUPPORTED_FORMATS_CHAT,
    MAX_FILE_SIZE_MB,
)
from .file_filter import FileFilter
from .models import (
    AudioInfo,
    FilePathSupportParams,
    TranscriptionResult,
    ChatResult,
    AudioProcessingResult,
    TTSResult,
)
from .processor import AudioProcessor

__all__ = [
    # Config
    "AudioConfig",
    "get_audio_config",
    # Constants
    "SUPPORTED_FORMATS_TRANSCRIBE",
    "SUPPORTED_FORMATS_CHAT",
    "MAX_FILE_SIZE_MB",
    # Core classes
    "FileFilter",
    "AudioProcessor",
    # Models
    "AudioInfo",
    "FilePathSupportParams",
    "TranscriptionResult",
    "ChatResult",
    "AudioProcessingResult",
    "TTSResult",
]

__version__ = "0.2.0"
```

**Adjust exports based on what actually exists in your migrated files.**

## Import Adaptation Patterns

Common import transformations:

| Source Import | Target Import |
|---------------|---------------|
| `from ..constants import X` | `from .constants import X` |
| `from ..models.audio import Y` | `from .models import Y` |
| `from .base import Z` | `# (same file now)` or `from .models import Z` |
| `from ..exceptions import E` | `from ..exceptions import E` (sanzaru root) |
| `from ..config import get_config` | `from .config import get_audio_config` |

## Self-Validation

Run these checks before committing:

```bash
cd /Users/rcaputo3/git/sanzaru

# 1. Check all files created
ls -la src/sanzaru/audio/
# Should see: __init__.py, config.py, constants.py, models.py, processor.py, file_filter.py

# 2. Syntax validation
python -m py_compile src/sanzaru/audio/*.py

# 3. Import validation (may fail if dependencies missing, that's okay)
python -c "from sanzaru.audio import constants"
python -c "from sanzaru.audio import models"
python -c "from sanzaru.audio import processor"
python -c "from sanzaru.audio import file_filter"
python -c "from sanzaru.audio import config"

# 4. Check no references to old package names
grep -r "mcp_server_whisper" src/sanzaru/audio/ && echo "ERROR: Old package name found!" || echo "✓ No old references"

# 5. Check for relative import issues
grep -r "from \\.\\." src/sanzaru/audio/*.py | grep -v "from ..exceptions" && echo "WARNING: Review these imports" || echo "✓ Imports look good"

# 6. Verify __all__ exports
python -c "from sanzaru.audio import AudioProcessor, FileFilter"
```

## Git Commit

```bash
cd /Users/rcaputo3/git/sanzaru

# Stage your changes
git add src/sanzaru/audio/

# Commit with descriptive message
git commit -m "migration: audio domain logic (Track A)

Migrate core audio domain models and processing logic from mcp-server-whisper:

Files added:
- audio/constants.py: Audio format constants and limits
- audio/models.py: Pydantic models (transcription, TTS, audio info)
- audio/processor.py: Audio processing (convert, compress, metadata)
- audio/file_filter.py: File filtering and sorting logic
- audio/config.py: Audio configuration management
- audio/__init__.py: Public API exports

All imports adapted for sanzaru structure.
All type hints and docstrings preserved.

Source: mcp-server-whisper v1.1.0 by Richie Caputo (MIT license)
Track: A (Domain Logic)
"

# Verify commit
git log -1 --stat
git show --name-status
```

## Success Criteria

- [ ] All 6 files created in `src/sanzaru/audio/`
- [ ] `__init__.py` exports key components
- [ ] All Python files have valid syntax
- [ ] No references to `mcp_server_whisper` package
- [ ] Imports updated to use relative imports within `audio/`
- [ ] Docstrings mention sanzaru
- [ ] Migration attribution included
- [ ] All self-validation checks pass
- [ ] Committed to `migration/track-a-audio-domain` branch
- [ ] Commit message is descriptive

## Common Issues

### Issue: Import Error with Pydantic

```python
# If you see: ImportError: cannot import name 'BaseModel' from 'pydantic'
# Ensure pydantic is in pyproject.toml dependencies (Phase 0 should have added it)
```

### Issue: Circular Import

```python
# If models.py imports from processor.py and vice versa:
# 1. Move shared types to models.py
# 2. Have processor.py import from models.py only
# 3. Never import processor in models
```

### Issue: Missing ConfigurationError

```python
# If ConfigurationError doesn't exist in sanzaru:
# Create src/sanzaru/exceptions.py with:

class ConfigurationError(Exception):
    """Configuration validation or loading error."""
    pass
```

## Time Estimate

- Step 1-2: ~5 minutes (constants and models)
- Step 3-4: ~8 minutes (processor and filter)
- Step 5: ~5 minutes (config adaptation)
- Step 6: ~2 minutes (exceptions)
- Step 7: ~2 minutes (__init__.py)
- Validation & commit: ~3 minutes

**Total: ~25 minutes**

## Notes

- You're working in **complete isolation** from other tracks
- No dependencies on Track B, C, or D
- Focus on getting the domain logic migrated cleanly
- Don't worry about how services or tools will use this yet
- Type hints and docstrings are your friends - preserve them!

## Questions?

Refer to:
- `audio-migration-feature-spec.md` - Overall architecture
- `/Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/` - Source code
- `src/sanzaru/types.py` - Sanzaru's type patterns for reference
