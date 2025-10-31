# Track B: Audio Services Migration

## Agent Role

**You are Track B Agent** - responsible for migrating audio service layer logic from mcp-server-whisper.

**Duration:** ~25 minutes
**Dependencies:** Phase 0 foundation must be complete
**Works in parallel with:** Tracks A, C, D (zero conflicts!)

## Objective

Copy and adapt the audio service modules from mcp-server-whisper into sanzaru's `audio/services/` directory. These services handle transcription, file management, TTS generation, and audio processing operations.

## Prerequisites

- [ ] Phase 0 foundation complete (`src/sanzaru/audio/services/` directory exists)
- [ ] Working directory: `/Users/rcaputo3/git/sanzaru`
- [ ] Source available: `/Users/rcaputo3/git/mcp-server-whisper`
- [ ] Feature branch: `migration/audio-feature` exists

## Your Directory Scope

You **only** work in this directory:
- `src/sanzaru/audio/services/` (your exclusive workspace)

**You do NOT touch:**
- `src/sanzaru/audio/` (root files) ← Track A's territory
- `src/sanzaru/infrastructure/` ← Track C's territory
- `src/sanzaru/tools/` ← Track C's territory
- `tests/` ← Track D's territory
- `docs/` (except maybe notes) ← Track D's territory

This isolation **prevents merge conflicts**!

## Setup

```bash
cd /Users/rcaputo3/git/sanzaru

# Ensure you're on the foundation branch
git checkout migration/audio-feature

# Create your track branch
git checkout -b migration/track-b-audio-services

# Verify foundation is in place
ls -la src/sanzaru/audio/services/
```

## Source Files to Migrate

From `/Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/`:

### 1. **Audio Service** → `src/sanzaru/audio/services/audio_service.py`
**Source:** `services/audio_service.py`
**Contains:** Audio processing operations (convert, compress, metadata extraction)

### 2. **File Service** → `src/sanzaru/audio/services/file_service.py`
**Source:** `services/file_service.py`
**Contains:** File listing, searching, filtering operations

### 3. **Transcription Service** → `src/sanzaru/audio/services/transcription_service.py`
**Source:** `services/transcription_service.py`
**Contains:** Whisper and GPT-4o transcription logic

### 4. **TTS Service** → `src/sanzaru/audio/services/tts_service.py`
**Source:** `services/tts_service.py`
**Contains:** Text-to-speech generation logic

### 5. **Services Init** → `src/sanzaru/audio/services/__init__.py`
**Source:** `services/__init__.py`
**Contains:** Service exports

## Migration Steps

### Step 1: Migrate Audio Service

```bash
# Copy audio service
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/audio_service.py \
   src/sanzaru/audio/services/audio_service.py
```

**Adapt imports:**

```python
# Old imports from mcp-server-whisper
from ..domain.audio_processor import AudioProcessor
from ..models.audio import AudioInfo, AudioProcessingResult
from ..infrastructure.path_resolver import PathResolver

# New imports for sanzaru
from ..processor import AudioProcessor
from ..models import AudioInfo, AudioProcessingResult
from ...infrastructure.path_resolver import PathResolver  # Will come from Track C
```

**Note:** If PathResolver doesn't exist yet (Track C's job), you can:
1. Comment out PathResolver usage temporarily
2. Or create a simple placeholder
3. Track C will provide the real implementation

**Verify syntax:**
```bash
python -m py_compile src/sanzaru/audio/services/audio_service.py
```

### Step 2: Migrate File Service

```bash
# Copy file service
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/file_service.py \
   src/sanzaru/audio/services/file_service.py
```

**Adapt imports:**

```python
# Old imports
from ..domain.file_filter import FileFilter
from ..models.audio import FilePathSupportParams
from ..infrastructure.path_resolver import PathResolver

# New imports
from ..file_filter import FileFilter
from ..models import FilePathSupportParams
from ...infrastructure.path_resolver import PathResolver
```

**Verify syntax:**
```bash
python -m py_compile src/sanzaru/audio/services/file_service.py
```

### Step 3: Migrate Transcription Service

```bash
# Copy transcription service
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/transcription_service.py \
   src/sanzaru/audio/services/transcription_service.py
```

**Adapt imports:**

```python
# Old imports
from ..models.transcription import TranscriptionResult, ChatResult
from ..infrastructure.openai_client import get_openai_client
from ..constants import SUPPORTED_FORMATS_TRANSCRIBE

# New imports
from ..models import TranscriptionResult, ChatResult
from ...infrastructure.openai_client import get_openai_client  # Track C provides this
from ..constants import SUPPORTED_FORMATS_TRANSCRIBE
```

**Handle OpenAI client:**
- If `get_openai_client` doesn't exist yet (Track C), you can:
  - Use sanzaru's existing `config.get_client()`
  - Or create a temporary stub
  - Track C will unify this

**Verify syntax:**
```bash
python -m py_compile src/sanzaru/audio/services/transcription_service.py
```

### Step 4: Migrate TTS Service

```bash
# Copy TTS service
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/tts_service.py \
   src/sanzaru/audio/services/tts_service.py
```

**Adapt imports:**

```python
# Old imports
from ..models.tts import TTSResult
from ..infrastructure.openai_client import get_openai_client
from ..utils.text_utils import split_text_into_chunks

# New imports
from ..models import TTSResult
from ...infrastructure.openai_client import get_openai_client
from ...infrastructure.text_utils import split_text_into_chunks  # May need to migrate utils
```

**Handle text utils:**
- If `split_text_into_chunks` is in `utils/text_utils.py` in source
- Track C should migrate this to `infrastructure/`
- For now, you can:
  - Copy the utility function inline
  - Or reference it from Track C's territory with a note

**Verify syntax:**
```bash
python -m py_compile src/sanzaru/audio/services/tts_service.py
```

### Step 5: Update Services `__init__.py`

Update `src/sanzaru/audio/services/__init__.py` to export services:

```python
"""Audio services for sanzaru - transcription, file management, TTS, processing.

This module provides high-level service classes that orchestrate audio operations.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

from .audio_service import AudioService
from .file_service import FileService
from .transcription_service import TranscriptionService
from .tts_service import TTSService

__all__ = [
    "AudioService",
    "FileService",
    "TranscriptionService",
    "TTSService",
]
```

**Adjust based on actual class names in the source files.**

### Step 6: Handle Utility Functions

Check if services depend on utility functions:

```bash
grep -r "from ..utils" /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/
```

**If utilities are found:**

**Option A:** Copy them to `src/sanzaru/audio/utils.py` (simple, self-contained)

```bash
# Check what's in utils
ls /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/utils/
cat /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/utils/text_utils.py
```

If small, copy to your territory:
```bash
cp /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/utils/text_utils.py \
   src/sanzaru/audio/text_utils.py
```

Update imports in services:
```python
from ..text_utils import split_text_into_chunks
```

**Option B:** Let Track C handle it in `infrastructure/`

Add a note in your commit message that Track C should migrate utils.

### Step 7: Handle OpenAI Client References

Services likely use an OpenAI client. Check the pattern:

```bash
grep -n "openai" /Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/*.py
```

**Sanzaru already has** `config.get_client()` which returns `AsyncOpenAI`.

**Update services to use sanzaru's pattern:**

```python
# Instead of:
from ..infrastructure.openai_client import get_openai_client
client = get_openai_client()

# Use sanzaru's:
from ...config import get_client
client = get_client()
```

This makes integration smoother!

## Import Adaptation Patterns

Common import transformations for services:

| Source Import | Target Import |
|---------------|---------------|
| `from ..domain.X import Y` | `from ..X import Y` |
| `from ..models.X import Y` | `from ..models import Y` |
| `from ..infrastructure.X import Y` | `from ...infrastructure.X import Y` |
| `from ..utils.X import Y` | `from ..X import Y` or `from ...infrastructure.X import Y` |
| `from ..config import` | `from ...config import` or use sanzaru's config |
| `from ..exceptions import` | `from ...exceptions import` |

## Self-Validation

Run these checks before committing:

```bash
cd /Users/rcaputo3/git/sanzaru

# 1. Check all service files created
ls -la src/sanzaru/audio/services/
# Should see: __init__.py, audio_service.py, file_service.py, transcription_service.py, tts_service.py

# 2. Syntax validation
python -m py_compile src/sanzaru/audio/services/*.py

# 3. Import validation (may fail if dependencies missing - that's okay for now)
python -c "from sanzaru.audio.services import AudioService, FileService"
python -c "from sanzaru.audio.services import TranscriptionService, TTSService"

# 4. Check no references to old package names
grep -r "mcp_server_whisper" src/sanzaru/audio/services/ && echo "ERROR: Old package name found!" || echo "✓ No old references"

# 5. Check relative imports are correct
grep "from \\.\\.\\." src/sanzaru/audio/services/*.py  # Should go up to sanzaru root
grep "from \\.\\." src/sanzaru/audio/services/*.py    # Should go to audio/

# 6. Verify exports work
python -c "from sanzaru.audio import services"
```

## Git Commit

```bash
cd /Users/rcaputo3/git/sanzaru

# Stage your changes
git add src/sanzaru/audio/services/

# If you copied utils to audio/text_utils.py
git add src/sanzaru/audio/text_utils.py

# Commit with descriptive message
git commit -m "migration: audio services (Track B)

Migrate audio service layer from mcp-server-whisper:

Files added:
- audio/services/audio_service.py: Audio processing operations
- audio/services/file_service.py: File listing and filtering
- audio/services/transcription_service.py: Transcription logic
- audio/services/tts_service.py: Text-to-speech generation
- audio/services/__init__.py: Service exports

All imports adapted for sanzaru structure.
OpenAI client references updated to use sanzaru's config.get_client().
All type hints and async patterns preserved.

Dependencies:
- Imports from audio/ domain (Track A)
- Will integrate with infrastructure/ (Track C) in Phase 2

Source: mcp-server-whisper v1.1.0 by Richie Caputo (MIT license)
Track: B (Services)
"

# Verify commit
git log -1 --stat
git show --name-status
```

## Success Criteria

- [ ] All 4 service files migrated to `audio/services/`
- [ ] `__init__.py` exports all services
- [ ] All Python files have valid syntax
- [ ] No references to `mcp_server_whisper` package
- [ ] Imports updated for sanzaru structure
- [ ] OpenAI client usage consistent with sanzaru patterns
- [ ] Utility functions handled (copied or noted for Track C)
- [ ] Async patterns preserved (await, anyio, etc.)
- [ ] All self-validation checks pass
- [ ] Committed to `migration/track-b-audio-services` branch
- [ ] Commit message is descriptive

## Common Issues

### Issue: Missing PathResolver

```python
# If PathResolver not found:
# Track C will provide it. For now:

# Option 1: Comment out temporarily
# resolver = PathResolver()  # TODO: Track C provides this

# Option 2: Use Path directly
# from pathlib import Path
# resolved = Path(filename).resolve()
```

### Issue: OpenAI Client Mismatch

```python
# If get_openai_client() has different signature than sanzaru's get_client():
# Update to use sanzaru's version:

from ...config import get_client

async def some_service_method():
    client = get_client()
    result = await client.audio.transcriptions.create(...)
```

### Issue: Missing Text Utils

```python
# If split_text_into_chunks not found:
# Copy it inline or to audio/text_utils.py:

def split_text_into_chunks(text: str, max_length: int = 4096) -> list[str]:
    """Split text into chunks of max_length."""
    # [implementation]
```

### Issue: Circular Imports

```python
# If you get circular import errors:
# Check that you're not importing from services/ back to domain
# Services should import FROM domain, never the reverse
```

## Time Estimate

- Step 1: ~5 minutes (audio service)
- Step 2: ~5 minutes (file service)
- Step 3: ~6 minutes (transcription service)
- Step 4: ~5 minutes (TTS service)
- Step 5-6: ~2 minutes (init and utils)
- Step 7: ~3 minutes (client references)
- Validation & commit: ~3 minutes

**Total: ~25 minutes**

## Notes

- You're working in **complete isolation** in `audio/services/`
- No conflicts with Track A (domain), C (infrastructure), or D (tests)
- If you need something from Track C (infrastructure), note it in commit
- Phase 2 integration will wire everything together
- Focus on clean migration of service logic
- Preserve all async patterns - sanzaru is fully async!

## Questions?

Refer to:
- `audio-migration-feature-spec.md` - Overall architecture
- `/Users/rcaputo3/git/mcp-server-whisper/src/mcp_server_whisper/services/` - Source code
- `src/sanzaru/tools/video.py` - Example of how sanzaru services might be called from tools
- `src/sanzaru/config.py` - Sanzaru's OpenAI client pattern
