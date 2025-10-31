# Audio Migration Roadmap: Orchestration Guide

## Overview

This roadmap guides the migration of **mcp-server-whisper** audio capabilities into **sanzaru** as an optional feature module. The migration is designed for maximum parallelization with 4 concurrent work tracks.

**Total Time:** ~40 minutes elapsed (vs ~120 minutes sequential)
**Speedup:** 3x faster with parallel execution

## Documentation Structure

| File | Purpose | Audience |
|------|---------|----------|
| `audio-migration-feature-spec.md` | Complete vision and technical specification | Everyone (read first!) |
| `audio-migration-README.md` | This file - orchestration guide | Migration coordinator |
| `phase-0-foundation.md` | Foundation scaffolding prompt | Foundation agent |
| `track-a-audio-domain.md` | Audio domain logic migration | Agent A |
| `track-b-audio-services.md` | Audio services migration | Agent B |
| `track-c-infrastructure.md` | Infrastructure & tools migration | Agent C |
| `track-d-tests-docs.md` | Tests & documentation migration | Agent D |
| `phase-2-integration.md` | Integration & wiring prompt | Integration agent |

## Prerequisites

1. **Read the feature spec first:** `audio-migration-feature-spec.md`
2. **Clean working directory:** Commit or stash any changes in sanzaru
3. **Access to both repos:**
   - Source: `/Users/rcaputo3/git/mcp-server-whisper`
   - Target: `/Users/rcaputo3/git/sanzaru`
4. **Development tools ready:**
   - `uv` installed
   - `git` configured
   - Python 3.10+

## Migration Timeline

```
┌────────────────────────────────────────────────────────────┐
│ Phase 0: Foundation (Sequential - 5 mins)                  │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ Agent: Create directory structure, update pyproject.toml│ │
│ └────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                            │
                            ├─ git commit "migration: foundation"
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Parallel Migration (4 tracks - 25 mins)           │
│ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌───────┐│
│ │  Track A    │ │  Track B    │ │   Track C    │ │Track D││
│ │   Domain    │ │  Services   │ │Infrastructure│ │ Tests ││
│ │   Logic     │ │             │ │   & Tools    │ │ Docs  ││
│ └─────────────┘ └─────────────┘ └──────────────┘ └───────┘│
│       │               │                 │             │     │
│   branch A        branch B          branch C     branch D  │
└────────────────────────────────────────────────────────────┘
                            │
                   All branches merge
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 2: Integration (Sequential - 10 mins)                │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ Agent: Wire server, config, run tests, update docs    │ │
│ └────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                            │
                            ├─ git commit "migration: complete"
                            ▼
                          DONE
```

## Execution Instructions

### Option 1: Sequential Execution (Simpler, Slower)

Execute each phase one at a time:

```bash
cd /Users/rcaputo3/git/sanzaru

# Phase 0: Foundation
git checkout -b migration/audio-feature
# Follow phase-0-foundation.md instructions
git add . && git commit -m "migration: foundation scaffolding"

# Phase 1: Track A
git checkout -b migration/track-a-audio-domain
# Follow track-a-audio-domain.md instructions
git add . && git commit -m "migration: audio domain logic"
git checkout migration/audio-feature
git merge migration/track-a-audio-domain --no-ff

# Phase 1: Track B
git checkout -b migration/track-b-audio-services
# Follow track-b-audio-services.md instructions
git add . && git commit -m "migration: audio services"
git checkout migration/audio-feature
git merge migration/track-b-audio-services --no-ff

# Phase 1: Track C
git checkout -b migration/track-c-infrastructure
# Follow track-c-infrastructure.md instructions
git add . && git commit -m "migration: infrastructure and tools"
git checkout migration/audio-feature
git merge migration/track-c-infrastructure --no-ff

# Phase 1: Track D
git checkout -b migration/track-d-tests-docs
# Follow track-d-tests-docs.md instructions
git add . && git commit -m "migration: tests and documentation"
git checkout migration/audio-feature
git merge migration/track-d-tests-docs --no-ff

# Phase 2: Integration
# Follow phase-2-integration.md instructions
git add . && git commit -m "migration: integration complete"

# Final merge to main
git checkout main
git merge migration/audio-feature --no-ff
```

**Time: ~45-50 minutes** (includes context switching overhead)

### Option 2: Parallel Execution with Multiple Agents (Faster)

Execute Phase 1 tracks in parallel using 4 concurrent agents/sessions:

**Foundation Setup:**
```bash
cd /Users/rcaputo3/git/sanzaru
git checkout -b migration/audio-feature
# Follow phase-0-foundation.md
git add . && git commit -m "migration: foundation scaffolding"
git push origin migration/audio-feature
```

**Launch 4 Parallel Sessions:**

**Session 1 (Agent A):**
```bash
cd /Users/rcaputo3/git/sanzaru
git checkout migration/audio-feature
git checkout -b migration/track-a-audio-domain
# Follow track-a-audio-domain.md
# Self-validate before committing
git add . && git commit -m "migration: audio domain logic"
git push origin migration/track-a-audio-domain
```

**Session 2 (Agent B):**
```bash
cd /Users/rcaputo3/git/sanzaru
git checkout migration/audio-feature
git checkout -b migration/track-b-audio-services
# Follow track-b-audio-services.md
# Self-validate before committing
git add . && git commit -m "migration: audio services"
git push origin migration/track-b-audio-services
```

**Session 3 (Agent C):**
```bash
cd /Users/rcaputo3/git/sanzaru
git checkout migration/audio-feature
git checkout -b migration/track-c-infrastructure
# Follow track-c-infrastructure.md
# Self-validate before committing
git add . && git commit -m "migration: infrastructure and tools"
git push origin migration/track-c-infrastructure
```

**Session 4 (Agent D):**
```bash
cd /Users/rcaputo3/git/sanzaru
git checkout migration/audio-feature
git checkout -b migration/track-d-tests-docs
# Follow track-d-tests-docs.md
# Self-validate before committing
git add . && git commit -m "migration: tests and documentation"
git push origin migration/track-d-tests-docs
```

**Integration (After All Tracks Complete):**
```bash
cd /Users/rcaputo3/git/sanzaru
git checkout migration/audio-feature

# Merge all tracks (should have ZERO conflicts!)
git merge migration/track-a-audio-domain --no-ff
git merge migration/track-b-audio-services --no-ff
git merge migration/track-c-infrastructure --no-ff
git merge migration/track-d-tests-docs --no-ff

# Follow phase-2-integration.md
git add . && git commit -m "migration: integration complete"

# Merge to main
git checkout main
git merge migration/audio-feature --no-ff
```

**Time: ~40 minutes elapsed**

### Option 3: Claude Code with Multiple Agents

Use Claude Code's agent system:

```bash
cd /Users/rcaputo3/git/sanzaru
claude
```

In Claude Code, launch agents in parallel:

```
Please execute the audio migration using parallel agents:

1. First, execute phase-0-foundation.md synchronously
2. Then launch 4 agents in parallel:
   - Agent A: track-a-audio-domain.md
   - Agent B: track-b-audio-services.md
   - Agent C: track-c-infrastructure.md
   - Agent D: track-d-tests-docs.md
3. After all complete, execute phase-2-integration.md

All prompts are in docs/roadmap/
```

## Dependency Graph

```
Phase 0 (Foundation)
    │
    ├─────────────┬─────────────┬─────────────┐
    │             │             │             │
    ▼             ▼             ▼             ▼
Track A       Track B       Track C       Track D
(Domain)     (Services)   (Infra/Tools)  (Tests/Docs)
    │             │             │             │
    └─────────────┴─────────────┴─────────────┘
                    │
                    ▼
            Phase 2 (Integration)
```

**Key:** No dependencies between Phase 1 tracks = perfect parallelization!

## Merge Conflict Prevention

Each track works in **isolated directories**:

- **Track A:** `src/sanzaru/audio/` (models, constants, processor)
- **Track B:** `src/sanzaru/audio/services/` (services only)
- **Track C:** `src/sanzaru/infrastructure/` + `src/sanzaru/tools/audio.py` + descriptions
- **Track D:** `tests/audio/` + `docs/audio/` + README sections

**Result:** Zero merge conflicts by design!

## Validation Checkpoints

### After Phase 0
```bash
# Verify structure created
ls -la src/sanzaru/audio/
ls -la src/sanzaru/infrastructure/
cat pyproject.toml | grep -A 10 "optional-dependencies"

# Verify it commits cleanly
git status
```

### After Each Phase 1 Track
Each agent should run self-validation (specified in their prompt):

```bash
# Check syntax
python -m py_compile src/sanzaru/audio/**/*.py

# Try imports (will fail if circular dependencies)
python -c "from sanzaru.audio import models"
```

### After Phase 1 Merge
```bash
# Verify all files present
find src/sanzaru/audio -type f | wc -l  # Should have ~15 files
find tests/audio -type f | wc -l        # Should have ~12 files

# Check for import errors
python -c "from sanzaru.audio import models, services"
```

### After Phase 2 Integration
```bash
# Full test suite
pytest tests/unit -m unit
pytest tests/integration
pytest tests/audio -m audio

# Type checking
mypy --strict src/sanzaru/

# Linting
ruff check src/sanzaru/

# Test installation scenarios
uv sync  # Base only
uv sync --extra audio  # With audio
uv sync --all-extras  # Everything
```

## Rollback Procedures

### Rollback Individual Track

```bash
# If Track A has issues
git checkout migration/audio-feature
git branch -D migration/track-a-audio-domain
# Redo Track A from scratch
```

### Rollback Entire Migration

```bash
# Delete feature branch
git checkout main
git branch -D migration/audio-feature

# Delete all track branches
git branch -D migration/track-a-audio-domain
git branch -D migration/track-b-audio-services
git branch -D migration/track-c-infrastructure
git branch -D migration/track-d-tests-docs

# Start over
```

### Rollback After Merge to Main

```bash
# Find the merge commit
git log --oneline --graph

# Reset to before merge
git reset --hard <commit-before-merge>

# Or revert the merge commit
git revert -m 1 <merge-commit-hash>
```

## Success Criteria

### Phase 0 Complete
- [ ] Directory structure exists
- [ ] pyproject.toml updated with optional deps skeleton
- [ ] Placeholder `__init__.py` files created
- [ ] Commits cleanly

### Phase 1 Complete (All Tracks)
- [ ] All source files copied and adapted
- [ ] Imports updated for sanzaru structure
- [ ] Each track passes self-validation
- [ ] No syntax errors
- [ ] All tracks merged without conflicts

### Phase 2 Complete
- [ ] server.py registers audio tools conditionally
- [ ] config.py supports AUDIO_FILES_PATH
- [ ] All tests pass (unit, integration, audio)
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation updated
- [ ] Installation scenarios verified

### Ready for Release
- [ ] Feature spec criteria met (see feature-spec.md)
- [ ] README updated with audio features
- [ ] CLAUDE.md updated with audio patterns
- [ ] Migration attributed properly
- [ ] Version bumped to 0.2.0

## Communication Protocol

### If Running Parallel Sessions

Create a shared status document:

```bash
# Create tracking file
cat > /tmp/migration-status.txt << 'EOF'
Phase 0: [ ] In Progress  [ ] Complete
Track A: [ ] In Progress  [ ] Complete
Track B: [ ] In Progress  [ ] Complete
Track C: [ ] In Progress  [ ] Complete
Track D: [ ] In Progress  [ ] Complete
Phase 2: [ ] Waiting      [ ] In Progress  [ ] Complete
EOF

# Each agent updates their line when done
# Integration agent waits for all tracks before starting
```

### If Running Sequential

No communication needed - just follow the prompts in order.

## Troubleshooting

### Import Errors After Migration

```bash
# Check circular dependencies
python -c "from sanzaru.audio import models"
python -c "from sanzaru.audio import services"

# If circular, review imports in affected files
```

### Test Failures

```bash
# Run specific failing test with verbose output
pytest tests/audio/test_audio_service.py::test_transcribe -vv

# Check if missing dependencies
pip list | grep pydub
```

### Merge Conflicts (Should Not Happen!)

If you somehow get merge conflicts:

```bash
# Check what files conflict
git status

# This means tracks weren't isolated properly
# Resolve manually or rollback and fix track isolation
```

### Missing Files

```bash
# Compare source vs target
find /Users/rcaputo3/git/mcp-server-whisper/src -name "*.py" | wc -l
find /Users/rcaputo3/git/sanzaru/src/sanzaru/audio -name "*.py" | wc -l

# Should be roughly equivalent
```

## Post-Migration Tasks

1. **Test all installation scenarios:**
   ```bash
   uv sync  # Base
   uv sync --extra audio  # With audio
   uv sync --all-extras  # Everything
   ```

2. **Update changelog:**
   ```bash
   echo "## v0.2.0 - Audio Feature Migration" >> CHANGELOG.md
   ```

3. **Tag release:**
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **Announce:**
   - Update README badges
   - Post to relevant communities
   - Update documentation sites

## Questions?

If anything is unclear:
1. Re-read `audio-migration-feature-spec.md`
2. Check the specific agent prompt (track-*.md or phase-*.md)
3. Review sanzaru's existing code patterns in `src/sanzaru/tools/video.py`
4. Check mcp-server-whisper's original implementation

## Attribution

This migration incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) by Richie Caputo, licensed under MIT.
