# Open Source Release Specification

## Overview

Production-ready open source release for sanzaru with automated CI/CD, PyPI publishing, dependency security scanning, and quality assurance.

**Author & Maintainer:** Richie Caputo (rcaputo3@tjclp.com)
**Organization:** TJC Data & AI
**Target:** Release 1.0.0 to PyPI as a public, production-ready MCP server

**⚠️ Pending Decision:** Package naming convention
- Current: `sanzaru`
- Considerations: `mcp-server-sora` (MCP convention) vs `mcp-server-video` (generic/safe)
- Status: Awaiting OpenAI partnership feedback on trademark usage
- Timeline: Decision required before Phase 2 (PyPI publishing)

## Current State

**✅ Assets:**
- MIT License
- Pre-commit hooks (ruff, mypy, pytest)
- GitHub Actions (Claude Code integration)
- PyPI packaging structure (pyproject.toml)
- Comprehensive test suite (91 tests, 65%+ coverage)
- Documentation (README, CLAUDE.md, async-optimizations.md)
- Security utilities (path traversal protection, symlink checking)

**❌ Gaps:**
- No CI/CD pipeline for automated testing
- No PyPI publishing automation
- No dependency security scanning (`safety`, `pip-audit`)
- No version management strategy
- Missing community standards (CONTRIBUTING.md, CODE_OF_CONDUCT.md)
- No issue/PR templates
- No changelog

## Roadmap

### Phase 1: CI/CD Foundation 🔴 HIGH PRIORITY

**Goal:** Automate testing, linting, and security checks on every commit/PR

#### 1.1 Automated Test Pipeline

**File:** `.github/workflows/tests.yml`

**Triggers:**
- `push` to `main` branch
- `pull_request` to `main` branch

**Matrix Strategy:**
- Python versions: 3.10, 3.11, 3.12, 3.14
- OS: ubuntu-latest (add macOS/Windows later if needed)

**Steps:**
1. Checkout code
2. Setup Python with `uv`
3. Install dependencies: `uv sync`
4. Run pytest: `uv run pytest --cov=src --cov-report=xml --cov-report=term`
5. Upload coverage to Codecov
6. Require ≥65% coverage to pass

**Success Criteria:** All tests pass on all Python versions

---

#### 1.2 Linting & Type Checking Pipeline

**File:** `.github/workflows/lint.yml`

**Triggers:** Same as tests

**Steps:**
1. Checkout code
2. Setup Python
3. Install dev dependencies
4. Run ruff check: `uv run ruff check .`
5. Run ruff format check: `uv run ruff format --check .`
6. Run mypy: `uv run mypy src/`

**Success Criteria:** No linting errors, no type errors

---

#### 1.3 Dependency Security Scanning

**File:** `.github/workflows/security.yml`

**Triggers:**
- `pull_request`
- `schedule`: weekly (Monday 9am UTC)
- `workflow_dispatch` (manual trigger)

**Tools:**
- **`safety`**: Checks dependencies against CVE database
- **`pip-audit`**: PyPA's official vulnerability scanner
- **`bandit`** (optional): Python code security linter

**Steps:**
1. Checkout code
2. Setup Python
3. Install dependencies
4. Run `uv run safety check`
5. Run `uv run pip-audit`
6. Report vulnerabilities as issues or PR comments

**Success Criteria:** No high/critical vulnerabilities in dependencies

**Configuration:**
- Fail on: HIGH, CRITICAL
- Warn on: MEDIUM
- Ignore: LOW (unless fixing is trivial)
- Allow exceptions via `.safety-policy.yml` for false positives

---

### Phase 2: PyPI Publishing 🔴 HIGH PRIORITY

**Goal:** Automated, secure publishing to PyPI with version management

#### 2.1 Package Metadata Updates

**File:** `pyproject.toml`

Add missing metadata fields:

```toml
[project]
name = "sanzaru"
version = "1.0.0"  # Bump from 0.1.0
description = "Fully async FastMCP server for OpenAI Sora Video API"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Richie Caputo", email = "rcaputo3@tjclp.com"}
]
maintainers = [
    {name = "Richie Caputo", email = "rcaputo3@tjclp.com"}
]
keywords = ["mcp", "sora", "openai", "video", "ai", "async"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Framework :: AsyncIO",
]

[project.urls]
Homepage = "https://github.com/TJC-LP/sanzaru"
Documentation = "https://github.com/TJC-LP/sanzaru/blob/main/README.md"
Repository = "https://github.com/TJC-LP/sanzaru"
Issues = "https://github.com/TJC-LP/sanzaru/issues"
Changelog = "https://github.com/TJC-LP/sanzaru/blob/main/CHANGELOG.md"
```

---

#### 2.2 Version Management Strategy

**Approach:** Semantic Versioning (SemVer 2.0.0)

**Version Format:** `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`

**Examples:**
- `1.0.0` - First stable release
- `1.0.1` - Patch (bug fixes)
- `1.1.0` - Minor (new features, backward compatible)
- `2.0.0` - Major (breaking changes)
- `1.0.0-rc1` - Release candidate
- `1.0.0-beta.1` - Beta release

**Versioning Rules:**
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, no API changes

**Version Source:** Git tags (`v1.0.0`, `v1.1.0`, etc.)

---

#### 2.3 PyPI Publishing Workflow

**File:** `.github/workflows/publish.yml`

**Trigger:** Git tag push (`v*.*.*`)

**Requirements:**
- Tag format: `v1.0.0` (must match regex `^v[0-9]+\.[0-9]+\.[0-9]+`)
- All tests must pass
- Security scans must pass
- Linting must pass

**Steps:**
1. Checkout code (with tags)
2. Validate tag format
3. Extract version from tag
4. Run full test suite
5. Run security scans
6. Build distributions:
   - `uv build` (creates wheel + sdist)
7. Publish to Test PyPI:
   - `uv publish --repository testpypi`
8. Run smoke test from Test PyPI
9. Publish to PyPI:
   - `uv publish`
10. Create GitHub Release with auto-generated notes

**Security:** Use PyPI Trusted Publishing (OIDC)
- No API tokens stored
- GitHub Actions identity verified by PyPI
- Configure at: https://pypi.org/manage/account/publishing/

**Rollback Plan:**
- If publish fails, delete tag: `git tag -d v1.0.0 && git push origin :refs/tags/v1.0.0`
- PyPI doesn't allow re-uploading same version
- Fix issues and create new patch version

---

#### 2.4 Release Automation

**File:** `.github/workflows/release.yml`

**Trigger:** Successful PyPI publish

**Steps:**
1. Generate changelog from commits since last tag
2. Create GitHub Release
3. Attach distribution artifacts (wheel, sdist)
4. Post announcement (optional: Discord, Twitter, etc.)

**Changelog Generation:**
- Use conventional commits format
- Group by: Features, Fixes, Breaking Changes, Documentation
- Include contributor list
- Link to PyPI package

---

### Phase 3: Dependency Security 🟡 MEDIUM PRIORITY

**Goal:** Continuous dependency vulnerability monitoring

#### 3.1 Add Security Tools

**Dependencies to add (dev group):**

```toml
[dependency-groups]
dev = [
  # ... existing deps ...
  "safety>=3.0.0",
  "pip-audit>=2.6.0",
  "bandit>=1.7.0",  # optional
]
```

#### 3.2 Security Policy File

**File:** `.safety-policy.yml`

```yaml
security:
  ignore-vulnerabilities:
    # Example: ignore specific CVE with justification
    # 12345:
    #   reason: False positive - not applicable to our usage
    #   expires: '2025-12-31'

  continue-on-vulnerability-error: false
```

#### 3.3 Dependabot Configuration

**File:** `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "automated"
```

**Benefits:**
- Auto-update dependencies weekly
- Security patches applied automatically
- Grouped updates to reduce PR noise

---

### Phase 4: Community Standards 🟡 MEDIUM PRIORITY

**Goal:** Establish contributor-friendly processes

#### 4.1 Contributing Guidelines

**File:** `CONTRIBUTING.md`

**Contents:**
- Development setup (use `setup.sh`)
- Running tests (`pytest`)
- Code style (ruff, mypy, pre-commit)
- Commit message conventions (conventional commits)
- PR submission process
- Review process
- Code of conduct reference

**Key Points:**
- Require tests for new features
- Require documentation updates
- Run pre-commit hooks before PR
- Link to CLAUDE.md for architecture guidance

---

#### 4.2 GitHub Issue Templates

**Files:**
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/ISSUE_TEMPLATE/question.md`

**Bug Report Template:**
- Environment (Python version, OS, package version)
- Steps to reproduce
- Expected vs actual behavior
- Minimal reproducible example
- Logs/error messages

**Feature Request Template:**
- Use case description
- Proposed solution
- Alternatives considered
- Implementation complexity estimate

---

#### 4.3 Pull Request Template

**File:** `.github/PULL_REQUEST_TEMPLATE.md`

**Sections:**
- **Description:** What changes and why
- **Type:** Feature / Bug Fix / Documentation / Refactor
- **Breaking Changes:** Yes/No, details if yes
- **Testing:** How was this tested
- **Checklist:**
  - [ ] Tests added/updated
  - [ ] Documentation updated
  - [ ] Pre-commit hooks pass
  - [ ] CHANGELOG.md updated (if applicable)

---

#### 4.4 Code of Conduct

**File:** `CODE_OF_CONDUCT.md`

Use **Contributor Covenant v2.1** (industry standard)

**Key Elements:**
- Expected behavior
- Unacceptable behavior
- Reporting process
- Enforcement guidelines

---

### Phase 5: Documentation & Polish 🟢 LOW PRIORITY

**Goal:** Professional, complete documentation

#### 5.1 Changelog

**File:** `CHANGELOG.md`

Follow **Keep a Changelog** format:

```markdown
# Changelog

## [1.0.0] - 2025-XX-XX

### Added
- Fully async architecture with aiofiles and anyio
- 8-10x performance improvements for concurrent operations
- Comprehensive async documentation
- Stress test validation (32 concurrent operations)

### Changed
- Refactored all I/O operations to be non-blocking
- Updated README with performance highlights

### Fixed
- Various bug fixes and improvements

## [0.1.0] - 2025-XX-XX

### Added
- Initial release
- Video generation via Sora API
- Image generation via Responses API
- Reference image management
```

---

#### 5.2 README Badges

Add to top of README:

```markdown
# sanzaru

[![PyPI version](https://badge.fury.io/py/sanzaru.svg)](https://pypi.org/project/sanzaru/)
[![Python Support](https://img.shields.io/pypi/pyversions/sanzaru.svg)](https://pypi.org/project/sanzaru/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/TJC-LP/sanzaru/actions/workflows/tests.yml/badge.svg)](https://github.com/TJC-LP/sanzaru/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/TJC-LP/sanzaru/branch/main/graph/badge.svg)](https://codecov.io/gh/TJC-LP/sanzaru)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
```

---

#### 5.3 API Documentation (Optional)

**Future Enhancement:** Generate API docs with Sphinx or mkdocs

**Contents:**
- Full API reference (all MCP tools)
- Usage examples
- Architecture diagrams
- Performance benchmarks

**Hosting:** GitHub Pages or Read the Docs

---

## Implementation Checklist

### Phase 1: CI/CD Foundation ⏱️ Week 1-2
- [ ] Create `.github/workflows/tests.yml`
- [ ] Create `.github/workflows/lint.yml`
- [ ] Create `.github/workflows/security.yml`
- [ ] Set up Codecov account and integration
- [ ] Add `safety` and `pip-audit` to dev dependencies
- [ ] Create `.safety-policy.yml`
- [ ] Test all workflows on feature branch
- [ ] Add workflow status badges to README

### Phase 2: PyPI Publishing ⏱️ Week 2-3
- [ ] Update `pyproject.toml` with full metadata
- [ ] Create `.github/workflows/publish.yml`
- [ ] Create `.github/workflows/release.yml`
- [ ] Set up PyPI Trusted Publishing
- [ ] Create Test PyPI account (if needed)
- [ ] Test publish to Test PyPI
- [ ] Tag release candidate `v1.0.0-rc1`
- [ ] Validate RC1 installation from Test PyPI
- [ ] Tag final release `v1.0.0`
- [ ] Publish to PyPI
- [ ] Verify installation: `uv pip install sanzaru`

### Phase 3: Dependency Security ⏱️ Week 3
- [ ] Create `.github/dependabot.yml`
- [ ] Run initial `safety check`
- [ ] Address any existing vulnerabilities
- [ ] Set up weekly security scan schedule
- [ ] Configure GitHub security alerts
- [ ] Document security policy in SECURITY.md

### Phase 4: Community Standards ⏱️ Week 4
- [ ] Write `CONTRIBUTING.md`
- [ ] Create `.github/ISSUE_TEMPLATE/bug_report.md`
- [ ] Create `.github/ISSUE_TEMPLATE/feature_request.md`
- [ ] Create `.github/ISSUE_TEMPLATE/question.md`
- [ ] Create `.github/PULL_REQUEST_TEMPLATE.md`
- [ ] Add `CODE_OF_CONDUCT.md`
- [ ] Update README with "Contributing" section
- [ ] Enable GitHub Discussions (optional)

### Phase 5: Documentation ⏱️ Ongoing
- [ ] Create `CHANGELOG.md` with full history
- [ ] Add PyPI and status badges to README
- [ ] Write migration guide (0.x → 1.0)
- [ ] Add architecture diagrams to docs
- [ ] Set up GitHub Pages for docs (optional)
- [ ] Create video tutorial or demo (optional)

---

## Release 1.0.0 Requirements

**Must Have (Blocking):**
- ✅ All CI/CD workflows passing
- ✅ Published to PyPI
- ✅ No HIGH/CRITICAL security vulnerabilities
- ✅ Test coverage ≥65%
- ✅ All tests passing on Python 3.10-3.14
- ✅ CONTRIBUTING.md exists
- ✅ CODE_OF_CONDUCT.md exists
- ✅ CHANGELOG.md exists

**Should Have (Non-blocking):**
- ✅ GitHub issue templates
- ✅ PR template
- ✅ Dependabot configured
- ✅ README badges
- ✅ Documentation up to date

**Nice to Have:**
- API documentation site
- GitHub Discussions enabled
- Video tutorial
- Architecture diagrams

---

## Success Metrics

**Adoption:**
- PyPI downloads: >100/month after 3 months
- GitHub stars: >50 after 6 months
- Active contributors: ≥3 people

**Quality:**
- Test coverage maintained ≥65%
- Zero HIGH/CRITICAL security issues
- <48h response time to issues
- Monthly dependency updates

**Community:**
- ≥5 external contributions (PRs/issues)
- Positive feedback in discussions
- Used in at least 3 public projects

---

## Risk Mitigation

### API Key Security
**Risk:** Leaked keys in CI/CD

**Mitigation:**
- Use GitHub Secrets for all keys
- Separate keys for CI vs production
- Rotate keys quarterly
- Monitor API usage for anomalies
- Never commit keys to git (pre-commit hook blocks)

### Breaking Changes
**Risk:** Users break on updates

**Mitigation:**
- Follow SemVer strictly
- Deprecate features before removal (1 major version)
- Document all breaking changes in CHANGELOG
- Provide migration guides
- Test against popular MCP clients before release

### Dependency Vulnerabilities
**Risk:** Vulnerable dependencies

**Mitigation:**
- Weekly Dependabot updates
- Weekly `safety` scans
- Auto-merge minor/patch updates
- Quick response to security alerts (<24h)

### Supply Chain Attacks
**Risk:** Malicious dependencies

**Mitigation:**
- Pin exact versions in uv.lock
- Review all dependency updates
- Use PyPI Trusted Publishing (no tokens)
- Enable GitHub security alerts
- Regular security audits

---

## Timeline

**Week 1-2:** CI/CD Foundation
**Week 3:** PyPI Publishing
**Week 4:** Security & Community
**Week 5:** Documentation & Polish
**Week 6:** Release 1.0.0 🎉

**Total:** ~6 weeks to production-ready OSS release

---

## Next Steps

1. ✅ Create this specification document
2. Start Phase 1: Create `tests.yml` workflow
3. Set up project board to track progress
4. Assign phase owners
5. Schedule weekly sync to review progress
