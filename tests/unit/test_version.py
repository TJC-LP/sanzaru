# SPDX-License-Identifier: MIT
"""Guard against version drift across the repo's version declarations.

`pyproject.toml` is the canonical version. `src/sanzaru/__init__.py` reads it
at runtime via `importlib.metadata`, so the only file that can silently drift
is `plugin/.claude-plugin/plugin.json` (shipped separately as a Claude Code
plugin manifest).
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import sanzaru

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_package_version_matches_pyproject() -> None:
    """`sanzaru.__version__` must match the canonical `pyproject.toml` version."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert sanzaru.__version__ == pyproject["project"]["version"]


@pytest.mark.unit
def test_plugin_manifest_version_matches_pyproject() -> None:
    """`plugin/.claude-plugin/plugin.json` must be bumped in lockstep with pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    manifest = json.loads((REPO_ROOT / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["version"] == pyproject["project"]["version"]
