# SPDX-License-Identifier: MIT
"""Skills over MCP (`io.modelcontextprotocol/skills`, SEP-2640).

sanzaru ships its MCP-facing skills (`sanzaru/skills/<name>/...`, package data)
and serves them two ways at once:

- every file is an ordinary resource at `skill://<name>/<path>`, so a client
  that has never heard of the extension still sees and can read them;
- `skills/list` and `skills/get` answer with each skill's frontmatter and a
  complete manifest (uri, sha256 digest, byte size) of its files, which is what
  lets a host load a skill only when the task needs it and verify what it read.

Why hand-built: the SDK's `Skills` extension landed on python-sdk main after the
2.3.0 release, so no released `mcp` has it. This module uses the same `Extension`
and `MethodBinding` API that SDK class is built on, so swapping it in later is
local to this file.

The skills are static package data, so the catalog is computed once at import
and every response is cacheable (`cacheScope: "public"`). `sanzaru-cli` is
deliberately not served: it teaches a shell command, which a hosted connector
cannot run. The plugin still ships it.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources.abc import Traversable

from mcp.server.context import HandlerResult, ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver.resources import TextResource
from mcp.shared.exceptions import MCPError
from mcp_types import Annotations, RequestParams
from mcp_types.jsonrpc import INVALID_PARAMS
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import Field

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"

#: The skills are package data: they change only with a release, so an hour of
#: client-side caching costs nothing and saves a round trip per conversation.
_TTL_MS = 60 * 60 * 1000

# SEP-2640 / the Agent Skills spec: names are lowercase words joined by hyphens.
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Spec ceilings, checked at import so a too-large skill fails the build, not a host.
_MAX_FILES_PER_SKILL = 512
_MAX_BYTES_PER_SKILL = 16 * 1024 * 1024


@dataclass(frozen=True)
class SkillFile:
    uri: str
    path: str
    text: str
    digest: str
    size: int


@dataclass(frozen=True)
class Skill:
    name: str
    frontmatter: dict[str, str]
    files: tuple[SkillFile, ...]

    @property
    def uri(self) -> str:
        return f"skill://{self.name}/SKILL.md"

    def entry(self) -> dict[str, object]:
        """The SEP-2640 `Skill` object: SKILL.md uri, frontmatter, and the complete file manifest."""
        return {
            "uri": self.uri,
            "frontmatter": dict(self.frontmatter),
            "resources": [{"uri": f.uri, "digest": f.digest, "size": f.size} for f in self.files],
        }


def parse_frontmatter(text: str, *, source: str) -> dict[str, str]:
    """Read a SKILL.md's leading `---` block of single-line `key: value` pairs.

    Deliberately strict: sanzaru's own skills only, and a multi-line or nested
    value would be silently mangled by anything looser. Raises ValueError.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source}: SKILL.md must begin with a '---' frontmatter block")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or line[:1].isspace():
            raise ValueError(f"{source}: unsupported frontmatter line {line!r}")
        fields[key.strip()] = value.strip()
    else:
        raise ValueError(f"{source}: frontmatter block is not closed")
    for required in ("name", "description"):
        if not fields.get(required):
            raise ValueError(f"{source}: frontmatter needs a non-empty {required!r}")
    return fields


def _walk(root: Traversable, prefix: str = "") -> list[tuple[str, Traversable]]:
    found: list[tuple[str, Traversable]] = []
    for child in sorted(root.iterdir(), key=lambda c: c.name):
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            found.extend(_walk(child, f"{rel}/"))
        elif child.name.endswith(".md"):
            found.append((rel, child))
    return found


def load_skills(root: Traversable | None = None) -> tuple[Skill, ...]:
    """Every skill under `root` (default: the bundled `sanzaru/skills`), validated against SEP-2640."""
    base = root if root is not None else importlib.resources.files("sanzaru").joinpath("skills")
    skills: list[Skill] = []
    for directory in sorted(base.iterdir(), key=lambda c: c.name):
        if not directory.is_dir() or not directory.joinpath("SKILL.md").is_file():
            continue
        name = directory.name
        files: list[SkillFile] = []
        for rel, entry in _walk(directory):
            data = entry.read_bytes()
            files.append(
                SkillFile(
                    uri=f"skill://{name}/{rel}",
                    path=rel,
                    text=data.decode("utf-8"),
                    digest="sha256:" + hashlib.sha256(data).hexdigest(),
                    size=len(data),
                )
            )
        skill_md = next(f for f in files if f.path == "SKILL.md")
        frontmatter = parse_frontmatter(skill_md.text, source=f"skills/{name}")
        # SEP-2640: the skill path's final segment MUST equal the frontmatter name.
        if frontmatter["name"] != name or not _NAME.fullmatch(name):
            raise ValueError(f"skills/{name}: frontmatter name {frontmatter['name']!r} must equal the directory name")
        if len(files) > _MAX_FILES_PER_SKILL or sum(f.size for f in files) > _MAX_BYTES_PER_SKILL:
            raise ValueError(f"skills/{name}: over the SEP-2640 per-skill limits")
        skills.append(Skill(name=name, frontmatter=frontmatter, files=tuple(files)))
    return tuple(skills)


def skill_resources(skills: Sequence[Skill]) -> list[TextResource]:
    """Each skill file as a plain resource, readable by any client via `resources/read`."""
    return [
        TextResource(
            uri=f.uri,
            name=f"{skill.name}/{f.path}",
            title=skill.frontmatter["name"] if f.path == "SKILL.md" else None,
            description=skill.frontmatter["description"] if f.path == "SKILL.md" else None,
            mime_type="text/markdown",
            text=f.text,
            annotations=Annotations(audience=["assistant"]),
        )
        for skill in skills
        for f in skill.files
    ]


class GetSkillParams(RequestParams):
    uri: str = Field(min_length=1, max_length=2048)


class ListSkillsParams(RequestParams):
    cursor: str | None = None


def _cacheable(result: dict[str, object], protocol_version: str) -> dict[str, object]:
    # ttlMs / cacheScope exist only on the 2026-07-28 wire; extension methods
    # bypass the runner's per-version sieve, so gate them here (as the SDK's own
    # Skills extension does).
    if protocol_version in MODERN_PROTOCOL_VERSIONS:
        return {**result, "ttlMs": _TTL_MS, "cacheScope": "public"}
    return result


class Skills(Extension):
    """Serve `skills/list` and `skills/get` over a fixed catalog."""

    identifier = SKILLS_EXTENSION

    def __init__(self, skills: Sequence[Skill]) -> None:
        self._skills: Mapping[str, Skill] = {s.uri: s for s in skills}

    def methods(self) -> Sequence[MethodBinding]:
        return (
            MethodBinding("skills/list", ListSkillsParams, self._list),
            MethodBinding("skills/get", GetSkillParams, self._get),
        )

    async def _list(self, ctx: ServerRequestContext[object, object], params: ListSkillsParams) -> HandlerResult:
        # One page: the catalog is three entries. A cursor from anywhere else is
        # not one we issued, and SEP-2640 entries are never split across pages.
        if params.cursor is not None:
            raise MCPError(code=INVALID_PARAMS, message="Unknown cursor")
        return _cacheable({"skills": [s.entry() for s in self._skills.values()]}, ctx.protocol_version)

    async def _get(self, ctx: ServerRequestContext[object, object], params: GetSkillParams) -> HandlerResult:
        skill = self._skills.get(params.uri)
        if skill is None:
            raise MCPError(code=INVALID_PARAMS, message=f"No skill is served at {params.uri[:200]}")
        return _cacheable({"skill": skill.entry()}, ctx.protocol_version)
