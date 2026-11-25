"""Skills over MCP (SEP-2640): sanzaru serves its MCP-facing skills as `skill://` resources."""

import hashlib
import importlib
import json
import pathlib
import sys

import pytest
from starlette.testclient import TestClient

from sanzaru.mcp_skills import load_skills, parse_frontmatter

PROTOCOL = "2026-07-28"
SKILLS = "io.modelcontextprotocol/skills"
REPO = pathlib.Path(__file__).resolve().parents[2]
SERVED = {"sanzaru-mcp", "prompt-guidance", "podcast-production"}


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
    for name in ("SANZARU_HTTP_TOKEN", "SANZARU_ALLOWED_HOSTS", "SANZARU_IDENTITY_HEADER", "SANZARU_TASKS"):
        monkeypatch.delenv(name, raising=False)
    mod = importlib.reload(importlib.import_module("sanzaru.server"))
    try:
        yield mod
    finally:
        sys.modules.pop("sanzaru.server", None)


def _post(client, method, params=None, *, name=None):
    body = dict(params or {})
    body["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientCapabilities": {"extensions": {SKILLS: {}}},
        "io.modelcontextprotocol/clientInfo": {"name": "test-host", "version": "0"},
    }
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "host": "127.0.0.1:8000",
        "MCP-Protocol-Version": PROTOCOL,
        "MCP-Method": method,
    }
    if name is not None:
        headers["MCP-Name"] = name
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": body}, headers=headers)
    text = response.text
    if not text.lstrip().startswith("{"):
        text = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)


@pytest.mark.integration
class TestWire:
    def test_discover_advertises_skills_without_directory_reads(self, server):
        with TestClient(server.build_http_app()) as client:
            caps = _post(client, "server/discover")["result"]["capabilities"]
        assert caps["extensions"][SKILLS] == {}
        assert "resources" in caps  # SEP-2640: the extension requires the resources capability

    def test_list_returns_every_served_skill_with_a_complete_manifest(self, server):
        with TestClient(server.build_http_app()) as client:
            result = _post(client, "skills/list")["result"]
            entries = {s["frontmatter"]["name"]: s for s in result["skills"]}
            assert set(entries) == SERVED
            assert result["ttlMs"] > 0 and result["cacheScope"] == "public"
            for name, entry in entries.items():
                assert entry["uri"] == f"skill://{name}/SKILL.md"
                assert entry["frontmatter"]["description"]
                uris = [r["uri"] for r in entry["resources"]]
                assert entry["uri"] in uris  # the manifest includes SKILL.md itself
                # Every manifest digest and size describes the bytes resources/read returns.
                for resource in entry["resources"]:
                    read = _post(client, "resources/read", {"uri": resource["uri"]}, name=resource["uri"])
                    (content,) = read["result"]["contents"]
                    data = content["text"].encode("utf-8")
                    assert resource["size"] == len(data)
                    assert resource["digest"] == "sha256:" + hashlib.sha256(data).hexdigest()

    def test_get_answers_for_a_served_skill_and_refuses_anything_else(self, server):
        with TestClient(server.build_http_app()) as client:
            got = _post(client, "skills/get", {"uri": "skill://sanzaru-mcp/SKILL.md"})["result"]
            missing = _post(client, "skills/get", {"uri": "skill://sanzaru-cli/SKILL.md"})
        assert got["skill"]["frontmatter"]["name"] == "sanzaru-mcp"
        assert missing["error"]["code"] == -32602

    def test_skill_files_are_ordinary_resources_too(self, server):
        with TestClient(server.build_http_app()) as client:
            listed = {r["uri"]: r for r in _post(client, "resources/list")["result"]["resources"]}
        skill_md = listed["skill://podcast-production/SKILL.md"]
        assert skill_md["mimeType"] == "text/markdown"
        assert skill_md["annotations"]["audience"] == ["assistant"]
        assert "skill://sanzaru-mcp/reference/WORKFLOWS.md" in listed


@pytest.mark.unit
class TestCatalog:
    def test_the_bundled_skills_match_the_plugin_copies_byte_for_byte(self):
        """One text, two delivery paths: the wheel (MCP) and the plugin (Claude Code).

        The plugin cannot reach outside its own directory once installed from a
        marketplace, so it carries a copy. Edit plugin/skills/<name>/ and copy the
        directory into src/sanzaru/skills/ (or the reverse); this test fails until both agree.
        """
        bundled = REPO / "src" / "sanzaru" / "skills"
        for name in SERVED:
            ours = {p.relative_to(bundled / name): p.read_bytes() for p in (bundled / name).rglob("*.md")}
            plugin_dir = REPO / "plugin" / "skills" / name
            theirs = {p.relative_to(plugin_dir): p.read_bytes() for p in plugin_dir.rglob("*.md")}
            assert ours == theirs, f"src/sanzaru/skills/{name} and plugin/skills/{name} have drifted"

    def test_the_cli_skill_is_not_served(self):
        assert {s.name for s in load_skills()} == SERVED

    def test_a_skill_whose_name_does_not_match_its_directory_is_refused(self, tmp_path):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "alpha" / "SKILL.md").write_text("---\nname: beta\ndescription: d\n---\nbody\n")
        with pytest.raises(ValueError, match="must equal the directory name"):
            load_skills(tmp_path)

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("no frontmatter", "must begin"),
            ("---\nname: x\n", "not closed"),
            ("---\nname: x\n---\n", "description"),
            ("---\nname: x\ndescription: d\n  nested: y\n---\n", "unsupported"),
        ],
    )
    def test_frontmatter_is_parsed_strictly(self, text, why):
        with pytest.raises(ValueError, match=why):
            parse_frontmatter(text, source="t")
