"""What a client is told on connect, before it reads any tool description.

Claude treated simulate_podcast as the flagship: its own description called it
the highest quality option, and nothing on the server said otherwise.
"""

import pytest

from sanzaru import descriptions
from sanzaru.server import mcp

pytestmark = pytest.mark.unit


def test_the_server_sends_instructions():
    assert mcp.instructions == descriptions.SERVER_INSTRUCTIONS


def test_instructions_stay_short():
    # Clients may truncate; tool-specific detail belongs in tool descriptions.
    assert len(descriptions.SERVER_INSTRUCTIONS) < 1500


def test_instructions_route_podcasts_to_generate_podcast():
    text = descriptions.SERVER_INSTRUCTIONS
    assert "generate_podcast" in text
    assert "eleven_v4" in text
    assert "`simulate_podcast` is EXPERIMENTAL" in text


def test_simulate_podcast_describes_itself_as_experimental():
    assert descriptions.SIMULATE_PODCAST.startswith("EXPERIMENTAL.")
    assert "wide margin" not in descriptions.SIMULATE_PODCAST
    assert "Highest" not in descriptions.SIMULATE_PODCAST


def test_generate_podcast_claims_the_topic_case():
    assert "from just a topic" in descriptions.GENERATE_PODCAST


def test_no_description_teaches_a_status_polling_loop():
    """wait_for replaced the poll-get_*_status loop; descriptions kept teaching it."""
    for name in dir(descriptions):
        value = getattr(descriptions, name)
        if isinstance(value, str) and name.isupper():
            assert "Poll get_" not in value and "poll for completion" not in value, name


@pytest.mark.parametrize("name", ["CREATE_IMAGE", "GENERATE_IMAGE"])
def test_the_two_image_tools_explain_how_they_differ(name):
    text = getattr(descriptions, name)
    assert "generate_image" in text and "create_image" in text
    assert "wait_for" in text


def test_no_description_mentions_sora():
    """OpenAI retired Sora on 2026-09-24; nothing may still steer a model toward it."""
    assert "Sora" not in descriptions.SERVER_INSTRUCTIONS
    for name in dir(descriptions):
        value = getattr(descriptions, name)
        if isinstance(value, str) and name.isupper():
            assert "Sora" not in value, name


def test_instructions_route_video_to_higgsfield():
    text = descriptions.SERVER_INSTRUCTIONS
    for needle in ("Higgsfield", "create_video", "edit_video", "max_cost_usd", "dry_run", "wait_for"):
        assert needle in text, needle


VIDEO_GENERATION_TOOLS = {
    "create_video",
    "edit_video",
    "extend_video",
    "get_video_status",
    "download_video",
    "cancel_video",
}
RETIRED_SORA_TOOLS = {"list_videos", "delete_video", "remix_video"}


class TestVideoRegistration:
    """Local video tools need only the path; generation also needs HF_KEY."""

    @pytest.fixture
    def tool_names(self, monkeypatch, tmp_path):
        import importlib
        import sys

        async def load(hf_key: str | None) -> set[str]:
            monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
            if hf_key is None:
                monkeypatch.delenv("HF_KEY", raising=False)
            else:
                monkeypatch.setenv("HF_KEY", hf_key)
            server = importlib.reload(importlib.import_module("sanzaru.server"))
            return {tool.name for tool in await server.mcp.list_tools()}

        try:
            yield load
        finally:
            sys.modules.pop("sanzaru.server", None)

    @pytest.mark.anyio
    async def test_without_hf_key_only_local_video_tools_register(self, tool_names):
        names = await tool_names(None)
        assert {"inspect_video_frame", "list_local_videos"} <= names
        assert not names & VIDEO_GENERATION_TOOLS
        assert not names & RETIRED_SORA_TOOLS

    @pytest.mark.anyio
    async def test_with_hf_key_the_generation_tools_register(self, tool_names):
        names = await tool_names("kid:secret")
        assert names >= VIDEO_GENERATION_TOOLS
        assert {"inspect_video_frame", "list_local_videos"} <= names
        assert not names & RETIRED_SORA_TOOLS
