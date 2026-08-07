"""Test the whisper server functionality."""

import pytest

from sanzaru.audio.constants import SortBy
from sanzaru.audio.models import ListAudioFilesInputParams
from sanzaru.tools.podcast import PodcastResult

pytestmark = pytest.mark.audio


def test_sort_by_enum() -> None:
    """Test the SortBy enum has all expected values."""
    assert SortBy.NAME.value == "name"
    assert SortBy.SIZE.value == "size"
    assert SortBy.DURATION.value == "duration"
    assert SortBy.MODIFIED_TIME.value == "modified_time"
    assert SortBy.FORMAT.value == "format"


def test_list_audio_files_params_defaults() -> None:
    """Test the default parameters for ListAudioFilesInputParams."""
    params: ListAudioFilesInputParams = ListAudioFilesInputParams()
    assert params.pattern is None
    assert params.min_size_bytes is None
    assert params.max_size_bytes is None
    assert params.min_duration_seconds is None
    assert params.max_duration_seconds is None
    assert params.min_modified_time is None
    assert params.max_modified_time is None
    assert params.format is None
    assert params.sort_by == SortBy.NAME
    assert params.reverse is False


@pytest.fixture
def audio_server(tmp_path, monkeypatch):
    """A `sanzaru.server` with the audio tools actually registered.

    They register at import time behind `check_audio_available()`, which needs
    AUDIO_PATH — unset in the test env, so importing the module normally yields
    a server with no podcast tool at all. Without the reload these tests would
    skip and read as passing.
    """
    import importlib
    import sys

    monkeypatch.setenv("AUDIO_PATH", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import sanzaru.config

    sanzaru.config.get_path.cache_clear()
    module = importlib.reload(importlib.import_module("sanzaru.server"))
    try:
        yield module
    finally:
        # Leave no half-configured server behind for the rest of the session.
        sys.modules.pop("sanzaru.server", None)
        sanzaru.config.get_path.cache_clear()


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_tool_forwards_output_filename(audio_server, mocker) -> None:
    """The MCP wrapper renames `output_filename` -> `filename` on the way down.

    Nothing asserted that before, so a rename on either side would have passed
    CI while silently dropping the caller's filename (#54's whole subject).
    """
    tools = {t.name: t for t in await audio_server.mcp.list_tools()}
    assert "generate_podcast" in tools, "audio tools should be registered with AUDIO_PATH set"
    # The parameter has to be in the schema, or an MCP caller cannot pass it.
    assert "output_filename" in tools["generate_podcast"].inputSchema["properties"]

    generate = mocker.patch(
        "sanzaru.tools.podcast.generate_podcast",
        mocker.AsyncMock(
            return_value=PodcastResult(
                output_file="ep.mp3",
                title="T",
                segment_count=1,
                estimated_duration_seconds=1.0,
                speakers=["A"],
                transcript="",
            )
        ),
    )
    script = {
        "title": "T",
        "speakers": [{"id": "a", "name": "A", "voice": "ash", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "a", "text": "hi"}],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    await audio_server.mcp.call_tool("generate_podcast", {"script": script, "output_filename": "ep.mp3"})

    assert generate.call_args.kwargs["filename"] == "ep.mp3"


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_tool_rejects_a_path_as_output_filename(audio_server, mocker) -> None:
    """Typed as `Filename`, so the schema refuses a path before an episode is
    synthesized — not the storage layer after it is paid for.

    The script has to be *valid*: with `script={}` the call fails on
    PodcastScript's own required keys, and since pydantic writes "N validation
    errors" into every message, a loose `match` would pass with `str` here and
    guard nothing. Asserting the field name is what makes this test real.
    """
    generate = mocker.patch("sanzaru.tools.podcast.generate_podcast", mocker.AsyncMock())
    valid_script = {
        "title": "T",
        "speakers": [{"id": "a", "name": "A", "voice": "ash", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "a", "text": "hi"}],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    with pytest.raises(Exception) as exc_info:
        await audio_server.mcp.call_tool(
            "generate_podcast", {"script": valid_script, "output_filename": "../escape.mp3"}
        )

    assert "output_filename" in str(exc_info.value)
    generate.assert_not_called()
