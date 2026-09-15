"""The MCP wrappers must advertise exactly what the image tools accept.

server.py re-declares the tool signatures for the schema the model sees. When
those drift from images_api.py the model is told the wrong story: a quality the
API supports is rejected at the boundary, or an edit runs on the generation
model. Both happened in review; these pin the two surfaces together.
"""

import importlib
import sys

import pytest

from sanzaru.config import DEFAULT_IMAGE_EDIT_MODEL, DEFAULT_IMAGE_MODEL
from sanzaru.image_models import ImageQuality


@pytest.fixture
async def image_schemas(monkeypatch, tmp_path):
    """The wire-level input schemas the model sees, from a server with image tools registered."""
    monkeypatch.setenv("IMAGE_PATH", str(tmp_path))
    mcp = importlib.reload(importlib.import_module("sanzaru.server")).mcp
    try:
        yield {tool.name: tool.input_schema for tool in await mcp.list_tools()}
    finally:
        sys.modules.pop("sanzaru.server", None)


@pytest.mark.integration
async def test_generate_and_edit_advertise_every_quality_the_api_accepts(image_schemas):
    expected = set(ImageQuality.__args__)
    for name in ("generate_image", "edit_image"):
        assert set(image_schemas[name]["properties"]["quality"]["enum"]) == expected, name


@pytest.mark.integration
async def test_defaults_match_the_tool_layer(image_schemas):
    assert image_schemas["generate_image"]["properties"]["model"]["default"] == DEFAULT_IMAGE_MODEL
    assert image_schemas["edit_image"]["properties"]["model"]["default"] == DEFAULT_IMAGE_EDIT_MODEL
    assert DEFAULT_IMAGE_EDIT_MODEL != DEFAULT_IMAGE_MODEL  # sunburst for edits, flare for generation
