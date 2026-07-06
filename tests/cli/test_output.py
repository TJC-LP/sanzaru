# SPDX-License-Identifier: MIT
"""Contract tests for the JSON envelope renderer."""

import json
import pathlib

import pydantic
import pytest

from sanzaru.cli._output import error_envelope, render, success_envelope


class _Usage(pydantic.BaseModel):
    total_tokens: int
    input_tokens: int


@pytest.mark.unit
def test_success_envelope_shape_and_elapsed_rounding():
    envelope = success_envelope("video.create", {"id": "video_x"}, elapsed_s=184.23456)

    parsed = json.loads(render(envelope, pretty=False))
    assert parsed == {"v": 1, "ok": True, "command": "video.create", "result": {"id": "video_x"}, "elapsed_s": 184.2}


@pytest.mark.unit
def test_tuples_render_as_json_arrays():
    envelope = success_envelope("image.generate", {"size": (1536, 1024)})

    parsed = json.loads(render(envelope, pretty=False))
    assert parsed["result"]["size"] == [1536, 1024]


@pytest.mark.unit
def test_paths_and_pydantic_models_render():
    result = {"path": pathlib.Path("/tmp/out.png"), "usage": _Usage(total_tokens=10, input_tokens=4)}

    parsed = json.loads(render(success_envelope("image.generate", result), pretty=False))
    assert parsed["result"]["path"] == "/tmp/out.png"
    assert parsed["result"]["usage"] == {"total_tokens": 10, "input_tokens": 4}


@pytest.mark.unit
def test_unrenderable_type_raises_instead_of_str_fallback():
    with pytest.raises(TypeError, match="Unrenderable"):
        render(success_envelope("x.y", {"bad": object()}), pretty=False)


@pytest.mark.unit
def test_error_envelope_carries_resume_and_extra():
    envelope = error_envelope(
        "video.wait",
        "timeout",
        "Video job video_x still running after 1800s",
        resume="sanzaru video wait video_x --download",
        extra={"id": "video_x", "last_status": "in_progress"},
    )

    parsed = json.loads(render(envelope, pretty=False))
    assert parsed["ok"] is False
    assert parsed["error"] == {"type": "timeout", "message": "Video job video_x still running after 1800s"}
    assert parsed["resume"] == "sanzaru video wait video_x --download"
    assert parsed["id"] == "video_x"
    assert parsed["last_status"] == "in_progress"


@pytest.mark.unit
def test_pretty_render_is_multiline_but_same_structure():
    envelope = success_envelope("x.y", {"a": 1})

    compact = render(envelope, pretty=False)
    pretty = render(envelope, pretty=True)
    assert "\n" not in compact
    assert "\n" in pretty
    assert json.loads(compact) == json.loads(pretty)
