# SPDX-License-Identifier: MIT
"""Contract tests for the JSON envelope renderer."""

import json
import pathlib

import pydantic
import pytest

from sanzaru.cli._output import aggregate_exit_code, error_envelope, note, render, success_envelope


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
@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ([], 0),
        ([0, 0], 0),
        ([0, 5], 6),  # partial
        ([5, 4], 4),  # all failed, any timeout → resumable signal wins
        ([1, 5], 5),  # all failed, no timeout → highest code, deterministic
        ([5, 1], 5),  # ...regardless of completion order
        ([1, 1], 1),
    ],
)
def test_aggregate_exit_code(codes, expected):
    assert aggregate_exit_code(codes) == expected


# ---------- stderr diagnostics (CWE-150) ----------
#
# note() interpolates strings this process did not author — act titles a planner
# invented from a premise that may quote third-party material, API error text.
# A raw ESC on a TTY is not text: CSI overwrites the cost warning printed a
# moment ago, OSC 52 writes to the clipboard.


@pytest.mark.unit
def test_note_neutralizes_terminal_control_sequences(capsys):
    note("act1: \x1b]0;pwned\x07 \x1b[2J\x1b[1;1Hcost $0.00 \x9b31m")

    err = capsys.readouterr().err
    assert "\x1b" not in err
    assert "\x07" not in err
    assert "\x9b" not in err
    assert "\\x1b]0;pwned\\x07" in err  # still legible as what it was
    assert err.endswith("\n")


@pytest.mark.unit
def test_note_neutralizes_a_carriage_return_rewriting_the_line(capsys):
    note("planning 4 acts\rsanzaru: nothing to worry about")

    err = capsys.readouterr().err
    assert "\r" not in err
    assert "\\x0d" in err
    assert err.count("\n") == 1  # one line in, one line out


@pytest.mark.unit
def test_note_leaves_real_text_alone(capsys):
    note("café 日本語 🎙 — naïve\n\tindented")

    assert capsys.readouterr().err == "sanzaru: café 日本語 🎙 — naïve\n\tindented\n"


@pytest.mark.unit
def test_pretty_render_is_multiline_but_same_structure():
    envelope = success_envelope("x.y", {"a": 1})

    compact = render(envelope, pretty=False)
    pretty = render(envelope, pretty=True)
    assert "\n" not in compact
    assert "\n" in pretty
    assert json.loads(compact) == json.loads(pretty)
