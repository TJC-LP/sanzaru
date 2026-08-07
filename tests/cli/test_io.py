# SPDX-License-Identifier: MIT
"""Unit tests for CLI path/content resolution."""

import io

import pytest

from sanzaru.cli._io import (
    OutputPlan,
    PathSession,
    finalize_output,
    install_overrides,
    plan_output,
    read_content_arg,
    reconcile_output_name,
    resolve_input,
)
from sanzaru.cli._runtime import CLIError
from sanzaru.storage import set_storage_backend


@pytest.fixture(autouse=True)
def _reset_storage_override():
    yield
    set_storage_backend(None)


# ---------- read_content_arg ----------


@pytest.mark.unit
def test_content_arg_inline_passthrough():
    assert read_content_arg("a plain prompt", "PROMPT") == "a plain prompt"


@pytest.mark.unit
def test_content_arg_at_file(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("from a file")
    assert read_content_arg(f"@{f}", "PROMPT") == "from a file"


@pytest.mark.unit
def test_content_arg_double_at_escapes_literal():
    assert read_content_arg("@@handle", "PROMPT") == "@handle"


@pytest.mark.unit
def test_content_arg_stdin(mocker):
    mocker.patch("sys.stdin", io.StringIO("piped in"))
    assert read_content_arg("-", "PROMPT") == "piped in"


@pytest.mark.unit
def test_content_arg_missing_file_is_usage_error():
    with pytest.raises(CLIError) as excinfo:
        read_content_arg("@/nonexistent/prompt.txt", "PROMPT")
    assert excinfo.value.exit_code == 2


# ---------- resolve_input ----------


@pytest.mark.unit
def test_bare_filename_keeps_default_backend():
    session = PathSession()
    assert resolve_input(session, "hero.png", "reference", "--input-ref") == "hero.png"
    assert session.overrides == {}
    assert "reference" in session.default_locked


@pytest.mark.unit
def test_path_input_overrides_parent_dir(tmp_path):
    f = tmp_path / "hero.png"
    f.write_bytes(b"x")
    session = PathSession()

    assert resolve_input(session, str(f), "reference", "--input-ref") == "hero.png"
    assert session.overrides["reference"] == tmp_path


@pytest.mark.unit
def test_missing_path_input_is_usage_error(tmp_path):
    session = PathSession()
    with pytest.raises(CLIError) as excinfo:
        resolve_input(session, str(tmp_path / "nope.png"), "reference", "--input-ref")
    assert excinfo.value.exit_code == 2


@pytest.mark.unit
def test_inputs_in_different_dirs_conflict(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.png").write_bytes(b"x")
    (b / "two.png").write_bytes(b"x")
    session = PathSession()
    resolve_input(session, str(a / "one.png"), "reference", "--input-image")

    with pytest.raises(CLIError, match="share one directory"):
        resolve_input(session, str(b / "two.png"), "reference", "--input-image")


@pytest.mark.unit
def test_mixing_bare_and_path_inputs_conflicts(tmp_path):
    f = tmp_path / "one.png"
    f.write_bytes(b"x")
    session = PathSession()
    resolve_input(session, "bare.png", "reference", "--input-image")

    with pytest.raises(CLIError, match="cannot mix"):
        resolve_input(session, str(f), "reference", "--input-image")


# ---------- plan_output ----------


@pytest.mark.unit
def test_output_file_sets_override_and_filename(tmp_path):
    session = PathSession()
    plan = plan_output(session, str(tmp_path / "out" / "clip.mp4"), "video")

    assert plan.filename == "clip.mp4"
    assert plan.final_dir is None
    assert session.overrides["video"] == tmp_path / "out"
    assert (tmp_path / "out").is_dir()  # parents created


@pytest.mark.unit
def test_output_existing_dir_autogenerates_name(tmp_path):
    session = PathSession()
    plan = plan_output(session, str(tmp_path), "video")

    assert plan.filename is None
    assert session.overrides["video"] == tmp_path


@pytest.mark.unit
def test_output_trailing_slash_is_dir_target(tmp_path):
    session = PathSession()
    plan = plan_output(session, str(tmp_path / "new") + "/", "video")

    assert plan.filename is None
    assert session.overrides["video"] == tmp_path / "new"
    assert (tmp_path / "new").is_dir()


@pytest.mark.unit
def test_no_output_unconfigured_falls_back_to_cwd(monkeypatch, tmp_path, capsys):
    for var in ("VIDEO_PATH", "IMAGE_PATH", "AUDIO_PATH", "SANZARU_MEDIA_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    session = PathSession()

    plan = plan_output(session, None, "video")

    assert plan.filename is None
    assert session.overrides["video"] == tmp_path
    assert "no media dir configured" in capsys.readouterr().err


@pytest.mark.unit
def test_no_output_with_media_env_uses_default_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
    session = PathSession()

    plan = plan_output(session, None, "video")

    assert plan.filename is None
    assert session.overrides == {}


@pytest.mark.unit
def test_output_conflicting_with_input_dir_plans_tmp_and_move(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "src.wav").write_bytes(b"x")
    session = PathSession()
    resolve_input(session, str(in_dir / "src.wav"), "audio", "FILE")

    plan = plan_output(session, str(out_dir / "converted.mp3"), "audio")

    assert plan.filename == "converted__sanzaru_tmp.mp3"
    assert plan.final_dir == out_dir
    assert plan.final_name == "converted.mp3"
    assert session.overrides["audio"] == in_dir  # input dir still wins for the write


@pytest.mark.unit
def test_output_with_bare_input_goes_via_default_backend(tmp_path):
    session = PathSession()
    resolve_input(session, "library.wav", "audio", "FILE")

    plan = plan_output(session, str(tmp_path / "copy.mp3"), "audio")

    assert plan.via_default_backend is True
    assert plan.final_dir == tmp_path
    assert plan.final_name == "copy.mp3"
    assert session.overrides == {}


# ---------- finalize_output ----------


@pytest.mark.unit
async def test_finalize_moves_tmp_file_to_target(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    (in_dir / "clip__sanzaru_tmp.mp3").write_bytes(b"audio")
    session = PathSession(overrides={"audio": in_dir})
    plan = OutputPlan(path_type="audio", filename="clip__sanzaru_tmp.mp3", final_dir=out_dir, final_name="clip.mp3")

    final = await finalize_output(session, plan, "clip__sanzaru_tmp.mp3")

    assert final == str(out_dir / "clip.mp3")
    assert (out_dir / "clip.mp3").read_bytes() == b"audio"
    assert not (in_dir / "clip__sanzaru_tmp.mp3").exists()


@pytest.mark.unit
async def test_finalize_returns_display_path_for_direct_write(tmp_path):
    session = PathSession(overrides={"video": tmp_path})
    install_overrides(session)
    (tmp_path / "clip.mp4").write_bytes(b"video")
    plan = OutputPlan(path_type="video", filename="clip.mp4")

    final = await finalize_output(session, plan, "clip.mp4")

    assert final == str(tmp_path / "clip.mp4")


@pytest.mark.unit
async def test_finalize_copies_bytes_out_of_default_backend(tmp_path, monkeypatch):
    media, target = tmp_path / "media", tmp_path / "target"
    media.mkdir()
    target.mkdir()
    (media / "tts.mp3").write_bytes(b"speech")
    # The "default backend" for this test is a local backend rooted at media/.
    from sanzaru.storage.local import LocalStorageBackend

    set_storage_backend(LocalStorageBackend(path_overrides={"audio": media}))
    session = PathSession(default_locked={"audio"})
    plan = OutputPlan(
        path_type="audio", filename="tts.mp3", final_dir=target, final_name="tts.mp3", via_default_backend=True
    )

    final = await finalize_output(session, plan, "tts.mp3")

    assert final == str(target / "tts.mp3")
    assert (target / "tts.mp3").read_bytes() == b"speech"
    assert (media / "tts.mp3").exists()  # library copy retained


# ---------- reconcile_output_name (#54) ----------


@pytest.mark.unit
@pytest.mark.parametrize("key", ["output_file", "output_filename", "filename"])
def test_reconcile_rewrites_every_name_spelling(key, tmp_path):
    payload: dict[str, object] = {key: "Some_Title_1737000000.mp3", "title": "Some Title"}

    reconcile_output_name(payload, str(tmp_path / "eleven-demo.mp3"))

    assert payload[key] == "eleven-demo.mp3"
    assert payload["title"] == "Some Title"  # untouched


@pytest.mark.unit
def test_reconcile_replaces_tmp_name_left_by_a_cross_dir_move(tmp_path):
    # plan_output hands the tool layer a __sanzaru_tmp name when the output dir
    # differs from the one the inputs pinned; finalize_output then moves it.
    payload: dict[str, object] = {"output_file": "converted__sanzaru_tmp.wav"}

    reconcile_output_name(payload, str(tmp_path / "out" / "converted.wav"))

    assert payload["output_file"] == "converted.wav"


@pytest.mark.unit
def test_reconcile_is_a_noop_without_a_name_field():
    payload: dict[str, object] = {"id": "vid_123"}

    reconcile_output_name(payload, "/tmp/whatever.mp4")

    assert payload == {"id": "vid_123"}
