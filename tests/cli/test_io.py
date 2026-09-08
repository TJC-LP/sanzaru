# SPDX-License-Identifier: MIT
"""Unit tests for CLI path/content resolution."""

import errno
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
    write_output_bytes,
)
from sanzaru.cli._runtime import CLIError
from sanzaru.storage import set_storage_backend


@pytest.fixture(autouse=True)
def _reset_storage_override():
    yield
    set_storage_backend(None)


@pytest.fixture(autouse=True)
def _reset_path_cache():
    """get_path is lru_cached, so a test that sets SANZARU_MEDIA_PATH must not leak."""
    from sanzaru.config import get_path

    get_path.cache_clear()
    yield
    get_path.cache_clear()


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
def test_inputs_may_span_directories(tmp_path):
    """#38: the batch is anchored per file, not per directory.

    The QC workflow this exists for is an episode in one directory and its
    ffmpeg-cut windows in another; before, you had to copy one next to the
    other first.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.png").write_bytes(b"x")
    (b / "two.png").write_bytes(b"x")
    session = PathSession()

    assert resolve_input(session, str(a / "one.png"), "reference", "--input-image") == "one.png"
    assert resolve_input(session, str(b / "two.png"), "reference", "--input-image") == "two.png"

    assert session.file_overrides == {("reference", "one.png"): a, ("reference", "two.png"): b}
    # The output side still anchors to one directory: the first input's.
    assert session.overrides["reference"] == a


@pytest.mark.unit
def test_same_basename_in_two_dirs_is_still_a_usage_error(tmp_path):
    """The one genuinely unresolvable case, and all that is left of the old rule.

    Tools are handed bare names and the envelope reports bare names, so two
    files called `ep.mp3` would leave one of them unaddressable.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "ep.mp3").write_bytes(b"x")
    (b / "ep.mp3").write_bytes(b"x")
    session = PathSession()
    resolve_input(session, str(a / "ep.mp3"), "audio", "FILES")

    with pytest.raises(CLIError, match="both named") as excinfo:
        resolve_input(session, str(b / "ep.mp3"), "audio", "FILES")
    assert excinfo.value.exit_code == 2


@pytest.mark.unit
def test_the_same_file_named_twice_is_not_a_collision(tmp_path):
    f = tmp_path / "ep.mp3"
    f.write_bytes(b"x")
    session = PathSession()

    assert resolve_input(session, str(f), "audio", "FILES") == "ep.mp3"
    assert resolve_input(session, str(f), "audio", "FILES") == "ep.mp3"


@pytest.mark.unit
def test_mixing_bare_and_path_inputs_conflicts(tmp_path):
    f = tmp_path / "one.png"
    f.write_bytes(b"x")
    session = PathSession()
    resolve_input(session, "bare.png", "reference", "--input-image")

    with pytest.raises(CLIError, match="cannot mix"):
        resolve_input(session, str(f), "reference", "--input-image")


# ---------- bare-name resolution (CWE-427) ----------


@pytest.mark.unit
def test_a_cwd_file_capturing_a_bare_name_is_announced(monkeypatch, tmp_path, capsys):
    """The substitution used to be invisible: cwd won and nothing said so.

    Agents run in workspaces holding material they did not write, so a planted
    `episode.mp3` stood in for the operator's library file of the same name and
    the envelope — which reports bare names — showed a matching basename either
    way. Cwd still wins; it just has to be legible.
    """
    media, workspace = tmp_path / "media", tmp_path / "workspace"
    (media / "audio").mkdir(parents=True)
    (media / "audio" / "episode.mp3").write_bytes(b"the operator's episode")
    workspace.mkdir()
    (workspace / "episode.mp3").write_bytes(b"planted")
    monkeypatch.setenv("SANZARU_MEDIA_PATH", str(media))
    monkeypatch.chdir(workspace)
    session = PathSession()

    assert resolve_input(session, "episode.mp3", "audio", "FILE") == "episode.mp3"

    err = capsys.readouterr().err
    assert str((workspace / "episode.mp3").resolve()) in err
    assert str((media / "audio" / "episode.mp3").resolve()) in err
    assert session.overrides["audio"] == workspace.resolve()


@pytest.mark.unit
def test_a_bare_name_with_no_library_twin_stays_quiet(monkeypatch, tmp_path, capsys):
    """Only ambiguity is worth a line — the common chained-command case is not."""
    media, workspace = tmp_path / "media", tmp_path / "workspace"
    media.mkdir()
    workspace.mkdir()
    (workspace / "notes.mp3").write_bytes(b"x")
    monkeypatch.setenv("SANZARU_MEDIA_PATH", str(media))
    monkeypatch.chdir(workspace)
    session = PathSession()

    assert resolve_input(session, "notes.mp3", "audio", "FILE") == "notes.mp3"

    assert capsys.readouterr().err == ""
    assert session.overrides["audio"] == workspace.resolve()


@pytest.mark.unit
def test_a_bare_name_is_never_resolved_through_a_cwd_symlink(monkeypatch, tmp_path):
    elsewhere = tmp_path / "elsewhere.mp3"
    elsewhere.write_bytes(b"not what was asked for")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "episode.mp3").symlink_to(elsewhere)
    monkeypatch.chdir(workspace)
    session = PathSession()

    with pytest.raises(CLIError, match="symbolic link") as excinfo:
        resolve_input(session, "episode.mp3", "audio", "FILE")

    assert excinfo.value.exit_code == 2
    assert "./episode.mp3" in str(excinfo.value)  # the message says how to opt in


@pytest.mark.unit
def test_path_form_still_follows_a_symlink_without_complaint(monkeypatch, tmp_path, capsys):
    """`./name` is the documented way to say "the local file, links and all"."""
    real = tmp_path / "real.mp3"
    real.write_bytes(b"x")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "episode.mp3").symlink_to(real)
    monkeypatch.chdir(workspace)
    session = PathSession()

    assert resolve_input(session, "./episode.mp3", "audio", "FILE") == "real.mp3"

    assert capsys.readouterr().err == ""
    assert session.overrides["audio"] == tmp_path.resolve()


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

    assert plan.filename is not None
    assert plan.filename.startswith("sanzaru_tmp_")
    assert plan.filename.endswith(".mp3")  # format detection reads the suffix
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
    (in_dir / "sanzaru_tmp_0123456789abcdef.mp3").write_bytes(b"audio")
    session = PathSession(overrides={"audio": in_dir})
    plan = OutputPlan(
        path_type="audio", filename="sanzaru_tmp_0123456789abcdef.mp3", final_dir=out_dir, final_name="clip.mp3"
    )

    final = await finalize_output(session, plan, "sanzaru_tmp_0123456789abcdef.mp3")

    assert final == str(out_dir / "clip.mp3")
    assert (out_dir / "clip.mp3").read_bytes() == b"audio"
    assert not (in_dir / "sanzaru_tmp_0123456789abcdef.mp3").exists()


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


# ---------- symlinked destinations (CWE-59) ----------
#
# `-o` writes outside the media sandbox by design, so security.py's
# check_not_symlink never sees these paths. Every relocation primitive used to
# follow a link planted at the destination, turning an artifact write into a
# truncate of whatever the link pointed at, with the operator's privileges.


def _stage(tmp_path):
    """in/ with a staged artifact, out/, and a victim file a link can aim at."""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    (in_dir / "staged.mp3").write_bytes(b"audio")
    victim = tmp_path / "bashrc"
    victim.write_text("# the operator's shell")
    return in_dir, out_dir, victim


@pytest.mark.unit
async def test_finalize_refuses_to_move_through_a_symlinked_target(tmp_path):
    in_dir, out_dir, victim = _stage(tmp_path)
    (out_dir / "clip.mp3").symlink_to(victim)
    session = PathSession(overrides={"audio": in_dir})
    plan = OutputPlan(path_type="audio", filename="staged.mp3", final_dir=out_dir, final_name="clip.mp3")

    with pytest.raises(CLIError, match="symbolic link") as excinfo:
        await finalize_output(session, plan, "staged.mp3")

    assert excinfo.value.exit_code == 2
    assert victim.read_text() == "# the operator's shell"


@pytest.mark.unit
async def test_finalize_refuses_a_cross_device_move_through_a_symlink(tmp_path, mocker):
    """Cross-device, shutil.move fell back to copy2 — which opens and follows."""
    mocker.patch("sanzaru.cli._io.os.replace", side_effect=OSError(errno.EXDEV, "cross-device link"))
    in_dir, out_dir, victim = _stage(tmp_path)
    (out_dir / "clip.mp3").symlink_to(victim)
    session = PathSession(overrides={"audio": in_dir})
    plan = OutputPlan(path_type="audio", filename="staged.mp3", final_dir=out_dir, final_name="clip.mp3")

    with pytest.raises(CLIError, match="symbolic link"):
        await finalize_output(session, plan, "staged.mp3")

    assert victim.read_text() == "# the operator's shell"


@pytest.mark.unit
async def test_a_cross_device_move_still_relocates_the_artifact(tmp_path, mocker):
    """The EXDEV fallback is hand-rolled now, so prove it actually moves."""
    mocker.patch("sanzaru.cli._io.os.replace", side_effect=OSError(errno.EXDEV, "cross-device link"))
    in_dir, out_dir, _ = _stage(tmp_path)
    session = PathSession(overrides={"audio": in_dir})
    plan = OutputPlan(path_type="audio", filename="staged.mp3", final_dir=out_dir, final_name="clip.mp3")

    final = await finalize_output(session, plan, "staged.mp3")

    assert final == str(out_dir / "clip.mp3")
    assert (out_dir / "clip.mp3").read_bytes() == b"audio"
    assert not (in_dir / "staged.mp3").exists()


@pytest.mark.unit
async def test_finalize_refuses_to_move_a_symlinked_staging_file(tmp_path):
    """A swapped source moves as a link, so the artifact "lands" as a redirect."""
    in_dir, out_dir, victim = _stage(tmp_path)
    (in_dir / "staged.mp3").unlink()
    (in_dir / "staged.mp3").symlink_to(victim)
    session = PathSession(overrides={"audio": in_dir})
    plan = OutputPlan(path_type="audio", filename="staged.mp3", final_dir=out_dir, final_name="clip.mp3")

    with pytest.raises(CLIError, match="symbolic link"):
        await finalize_output(session, plan, "staged.mp3")

    assert not (out_dir / "clip.mp3").exists()


@pytest.mark.unit
async def test_default_backend_copy_refuses_a_dangling_symlink(tmp_path):
    """Dangling on purpose: `exists() and is_symlink()` would wave this through,
    and the write would then *create* the file the link names."""
    from sanzaru.storage.local import LocalStorageBackend

    media, target = tmp_path / "media", tmp_path / "target"
    media.mkdir()
    target.mkdir()
    (media / "tts.mp3").write_bytes(b"speech")
    victim = tmp_path / "authorized_keys"  # deliberately absent
    (target / "tts.mp3").symlink_to(victim)
    set_storage_backend(LocalStorageBackend(path_overrides={"audio": media}))
    session = PathSession(default_locked={"audio"})
    plan = OutputPlan(
        path_type="audio", filename="tts.mp3", final_dir=target, final_name="tts.mp3", via_default_backend=True
    )

    with pytest.raises(CLIError, match="symbolic link"):
        await finalize_output(session, plan, "tts.mp3")

    assert not victim.exists()


@pytest.mark.unit
def test_a_link_planted_after_the_check_is_still_refused(tmp_path, mocker):
    """The explicit check is the readable error; O_NOFOLLOW is what wins the race."""
    mocker.patch("sanzaru.cli._io._refuse_symlink")  # lose the TOCTOU race on purpose
    victim = tmp_path / "bashrc"
    victim.write_text("# the operator's shell")
    (tmp_path / "clip.mp3").symlink_to(victim)

    with pytest.raises(CLIError, match="symbolic link"):
        write_output_bytes(tmp_path / "clip.mp3", b"audio")

    assert victim.read_text() == "# the operator's shell"


@pytest.mark.unit
def test_plan_output_refuses_a_symlinked_file_target(tmp_path):
    """Nothing relocates in the direct-write case — the storage backend writes
    straight to this path — so plan time is the only place to refuse it."""
    victim = tmp_path / "bashrc"
    victim.write_text("# the operator's shell")
    (tmp_path / "clip.mp3").symlink_to(victim)

    with pytest.raises(CLIError, match="symbolic link") as excinfo:
        plan_output(PathSession(), str(tmp_path / "clip.mp3"), "audio")

    assert excinfo.value.exit_code == 2


@pytest.mark.unit
def test_plan_output_follows_a_symlinked_directory_target(tmp_path):
    """A symlinked directory target is followed and resolved, not refused.

    A directory is a parent of what gets written, not the write target — the
    same rule `_refuse_symlink` applies to parent components. Refusing it broke
    `-o /tmp` on macOS, where /tmp itself is a symlink to /private/tmp.
    """
    real_dir = tmp_path / "elsewhere"
    real_dir.mkdir()
    (tmp_path / "out").symlink_to(real_dir, target_is_directory=True)

    session = PathSession()
    plan = plan_output(session, str(tmp_path / "out") + "/", "video")

    # Resolved to the real directory, so every later write and relocation sees
    # the true path rather than re-deciding what the link means.
    assert session.overrides["video"] == real_dir.resolve()
    assert plan.filename is None


@pytest.mark.unit
def test_plan_output_still_refuses_a_dangling_symlink_target(tmp_path):
    """Dangling is the plant: following it would *create* whatever it names."""
    (tmp_path / "out").symlink_to(tmp_path / "nowhere")

    with pytest.raises(CLIError, match="symbolic link"):
        plan_output(PathSession(), str(tmp_path / "out"), "video")


@pytest.mark.unit
def test_staging_names_are_unguessable(tmp_path):
    """A predictable staging name is plantable — it lands in the *inputs'* dir."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "src.wav").write_bytes(b"x")

    names = set()
    for _ in range(3):
        session = PathSession()
        resolve_input(session, str(in_dir / "src.wav"), "audio", "FILE")
        plan = plan_output(session, str(tmp_path / "out" / "converted.mp3"), "audio")
        assert plan.filename is not None
        assert "/" not in plan.filename and "\\" not in plan.filename  # tool layer takes bare names
        names.add(plan.filename)

    assert len(names) == 3
    assert "converted__sanzaru_tmp.mp3" not in names


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
    # plan_output hands the tool layer a sanzaru_tmp_* staging name when the
    # output dir differs from the one the inputs pinned; finalize_output moves it.
    payload: dict[str, object] = {"output_file": "sanzaru_tmp_0123456789abcdef.wav"}

    reconcile_output_name(payload, str(tmp_path / "out" / "converted.wav"))

    assert payload["output_file"] == "converted.wav"


@pytest.mark.unit
def test_reconcile_is_a_noop_without_a_name_field():
    payload: dict[str, object] = {"id": "vid_123"}

    reconcile_output_name(payload, "/tmp/whatever.mp4")

    assert payload == {"id": "vid_123"}
