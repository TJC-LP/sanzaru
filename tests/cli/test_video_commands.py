# SPDX-License-Identifier: MIT
"""Integration tests for `sanzaru video` (tool layer mocked, real envelopes)."""

import json

import pytest
from click.testing import CliRunner
from openai.types import Video

from sanzaru.cli import cli


def make_video(status: str = "queued", progress: int = 0, video_id: str = "video_test123") -> Video:
    return Video(
        id=video_id,
        created_at=1234567890,
        model="sora-2",
        object="video",
        progress=progress,
        seconds="8",
        size="1280x720",
        status=status,  # type: ignore[arg-type]
    )


@pytest.mark.integration
def test_video_create_no_wait_emits_job_envelope(mocker):
    """Bare create returns the queued job as one envelope — submission takes ~1s."""
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_video()))

    result = CliRunner().invoke(cli, ["video", "create", "a cat stretches"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is True
    assert parsed["command"] == "video.create"
    assert parsed["result"]["id"] == "video_test123"
    assert parsed["result"]["status"] == "queued"
    assert "submitted" in result.stderr


@pytest.mark.integration
def test_video_create_one_shot_downloads_to_output(mocker, tmp_path):
    """-o implies --download implies --wait: one command → final file path in envelope."""
    out = tmp_path / "out" / "clip.mp4"
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_video()))
    mocker.patch(
        "sanzaru.polling.wait_for_video",
        mocker.AsyncMock(return_value=make_video("completed", 100)),
    )

    async def fake_download(video_id, filename=None, variant="video"):
        (tmp_path / "out" / filename).write_bytes(b"video-bytes")
        return {"filename": filename, "variant": variant}

    mocker.patch("sanzaru.tools.video.download_video", fake_download)

    result = CliRunner().invoke(cli, ["video", "create", "a cat stretches", "-o", str(out)])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is True
    assert parsed["result"]["status"] == "completed"
    assert parsed["result"]["file"]["path"] == str(out)
    assert parsed["result"]["file"]["bytes"] == len(b"video-bytes")
    assert "elapsed_s" in parsed
    assert out.read_bytes() == b"video-bytes"


@pytest.mark.integration
def test_video_create_wait_failed_job_exits_5(mocker):
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_video()))
    mocker.patch("sanzaru.polling.wait_for_video", mocker.AsyncMock(return_value=make_video("failed")))

    result = CliRunner().invoke(cli, ["video", "create", "a cat", "--wait"])

    assert result.exit_code == 5
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is False
    assert parsed["error"]["type"] == "job_failed"
    assert parsed["id"] == "video_test123"


@pytest.mark.integration
def test_video_wait_timeout_exits_4_with_resume(mocker):
    from sanzaru.polling import WaitTimeoutError

    stuck = make_video("in_progress", 78, video_id="video_x")
    mocker.patch(
        "sanzaru.polling.wait_for_video",
        mocker.AsyncMock(side_effect=WaitTimeoutError("Video job video_x still running after 100s", stuck)),
    )

    result = CliRunner().invoke(cli, ["video", "wait", "video_x", "--timeout", "100s"])

    assert result.exit_code == 4
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "timeout"
    assert parsed["last_status"] == "in_progress"
    assert parsed["last_progress"] == 78
    assert "sanzaru video wait video_x" in parsed["resume"]


@pytest.mark.integration
def test_video_wait_multi_id_streams_jsonl_and_exits_partial(mocker):
    async def fake_wait(video_id, **kwargs):
        return make_video("completed" if video_id == "video_a" else "failed", 100, video_id=video_id)

    mocker.patch("sanzaru.polling.wait_for_video", fake_wait)

    result = CliRunner().invoke(cli, ["video", "wait", "video_a", "video_b"])

    assert result.exit_code == 6
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert len(lines) == 2
    by_ok = {line["ok"]: line for line in lines}
    assert by_ok[True]["result"]["id"] == "video_a"
    assert by_ok[False]["error"]["type"] == "job_failed"
    assert by_ok[False]["id"] == "video_b"


@pytest.mark.integration
def test_video_status_never_blocks(mocker):
    status = mocker.patch(
        "sanzaru.tools.video.get_video_status",
        mocker.AsyncMock(return_value=make_video("in_progress", 42)),
    )

    result = CliRunner().invoke(cli, ["video", "status", "video_test123"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["result"]["progress"] == 42
    status.assert_called_once_with("video_test123")


@pytest.mark.integration
def test_video_download_into_directory(mocker, tmp_path):
    async def fake_download(video_id, filename=None, variant="video"):
        name = filename or f"{video_id}.mp4"
        (tmp_path / name).write_bytes(b"x")
        return {"filename": name, "variant": variant}

    mocker.patch("sanzaru.tools.video.download_video", fake_download)

    result = CliRunner().invoke(cli, ["video", "download", "video_x", "-o", str(tmp_path) + "/"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["result"]["file"]["path"] == str(tmp_path / "video_x.mp4")


@pytest.mark.integration
def test_video_files_lists_local(mocker):
    mocker.patch("sanzaru.tools.video.list_local_videos", mocker.AsyncMock(return_value={"data": []}))

    result = CliRunner().invoke(cli, ["video", "files"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"] == {"data": []}


@pytest.mark.integration
def test_video_create_rejects_unknown_size():
    result = CliRunner().invoke(cli, ["video", "create", "a cat", "--size", "999x999"])

    assert result.exit_code == 2
