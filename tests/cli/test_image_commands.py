# SPDX-License-Identifier: MIT
"""Integration tests for `sanzaru image`, top-level `wait`, and `capabilities`."""

import json

import pytest
from click.testing import CliRunner
from openai.types import Video

from sanzaru.cli import cli
from sanzaru.types import ImageGenerateResult


def _image_response(status: str = "queued", response_id: str = "resp_test1"):
    return {"id": response_id, "status": status, "created_at": 1234567890.0}


def _video(status: str = "completed", video_id: str = "video_a") -> Video:
    return Video(
        id=video_id,
        created_at=1,
        model="sora-2",
        object="video",
        progress=100,
        seconds="8",
        size="1280x720",
        status=status,  # type: ignore[arg-type]
    )


@pytest.mark.integration
def test_image_create_no_wait_builds_sparse_tool_config(mocker):
    """Only explicitly-passed flags land in tool_config (tool injects gpt-image-2)."""
    create = mocker.patch("sanzaru.tools.image.create_image", mocker.AsyncMock(return_value=_image_response()))

    result = CliRunner().invoke(cli, ["image", "create", "a cyberpunk courier"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["result"]["id"] == "resp_test1"
    assert create.call_args.kwargs["tool_config"] == {"type": "image_generation"}
    assert create.call_args.kwargs["model"] == "gpt-5.2"
    assert create.call_args.kwargs["previous_response_id"] is None


@pytest.mark.integration
def test_image_create_forwards_flags_and_previous_id(mocker):
    create = mocker.patch("sanzaru.tools.image.create_image", mocker.AsyncMock(return_value=_image_response()))

    result = CliRunner().invoke(
        cli,
        [
            "image",
            "create",
            "add neon rain",
            "--image-model",
            "gpt-image-1.5",
            "--size",
            "1536x1024",
            "--quality",
            "high",
            "--previous-id",
            "resp_prev",
        ],
    )

    assert result.exit_code == 0
    config = create.call_args.kwargs["tool_config"]
    assert config == {
        "type": "image_generation",
        "model": "gpt-image-1.5",
        "size": "1536x1024",
        "quality": "high",
    }
    assert create.call_args.kwargs["previous_response_id"] == "resp_prev"


@pytest.mark.integration
def test_image_create_one_shot_downloads(mocker, tmp_path):
    out = tmp_path / "art" / "hero.png"
    mocker.patch("sanzaru.tools.image.create_image", mocker.AsyncMock(return_value=_image_response()))
    mocker.patch("sanzaru.polling.wait_for_image", mocker.AsyncMock(return_value=_image_response("completed")))

    async def fake_download(response_id, filename=None):
        (tmp_path / "art" / filename).write_bytes(b"png-bytes")
        return {"filename": filename, "size": (1024, 1024), "format": "png"}

    mocker.patch("sanzaru.tools.image.download_image", fake_download)

    result = CliRunner().invoke(cli, ["image", "create", "hero art", "-o", str(out)])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["result"]["status"] == "completed"
    assert parsed["result"]["size"] == [1024, 1024]
    assert parsed["result"]["file"]["path"] == str(out)
    assert out.read_bytes() == b"png-bytes"


@pytest.mark.integration
def test_image_wait_timeout_has_resume(mocker):
    from sanzaru.polling import WaitTimeoutError

    mocker.patch(
        "sanzaru.polling.wait_for_image",
        mocker.AsyncMock(
            side_effect=WaitTimeoutError(
                "Image job resp_x still running after 60s", _image_response("in_progress", "resp_x")
            )
        ),
    )

    result = CliRunner().invoke(cli, ["image", "wait", "resp_x", "--timeout", "60s"])

    assert result.exit_code == 4
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "timeout"
    assert parsed["last_status"] == "in_progress"
    assert "sanzaru image wait resp_x" in parsed["resume"]


@pytest.mark.integration
def test_image_generate_single_with_output_file(mocker, tmp_path):
    out = tmp_path / "icon.png"

    async def fake_generate(prompt, model, size, quality, background, output_format, moderation, filename):
        (tmp_path / filename).write_bytes(b"i")
        return ImageGenerateResult(filename=filename, size=(1024, 1024), format="png", model=model)

    mocker.patch("sanzaru.tools.images_api.generate_image", fake_generate)

    result = CliRunner().invoke(cli, ["image", "generate", "an app icon", "-o", str(out)])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["result"]["file"]["path"] == str(out)
    assert parsed["result"]["model"] == "gpt-image-2"  # DEFAULT_IMAGE_MODEL applied
    assert parsed["input"] == {"index": 0, "prompt": "an app icon"}


@pytest.mark.integration
def test_image_generate_batch_fans_out_jsonl(mocker, tmp_path):
    calls: list[str] = []

    async def fake_generate(prompt, model, size, quality, background, output_format, moderation, filename):
        calls.append(filename)
        (tmp_path / filename).write_bytes(b"i")
        return ImageGenerateResult(filename=filename, size=(1024, 1024), format="png", model=model)

    mocker.patch("sanzaru.tools.images_api.generate_image", fake_generate)

    result = CliRunner().invoke(cli, ["image", "generate", "icon", "banner", "--count", "2", "-o", str(tmp_path) + "/"])

    assert result.exit_code == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert len(lines) == 4
    assert len(set(calls)) == 4  # collision-proof per-image filenames
    assert {line["input"]["index"] for line in lines} == {0, 1, 2, 3}


@pytest.mark.integration
def test_image_generate_partial_failure_exits_6(mocker, tmp_path):
    async def fake_generate(prompt, model, size, quality, background, output_format, moderation, filename):
        if prompt == "bad":
            raise ValueError("moderation blocked")
        (tmp_path / filename).write_bytes(b"i")
        return ImageGenerateResult(filename=filename, size=(1024, 1024), format="png", model=model)

    mocker.patch("sanzaru.tools.images_api.generate_image", fake_generate)

    result = CliRunner().invoke(cli, ["image", "generate", "good", "bad", "-o", str(tmp_path) + "/"])

    assert result.exit_code == 6
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    failures = [line for line in lines if not line["ok"]]
    assert len(failures) == 1
    assert failures[0]["input"]["prompt"] == "bad"


@pytest.mark.integration
def test_image_edit_resolves_path_inputs(mocker, tmp_path):
    src = tmp_path / "base.png"
    src.write_bytes(b"x")

    async def fake_edit(
        prompt, input_images, model, mask_filename, size, quality, background, output_format, input_fidelity, filename
    ):
        name = filename or "edited.png"
        (tmp_path / name).write_bytes(b"e")
        return ImageGenerateResult(filename=name, size=(1024, 1024), format="png", model=model)

    edit_spy = mocker.patch("sanzaru.tools.images_api.edit_image", mocker.AsyncMock(side_effect=fake_edit))

    result = CliRunner().invoke(
        cli, ["image", "edit", "make it blue", "--input-image", str(src), "-o", str(tmp_path / "blue.png")]
    )

    assert result.exit_code == 0, result.stderr
    assert edit_spy.call_args.kwargs["input_images"] == ["base.png"]
    parsed = json.loads(result.stdout)
    assert parsed["result"]["file"]["path"] == str(tmp_path / "blue.png")


@pytest.mark.integration
def test_image_prepare_renders_tuple_sizes(mocker, tmp_path):
    src = tmp_path / "hero.png"
    src.write_bytes(b"x")
    mocker.patch(
        "sanzaru.tools.reference.prepare_reference_image",
        mocker.AsyncMock(
            return_value={
                "output_filename": "hero_1280x720.png",
                "original_size": (1536, 1024),
                "target_size": (1280, 720),
                "resize_mode": "crop",
            }
        ),
    )

    result = CliRunner().invoke(cli, ["image", "prepare", str(src), "--size", "1280x720"])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["result"]["original_size"] == [1536, 1024]
    assert parsed["result"]["target_size"] == [1280, 720]


@pytest.mark.integration
def test_image_prepare_reports_the_final_name_across_dirs(mocker, tmp_path):
    """#54 on the image side: the input pins the reference dir, so `-o` into a
    different one makes plan_output stage under a __sanzaru_tmp name. The
    envelope must name the file that survives the move, not the staging one."""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "hero.png").write_bytes(b"x")

    async def fake_prepare(input_filename, target_size, output_filename, resize_mode):
        (in_dir / output_filename).write_bytes(b"prepared")
        return {
            "output_filename": output_filename,
            "original_size": (1536, 1024),
            "target_size": (1280, 720),
            "resize_mode": resize_mode,
        }

    prepare = mocker.patch(
        "sanzaru.tools.reference.prepare_reference_image", mocker.AsyncMock(side_effect=fake_prepare)
    )

    result = CliRunner().invoke(
        cli,
        ["image", "prepare", str(in_dir / "hero.png"), "--size", "1280x720", "-o", str(out_dir / "ref.png")],
    )

    assert result.exit_code == 0, result.stderr
    assert prepare.call_args.kwargs["output_filename"] == "ref__sanzaru_tmp.png"
    parsed = json.loads(result.stdout)
    assert parsed["result"]["output_filename"] == "ref.png"
    assert parsed["result"]["file"]["path"] == str(out_dir / "ref.png")
    assert (out_dir / "ref.png").read_bytes() == b"prepared"


@pytest.mark.integration
def test_top_level_wait_dispatches_mixed_types(mocker):
    mocker.patch("sanzaru.polling.wait_for_video", mocker.AsyncMock(return_value=_video()))
    mocker.patch(
        "sanzaru.polling.wait_for_image",
        mocker.AsyncMock(return_value=_image_response("completed", "resp_b")),
    )

    result = CliRunner().invoke(cli, ["wait", "video_a", "resp_b"])

    assert result.exit_code == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    ids = {line["result"]["id"] for line in lines}
    assert ids == {"video_a", "resp_b"}


@pytest.mark.integration
def test_top_level_wait_unknown_prefix_is_usage_error():
    result = CliRunner().invoke(cli, ["wait", "job_123"])

    assert result.exit_code == 2
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"
    assert "--type" in parsed["error"]["message"]


@pytest.mark.integration
def test_capabilities_reports_structure():
    result = CliRunner().invoke(cli, ["capabilities"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    payload = parsed["result"]
    assert set(payload["features"]) == {"video", "audio", "image"}
    assert all("available" in entry for entry in payload["features"].values())
    assert set(payload["paths_configured"]) == {"video", "reference", "audio"}
    assert isinstance(payload["api_key_present"], bool)
    assert "create" in payload["commands"]["video"]
    assert "generate" in payload["commands"]["image"]
    assert payload["commands"]["capabilities"] == []
