# SPDX-License-Identifier: MIT
"""Integration tests for `sanzaru video` on Higgsfield (tool layer mocked, real envelopes)."""

import json

import pytest
from click.testing import CliRunner

from sanzaru.cli import cli

HF_ID = "hf_d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"
HF_B = "hf_22222222-2222-4222-8222-222222222222"


def make_job(status: str = "queued", job_id: str | None = HF_ID, usd: float | None = 2.31) -> dict:
    """A tools.video.VideoJob."""
    return {
        "id": job_id,
        "status": status,
        "model": "seedance-2.5",
        "slug": "bytedance/seedance-2.5/text-to-video",
        "operation": "text",
        "cost": {
            "usd": usd,
            "credits": None,
            "basis": "local_table",
            "usd_after_discount": None,
            "pricing_description": None,
            "note": None,
        },
        "arguments": {"prompt": "a cat"},
    }


def make_status(status: str = "completed", job_id: str = HF_ID, error: str | None = None) -> dict:
    """A tools.video.VideoStatus."""
    return {
        "id": job_id,
        "status": status,
        "done": status in ("completed", "failed", "nsfw", "canceled"),
        "error": error,
        "video_url": "https://cdn.example/out.mp4" if status == "completed" else None,
    }


# ==================== create ====================


@pytest.mark.integration
def test_create_no_wait_emits_job_envelope_with_cost(mocker):
    create = mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))

    result = CliRunner().invoke(cli, ["video", "create", "a cat stretches"])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["command"] == "video.create"
    assert parsed["result"]["id"] == HF_ID
    assert parsed["result"]["cost"]["usd"] == 2.31
    assert "submitted" in result.stderr and "$2.31" in result.stderr
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "seedance-2.5"
    assert kwargs["dry_run"] is False and kwargs["max_cost_usd"] is None


@pytest.mark.integration
def test_create_one_shot_downloads_to_output(mocker, tmp_path):
    """-o implies --download implies --wait: one command → final file path in envelope."""
    out = tmp_path / "out" / "clip.mp4"
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))
    mocker.patch("sanzaru.polling.wait_for_video", mocker.AsyncMock(return_value=make_status("completed")))

    async def fake_download(video_id, filename=None):
        (tmp_path / "out" / filename).write_bytes(b"video-bytes")
        return {"filename": filename, "format": "mp4"}

    mocker.patch("sanzaru.tools.video.download_video", fake_download)

    result = CliRunner().invoke(cli, ["video", "create", "a cat stretches", "-o", str(out)])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["result"]["status"] == "completed"
    assert parsed["result"]["file"] == {"path": str(out), "format": "mp4", "bytes": len(b"video-bytes")}
    assert parsed["result"]["cost"]["usd"] == 2.31
    assert out.read_bytes() == b"video-bytes"


@pytest.mark.integration
def test_create_dry_run_emits_price_and_never_waits(mocker, tmp_path):
    create = mocker.patch(
        "sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job("dry_run", job_id=None))
    )
    wait = mocker.patch("sanzaru.polling.wait_for_video", mocker.AsyncMock())

    result = CliRunner().invoke(cli, ["video", "create", "a cat", "--dry-run", "-o", str(tmp_path / "x.mp4")])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["result"]["status"] == "dry_run" and parsed["result"]["id"] is None
    assert create.call_args.kwargs["dry_run"] is True
    wait.assert_not_called()
    assert not (tmp_path / "x.mp4").exists()


@pytest.mark.integration
def test_create_over_budget_exits_2(mocker):
    from sanzaru.higgsfield.errors import CostCapExceededError

    mocker.patch(
        "sanzaru.tools.video.create_video",
        mocker.AsyncMock(
            side_effect=CostCapExceededError(
                "estimate $4.62 is over max_cost_usd $1.00",
                estimate_usd=4.62,
                limit_usd=1.0,
                basis="local_table",
                model="seedance-2.5",
            )
        ),
    )

    result = CliRunner().invoke(cli, ["video", "create", "a cat", "--max-cost", "1"])

    assert result.exit_code == 2
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "over_budget"
    assert parsed["estimate_usd"] == 4.62 and parsed["limit_usd"] == 1.0 and parsed["basis"] == "local_table"


@pytest.mark.integration
def test_create_without_hf_key_exits_3(mocker, monkeypatch):
    """The real tool runs: no key means get_higgsfield_client raises before any request."""
    from sanzaru.config import set_higgsfield_client

    monkeypatch.delenv("HF_KEY", raising=False)
    set_higgsfield_client(None)

    result = CliRunner().invoke(cli, ["video", "create", "a cat"])

    assert result.exit_code == 3, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "config"
    assert "HF_KEY" in parsed["error"]["message"]


@pytest.mark.integration
def test_create_maps_flags_onto_the_tool(mocker, tmp_path):
    (tmp_path / "hero.png").write_bytes(b"png")
    (tmp_path / "last.png").write_bytes(b"png")
    create = mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))

    result = CliRunner().invoke(
        cli,
        [
            "video",
            "create",
            "she turns",
            "--model",
            "kling-3.0-pro",
            "--seconds",
            "6",
            "--no-audio",
            "--image",
            str(tmp_path / "hero.png"),
            "--end-image",
            str(tmp_path / "last.png"),
            "--arg",
            "cfg_scale=0.7",
            "--arg",
            "mode=fast",
            "--max-cost",
            "2.5",
        ],
    )

    assert result.exit_code == 0, result.stdout
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "kling-3.0-pro"
    assert kwargs["duration"] == 6
    assert kwargs["audio"] is False
    assert kwargs["reference_image"] == "hero.png" and kwargs["end_image"] == "last.png"
    assert kwargs["extra"] == {"cfg_scale": 0.7, "mode": "fast"}
    assert kwargs["max_cost_usd"] == 2.5


@pytest.mark.integration
def test_create_args_file_merges_and_arg_wins(mocker, tmp_path):
    args = tmp_path / "args.json"
    args.write_text(json.dumps({"bitrate_mode": "high", "output_format": "mov"}))
    create = mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))

    result = CliRunner().invoke(cli, ["video", "create", "x", "--args", f"@{args}", "--arg", "bitrate_mode=standard"])

    assert result.exit_code == 0, result.stdout
    assert create.call_args.kwargs["extra"] == {"bitrate_mode": "standard", "output_format": "mov"}


@pytest.mark.integration
@pytest.mark.parametrize("bad", ["novalue", "=1"])
def test_create_malformed_arg_is_usage(bad):
    result = CliRunner().invoke(cli, ["video", "create", "x", "--arg", bad])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["type"] == "usage"


@pytest.mark.integration
def test_create_accepts_a_catalog_slug_but_not_an_unknown_id(mocker):
    create = mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))

    ok = CliRunner().invoke(cli, ["video", "create", "x", "--model", "minimax/hailuo-2.3/standard/text-to-video"])
    bad = CliRunner().invoke(cli, ["video", "create", "x", "--model", "sora-2"])

    assert ok.exit_code == 0, ok.stdout
    assert create.call_args.kwargs["model"] == "minimax/hailuo-2.3/standard/text-to-video"
    assert bad.exit_code == 2


@pytest.mark.integration
def test_create_duration_and_seconds_alias_must_agree():
    result = CliRunner().invoke(cli, ["video", "create", "x", "--duration", "5", "--seconds", "6"])

    assert result.exit_code == 2


@pytest.mark.integration
def test_create_wait_failed_job_exits_5(mocker):
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))
    mocker.patch(
        "sanzaru.polling.wait_for_video",
        mocker.AsyncMock(return_value=make_status("failed", error="Generation failed")),
    )

    result = CliRunner().invoke(cli, ["video", "create", "a cat", "--wait"])

    assert result.exit_code == 5
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "job_failed"
    assert parsed["id"] == HF_ID and parsed["details"] == "Generation failed"


@pytest.mark.integration
def test_create_wait_nsfw_exits_5_with_its_own_message(mocker):
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(return_value=make_job()))
    mocker.patch("sanzaru.polling.wait_for_video", mocker.AsyncMock(return_value=make_status("nsfw")))

    result = CliRunner().invoke(cli, ["video", "create", "a cat", "--wait"])

    assert result.exit_code == 5
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "nsfw"
    assert "content moderation" in parsed["error"]["message"] and "not charged" in parsed["error"]["message"]


# ==================== --retry-busy ====================


def _busy():
    from sanzaru.higgsfield.errors import error_from_response

    return error_from_response(400, "Maximum number of concurrent requests (4) has been reached")


@pytest.mark.integration
def test_retry_busy_retries_only_the_concurrency_error(mocker):
    mocker.patch("sanzaru.cli.video.anyio.sleep", mocker.AsyncMock())
    create = mocker.patch(
        "sanzaru.tools.video.create_video", mocker.AsyncMock(side_effect=[_busy(), _busy(), make_job()])
    )

    result = CliRunner().invoke(cli, ["video", "create", "x", "--retry-busy", "5m"])

    assert result.exit_code == 0, result.stdout
    assert create.await_count == 3
    assert "concurrency limit" in result.stderr


@pytest.mark.integration
def test_retry_busy_does_not_retry_other_errors(mocker):
    from sanzaru.higgsfield.errors import error_from_response

    mocker.patch("sanzaru.cli.video.anyio.sleep", mocker.AsyncMock())
    create = mocker.patch(
        "sanzaru.tools.video.create_video", mocker.AsyncMock(side_effect=error_from_response(500, "boom"))
    )

    result = CliRunner().invoke(cli, ["video", "create", "x", "--retry-busy", "5m"])

    assert result.exit_code == 1
    assert create.await_count == 1


@pytest.mark.integration
def test_concurrency_error_without_retry_busy_is_retryable_envelope(mocker):
    mocker.patch("sanzaru.tools.video.create_video", mocker.AsyncMock(side_effect=_busy()))

    result = CliRunner().invoke(cli, ["video", "create", "x"])

    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "concurrency_limit" and parsed["retryable"] is True


# ==================== edit / extend ====================


@pytest.mark.integration
def test_edit_resolves_a_local_source_and_refs(mocker, tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"mp4")
    (tmp_path / "style.png").write_bytes(b"png")
    edit = mocker.patch("sanzaru.tools.video.edit_video", mocker.AsyncMock(return_value=make_job()))

    result = CliRunner().invoke(
        cli,
        [
            "video",
            "edit",
            str(tmp_path / "clip.mp4"),
            "make it noir",
            "--ref",
            str(tmp_path / "style.png"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    kwargs = edit.call_args.kwargs
    assert kwargs["source_video"] == "clip.mp4"
    assert kwargs["reference_images"] == ["style.png"]
    assert kwargs["prompt"] == "make it noir"
    assert kwargs["dry_run"] is True


@pytest.mark.integration
def test_extend_passes_an_hf_id_source_through(mocker):
    extend = mocker.patch("sanzaru.tools.video.extend_video", mocker.AsyncMock(return_value=make_job()))

    result = CliRunner().invoke(cli, ["video", "extend", HF_B, "the camera keeps rising", "--duration", "6"])

    assert result.exit_code == 0, result.stdout
    kwargs = extend.call_args.kwargs
    assert kwargs["source_video"] == HF_B
    assert kwargs["duration"] == 6
    assert kwargs["reference_images"] is None


# ==================== wait / status / download / cancel ====================


@pytest.mark.integration
def test_wait_timeout_exits_4_with_resume(mocker):
    from sanzaru.polling import WaitTimeoutError

    mocker.patch(
        "sanzaru.polling.wait_for_video",
        mocker.AsyncMock(
            side_effect=WaitTimeoutError(f"Video job {HF_ID} still running after 100s", make_status("in_progress"))
        ),
    )

    result = CliRunner().invoke(cli, ["video", "wait", HF_ID, "--timeout", "100s"])

    assert result.exit_code == 4
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "timeout"
    assert parsed["last_status"] == "in_progress"
    assert f"sanzaru video wait {HF_ID}" in parsed["resume"]


@pytest.mark.integration
def test_wait_multi_id_streams_jsonl_and_exits_partial(mocker):
    async def fake_wait(video_id, **kwargs):
        return make_status("completed" if video_id == HF_ID else "failed", job_id=video_id)

    mocker.patch("sanzaru.polling.wait_for_video", fake_wait)

    result = CliRunner().invoke(cli, ["video", "wait", HF_ID, HF_B])

    assert result.exit_code == 6
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    by_ok = {line["ok"]: line for line in lines}
    assert by_ok[True]["result"]["id"] == HF_ID
    assert by_ok[False]["error"]["type"] == "job_failed" and by_ok[False]["id"] == HF_B


@pytest.mark.integration
def test_status_never_blocks(mocker):
    status = mocker.patch(
        "sanzaru.tools.video.get_video_status", mocker.AsyncMock(return_value=make_status("in_progress"))
    )

    result = CliRunner().invoke(cli, ["video", "status", HF_ID])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"]["status"] == "in_progress"
    status.assert_called_once_with(HF_ID)


@pytest.mark.integration
def test_download_into_directory(mocker, tmp_path):
    async def fake_download(video_id, filename=None):
        name = filename or f"{video_id}.mp4"
        (tmp_path / name).write_bytes(b"x")
        return {"filename": name, "format": "mp4"}

    mocker.patch("sanzaru.tools.video.download_video", fake_download)

    result = CliRunner().invoke(cli, ["video", "download", HF_ID, "-o", str(tmp_path) + "/"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["result"]["file"]["path"] == str(tmp_path / f"{HF_ID}.mp4")


@pytest.mark.integration
def test_cancel_emits_result(mocker):
    mocker.patch("sanzaru.tools.video.cancel_video", mocker.AsyncMock(return_value={"id": HF_ID, "canceled": True}))

    result = CliRunner().invoke(cli, ["video", "cancel", HF_ID])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"] == {"id": HF_ID, "canceled": True}


@pytest.mark.integration
@pytest.mark.parametrize("argv", [["video", "remix", HF_ID, "x"], ["video", "list"], ["video", "delete", HF_ID]])
def test_sora_only_commands_are_gone(argv):
    assert CliRunner().invoke(cli, argv).exit_code == 2


# ==================== models / files ====================


@pytest.mark.integration
def test_models_lists_the_curated_table():
    from sanzaru.video_models import DEFAULT_VIDEO_MODEL, VIDEO_MODEL_IDS

    result = CliRunner().invoke(cli, ["video", "models"])

    assert result.exit_code == 0, result.stdout
    curated = json.loads(result.stdout)["result"]["curated"]
    assert [m["id"] for m in curated] == list(VIDEO_MODEL_IDS)
    assert [m["id"] for m in curated if m["default"]] == [DEFAULT_VIDEO_MODEL]
    assert "catalog" not in json.loads(result.stdout)["result"]


@pytest.mark.integration
def test_models_catalog_lists_video_slugs(mocker):
    client = mocker.Mock()
    client.models = mocker.AsyncMock(
        return_value=[
            {
                "slug": "bytedance/seedance-2.5/text-to-video",
                "title": "Seedance",
                "output_type": "video",
                "operation_type": ["text2video"],
            },
            {"slug": "higgsfield-ai/soul/standard", "title": "Soul", "output_type": "image", "operation_type": []},
        ]
    )
    mocker.patch("sanzaru.config.get_higgsfield_client", return_value=client)

    result = CliRunner().invoke(cli, ["video", "models", "--catalog"])

    assert result.exit_code == 0, result.stdout
    catalog = json.loads(result.stdout)["result"]["catalog"]
    assert [m["slug"] for m in catalog] == ["bytedance/seedance-2.5/text-to-video"]


@pytest.mark.integration
def test_files_lists_local(mocker):
    mocker.patch("sanzaru.tools.video.list_local_videos", mocker.AsyncMock(return_value={"data": []}))

    result = CliRunner().invoke(cli, ["video", "files"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"] == {"data": []}


# ==================== id guard, end to end ====================

_HOSTILE_ID = "../files/file-XYZ/content?"


@pytest.mark.integration
@pytest.mark.parametrize(
    "argv",
    [
        ["video", "status", _HOSTILE_ID],
        ["video", "download", _HOSTILE_ID],
        ["video", "cancel", _HOSTILE_ID],
        ["video", "wait", _HOSTILE_ID],
        ["wait", "--type", "video", _HOSTILE_ID],
    ],
    ids=["status", "download", "cancel", "wait", "top-level-wait"],
)
def test_hostile_video_id_is_a_usage_error(mocker, argv):
    """The real tool runs: the id guard's ValueError reaches the shell as exit 2 before any request."""
    get_client = mocker.patch("sanzaru.tools.video.get_higgsfield_client")

    result = CliRunner().invoke(cli, argv)

    assert result.exit_code == 2, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"
    client = get_client.return_value
    client.status.assert_not_called()
    client.cancel.assert_not_called()


@pytest.mark.integration
def test_video_wait_on_a_sora_id_explains_the_retirement(mocker):
    mocker.patch("sanzaru.tools.video.get_higgsfield_client")

    result = CliRunner().invoke(cli, ["video", "wait", "video_abc123"])

    assert result.exit_code == 2
    assert "retired the Videos API" in json.loads(result.stdout)["error"]["message"]


# ==================== capabilities ====================


@pytest.mark.integration
def test_capabilities_reports_missing_hf_key(monkeypatch):
    monkeypatch.delenv("HF_KEY", raising=False)

    result = CliRunner().invoke(cli, ["capabilities"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)["result"]
    generation = payload["features"]["video"]["generation"]
    assert generation["available"] is False and "HF_KEY" in generation["reason"]
    assert payload["higgsfield_key_present"] is False
    assert payload["video_providers"] == {"higgsfield": False}


# ==================== _classify: Higgsfield errors ====================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "detail", "error_type", "exit_code"),
    [
        (401, "Invalid credentials", "config", 3),
        (403, "Insufficient credits", "insufficient_credits", 1),
        (404, "Request not found", "not_found", 1),
        (400, "duration: 99 is greater than the maximum of 15", "usage", 2),
        (422, "Request body validation failed", "usage", 2),
        (423, "Model is temporarily blocked", "model_unavailable", 1),
        (503, "Model is disabled", "model_unavailable", 1),
        (500, "Unexpected server error", "api_error", 1),
        (400, "Maximum number of concurrent requests (4) has been reached", "concurrency_limit", 1),
    ],
)
def test_classify_maps_higgsfield_errors(status, detail, error_type, exit_code):
    from sanzaru.cli._runtime import _classify
    from sanzaru.higgsfield.errors import error_from_response

    error = _classify(error_from_response(status, detail))

    assert (error.error_type, error.exit_code) == (error_type, exit_code)


@pytest.mark.unit
def test_classify_insufficient_credits_names_the_separate_api_balance():
    from sanzaru.cli._runtime import _classify
    from sanzaru.higgsfield.errors import error_from_response

    error = _classify(error_from_response(403, "Insufficient credits"))

    assert "separate" in str(error) and "subscription" in str(error)


@pytest.mark.unit
def test_classify_ambiguous_submit_flags_maybe_submitted():
    from sanzaru.cli._runtime import _classify
    from sanzaru.higgsfield.errors import HiggsfieldAPIError

    error = _classify(
        HiggsfieldAPIError("may have been created", status_code=None, detail="ReadTimeout", kind="ambiguous_submit")
    )

    assert error.error_type == "api_error" and error.extra == {"maybe_submitted": True}


@pytest.mark.unit
def test_classify_over_budget_is_usage_with_amounts():
    from sanzaru.cli._runtime import _classify
    from sanzaru.higgsfield.errors import UnpricedVideoError

    error = _classify(
        UnpricedVideoError("cannot price", estimate_usd=None, limit_usd=1.0, basis="unpriced", model="x/y")
    )

    assert (error.error_type, error.exit_code) == ("over_budget", 2)
    assert error.extra == {"estimate_usd": None, "limit_usd": 1.0, "basis": "unpriced"}
