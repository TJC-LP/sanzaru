"""Tests for `sanzaru podcast rundown` and `sanzaru podcast simulate`.

Mocked at the `sanzaru.tools.simulate_podcast` / `audio.realtime.rundown` layer,
so nothing here can spend money — including the tests that assert a dry run
spends nothing.
"""

import json
import sys

import pytest
from click.testing import CliRunner

from sanzaru.cli import cli

if sys.version_info < (3, 11):  # pragma: no cover - 3.11+ has it as a builtin
    from exceptiongroup import BaseExceptionGroup

pytestmark = pytest.mark.integration


RUNDOWN = {
    "title": "Stitch",
    "premise": "the plumbing",
    "style": "dry",
    "hosts": [
        {"id": "avery", "name": "Avery", "voice": "marin", "persona": "You host."},
        {"id": "rory", "name": "Rory", "voice": "cedar", "persona": "You engineer."},
    ],
    "acts": [
        {
            "id": "act1",
            "title": "One",
            "topic": "first",
            "talking_points": ["a"],
            "target_seconds": 60.0,
            "max_turns": 5,
            "prior_context": "",
            "upcoming": "",
            "handoff": "on the question",
        },
        {
            "id": "act2",
            "title": "Two",
            "topic": "second",
            "talking_points": ["b"],
            "target_seconds": 60.0,
            "max_turns": 5,
            "prior_context": "act 1 did first",
            "upcoming": "",
            "handoff": "",
        },
    ],
}


def _envelope(output: str) -> dict:
    """Parse the single JSON envelope from stdout."""
    return json.loads(output.strip().splitlines()[-1])


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_result(mocker):
    """A SimulatedPodcastResult stand-in, built from the real model."""
    from sanzaru.audio.realtime.types import ActSummary, RealtimeUsage, Rundown
    from sanzaru.tools.simulate_podcast import CostReport, SimulatedPodcastResult

    def build(**overrides):
        defaults = {
            "run_id": "abc12345",
            "title": "Stitch",
            "output_file": "stitch_abc12345.mp3",
            "duration_seconds": 120.0,
            "turn_count": 10,
            "hosts": ["Avery", "Rory"],
            "acts": [
                ActSummary(act_id="act1", title="One", turns=5, seconds=60.0, stop_reason="complete"),
                ActSummary(act_id="act2", title="Two", turns=5, seconds=60.0, stop_reason="complete"),
            ],
            "cost": CostReport(usd=0.42, usage=RealtimeUsage(output_audio_tokens=2400)),
            "transcript": "**Avery:** hi",
            "rundown": Rundown.model_validate(RUNDOWN),
            "resume_command": "sanzaru podcast simulate --resume abc12345",
        }
        defaults.update(overrides)
        return SimulatedPodcastResult(**defaults)

    return build


# ---------- podcast rundown ----------


class TestPodcastRundown:
    def test_emits_a_rundown_envelope(self, runner, mocker):
        from sanzaru.audio.realtime.types import Rundown

        planned = Rundown.model_validate(RUNDOWN)
        mocker.patch("sanzaru.audio.realtime.rundown.plan_rundown", return_value=planned)

        result = runner.invoke(cli, ["podcast", "rundown", "a premise", "--acts", "2", "-m", "2"])
        assert result.exit_code == 0
        envelope = _envelope(result.stdout)
        assert envelope["ok"] is True
        assert envelope["command"] == "podcast.rundown"
        assert [a["id"] for a in envelope["result"]["acts"]] == ["act1", "act2"]

    def test_writes_a_file_that_simulate_can_read_back(self, runner, mocker, tmp_path):
        from sanzaru.audio.realtime.types import Rundown

        mocker.patch("sanzaru.audio.realtime.rundown.plan_rundown", return_value=Rundown.model_validate(RUNDOWN))
        target = tmp_path / "rundown.json"

        result = runner.invoke(cli, ["podcast", "rundown", "a premise", "-o", str(target)])
        assert result.exit_code == 0
        written = json.loads(target.read_text())
        # The file is the pure rundown — no envelope keys leak into it.
        assert "file" not in written
        assert Rundown.model_validate(written).title == "Stitch"

    def test_passes_hosts_through(self, runner, mocker):
        from sanzaru.audio.realtime.types import Rundown

        patched = mocker.patch(
            "sanzaru.audio.realtime.rundown.plan_rundown", return_value=Rundown.model_validate(RUNDOWN)
        )
        result = runner.invoke(
            cli, ["podcast", "rundown", "p", "--host", "Avery:marin:You host.", "--host", "Rory:cedar:You engineer."]
        )
        assert result.exit_code == 0
        request = patched.call_args.args[0]
        assert [h.name for h in request.hosts] == ["Avery", "Rory"]
        assert [h.voice for h in request.hosts] == ["marin", "cedar"]

    def test_rejects_an_unknown_voice(self, runner):
        result = runner.invoke(cli, ["podcast", "rundown", "p", "--host", "Avery:notavoice:x"])
        assert result.exit_code == 2
        assert _envelope(result.stdout)["error"]["type"] == "usage"

    def test_rejects_a_nameless_host(self, runner):
        result = runner.invoke(cli, ["podcast", "rundown", "p", "--host", ":marin:x"])
        assert result.exit_code == 2


# ---------- podcast simulate ----------


class TestPodcastSimulate:
    def test_dry_run_spends_nothing_and_needs_no_output_dir(self, runner, mocker, fake_result):
        patched = mocker.patch(
            "sanzaru.tools.simulate_podcast.simulate_podcast",
            return_value=fake_result(dry_run=True, output_file=""),
        )
        result = runner.invoke(cli, ["podcast", "simulate", "-p", "a premise", "--dry-run"])
        assert result.exit_code == 0
        assert patched.call_args.args[0].dry_run is True
        envelope = _envelope(result.stdout)
        assert envelope["result"]["dry_run"] is True
        assert "nothing was recorded" in result.stderr

    def test_a_bare_rundown_is_wrapped_as_the_brief_rundown(self, runner, mocker, fake_result, tmp_path):
        patched = mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        path = tmp_path / "r.json"
        path.write_text(json.dumps(RUNDOWN))

        result = runner.invoke(cli, ["podcast", "simulate", f"@{path}", "--dry-run"])
        assert result.exit_code == 0
        brief = patched.call_args.args[0]
        # `acts` is a list in a rundown and an int in a brief — that is the tell.
        assert brief.rundown is not None
        assert brief.rundown.title == "Stitch"

    def test_a_full_brief_is_used_as_is(self, runner, mocker, fake_result, tmp_path):
        patched = mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"premise": "from the brief", "acts": 5, "target_minutes": 20}))

        result = runner.invoke(cli, ["podcast", "simulate", f"@{path}", "--dry-run"])
        assert result.exit_code == 0
        brief = patched.call_args.args[0]
        assert brief.premise == "from the brief"
        assert brief.acts == 5

    def test_flags_override_the_brief(self, runner, mocker, fake_result, tmp_path):
        patched = mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"premise": "p", "acts": 5, "model": "gpt-realtime-2.1"}))

        result = runner.invoke(
            cli,
            ["podcast", "simulate", f"@{path}", "--acts", "2", "--model", "gpt-realtime-2.1-mini", "--dry-run"],
        )
        assert result.exit_code == 0
        brief = patched.call_args.args[0]
        assert brief.acts == 2
        assert brief.model == "gpt-realtime-2.1-mini"

    def test_nothing_to_record_is_a_usage_error(self, runner):
        result = runner.invoke(cli, ["podcast", "simulate"])
        assert result.exit_code == 2
        assert "nothing to record" in _envelope(result.stdout)["error"]["message"]

    def test_malformed_brief_json_is_a_usage_error(self, runner):
        result = runner.invoke(cli, ["podcast", "simulate", "{not json"])
        assert result.exit_code == 2
        assert _envelope(result.stdout)["error"]["type"] == "usage"

    def test_a_brief_that_fails_validation_is_a_usage_error(self, runner, tmp_path):
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"premise": "p", "acts": "not a number"}))
        result = runner.invoke(cli, ["podcast", "simulate", f"@{path}", "--dry-run"])
        assert result.exit_code == 2

    def test_prints_the_run_id_before_recording(self, runner, mocker, fake_result, tmp_path):
        mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        mocker.patch("sanzaru.cli.podcast.finalize_output", return_value=str(tmp_path / "ep.mp3"))

        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 0
        # The id has to reach the user before any spend, or an interrupt is unresumable.
        assert "resume with: sanzaru podcast simulate --resume" in result.stderr

    def test_resume_sets_the_run_id(self, runner, mocker, fake_result, tmp_path):
        patched = mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        mocker.patch("sanzaru.cli.podcast.finalize_output", return_value=str(tmp_path / "ep.mp3"))

        result = runner.invoke(cli, ["podcast", "simulate", "--resume", "abc12345", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 0
        brief = patched.call_args.args[0]
        assert brief.resume is True
        assert brief.run_id == "abc12345"

    def test_cost_ceiling_exits_partial_with_a_resume_command(self, runner, mocker, tmp_path):
        from sanzaru.exceptions import CostCeilingError

        mocker.patch(
            "sanzaru.tools.simulate_podcast.simulate_podcast",
            side_effect=CostCeilingError(
                "cost ceiling reached: $2.10 spent of $2.00 limit",
                spent_usd=2.10,
                limit_usd=2.00,
                completed_acts=["act1"],
            ),
        )
        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 6
        envelope = _envelope(result.stdout)
        assert envelope["error"]["type"] == "cost_limit"
        assert envelope["completed_acts"] == ["act1"]
        assert envelope["resume"].startswith("sanzaru podcast simulate --resume ")

    def test_cost_ceiling_inside_a_task_group_is_still_recognised(self, runner, mocker, tmp_path):
        from sanzaru.exceptions import CostCeilingError

        ceiling = CostCeilingError("over", spent_usd=1.0, limit_usd=0.5, completed_acts=[])
        mocker.patch(
            "sanzaru.tools.simulate_podcast.simulate_podcast",
            side_effect=BaseExceptionGroup("unhandled errors in a TaskGroup", [ceiling, ceiling]),
        )
        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 6
        assert _envelope(result.stdout)["error"]["type"] == "cost_limit"

    def test_reports_qc_flags_on_stderr(self, runner, mocker, fake_result, tmp_path):
        from sanzaru.audio.realtime.qc import ActVerdict, QCReport

        report = QCReport(
            verdict="warn",
            flagged_acts=["act2"],
            acts=[ActVerdict(act_id="act2", similarity=0.5)],
        )
        mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result(qc=report))
        mocker.patch("sanzaru.cli.podcast.finalize_output", return_value=str(tmp_path / "ep.mp3"))

        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 0
        assert "qc warn: act2" in result.stderr
        assert "--qc-retry" in result.stderr

    def test_warns_when_turns_were_cut_short(self, runner, mocker, fake_result, tmp_path):
        from sanzaru.audio.realtime.types import ActSummary

        acts = [
            ActSummary(act_id="act1", title="One", turns=5, seconds=60.0, stop_reason="complete", truncated_turns=4)
        ]
        mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result(acts=acts))
        mocker.patch("sanzaru.cli.podcast.finalize_output", return_value=str(tmp_path / "ep.mp3"))

        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert "raise --turn-tokens" in result.stderr

    def test_quiet_suppresses_stderr_but_not_the_envelope(self, runner, mocker, fake_result, tmp_path):
        mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        mocker.patch("sanzaru.cli.podcast.finalize_output", return_value=str(tmp_path / "ep.mp3"))

        result = runner.invoke(cli, ["-q", "podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 0
        assert result.stderr.strip() == ""
        assert _envelope(result.stdout)["ok"] is True

    def test_envelope_carries_the_final_path(self, runner, mocker, fake_result, tmp_path):
        mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        mocker.patch("sanzaru.cli.podcast.finalize_output", return_value=str(tmp_path / "ep.mp3"))

        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert _envelope(result.stdout)["result"]["file"]["path"] == str(tmp_path / "ep.mp3")


class TestHelp:
    """The help text is the schema an agent reads before spending money."""

    def test_simulate_help_leads_with_the_cost_warning(self, runner):
        result = runner.invoke(cli, ["podcast", "simulate", "--help"])
        assert result.exit_code == 0
        assert "--dry-run first" in result.output
        assert "--resume" in result.output

    def test_podcast_group_lists_all_three_verbs(self, runner):
        result = runner.invoke(cli, ["podcast", "--help"])
        assert {"rundown", "simulate", "generate"} <= set(result.output.split())
