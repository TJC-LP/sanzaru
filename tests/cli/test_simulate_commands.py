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

    def test_a_host_who_names_no_voice_leaves_the_choice_open(self, runner, mocker):
        """`--host NAME` must reach pre-production with no voice at all.

        This path is the reason the marin-for-everyone bug stayed invisible: the
        CLI has always sent an empty voice, so it alone got distinct voices while
        every other way in collapsed onto the default.
        """
        from sanzaru.audio.realtime.rundown import assign_voices
        from sanzaru.audio.realtime.types import Rundown

        patched = mocker.patch(
            "sanzaru.audio.realtime.rundown.plan_rundown", return_value=Rundown.model_validate(RUNDOWN)
        )
        result = runner.invoke(cli, ["podcast", "rundown", "p", "--host", "Avery::You host.", "--host", "Rory"])
        assert result.exit_code == 0
        request = patched.call_args.args[0]
        assert [h.voice for h in request.hosts] == ["", ""]
        assert [h.voice for h in assign_voices(request.hosts)] == ["marin", "cedar"]

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

    def test_a_run_id_in_a_rundown_brief_is_lifted_into_the_run(self, runner, mocker, fake_result, tmp_path):
        # SKILL docs told callers to set a top-level "run_id" in the rundown; the
        # wrapper dropped it and the run recorded under a minted id that only
        # ever appeared on stderr.
        patched = mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        path = tmp_path / "r.json"
        path.write_text(json.dumps({**RUNDOWN, "run_id": "ep1-showcase"}))

        result = runner.invoke(cli, ["podcast", "simulate", f"@{path}", "--dry-run"])
        assert result.exit_code == 0
        brief = patched.call_args.args[0]
        assert brief.rundown is not None
        assert brief.run_id == "ep1-showcase"

    def test_the_run_id_flag_beats_the_brief(self, runner, mocker, fake_result, tmp_path):
        patched = mocker.patch("sanzaru.tools.simulate_podcast.simulate_podcast", return_value=fake_result())
        path = tmp_path / "r.json"
        path.write_text(json.dumps({**RUNDOWN, "run_id": "from-the-file"}))

        result = runner.invoke(cli, ["podcast", "simulate", f"@{path}", "--run-id", "from-the-flag", "--dry-run"])
        assert result.exit_code == 0
        assert patched.call_args.args[0].run_id == "from-the-flag"

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

    def test_the_ceiling_resume_command_raises_the_ceiling(self, runner, mocker, tmp_path):
        """The printed command has to be one that can finish.

        A resume restores the run's ceiling and replays the spend already on
        disk against it, so `--resume <id>` alone re-records the missing acts,
        trips at the same total, and prints itself again — a loop that bills
        every lap.
        """
        from sanzaru.exceptions import CostCeilingError

        mocker.patch(
            "sanzaru.tools.simulate_podcast.simulate_podcast",
            side_effect=CostCeilingError(
                "cost ceiling reached: $2.10 spent of $2.00 limit",
                spent_usd=2.10,
                limit_usd=2.00,
                completed_acts=["act1"],
                suggested_limit_usd=2.63,
            ),
        )
        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 6
        envelope = _envelope(result.stdout)
        assert envelope["resume"].endswith("--max-cost 2.63")
        assert envelope["suggested_limit_usd"] == 2.63
        assert "--max-cost" in envelope["error"]["message"]

    def test_any_other_failure_still_hands_back_the_run_id(self, runner, mocker, tmp_path):
        """Acts checkpointed before the failure are only reachable by run id.

        A stalled turn cancels the acts recording alongside it and fails the
        run; without this the envelope carried no way back to what survived.
        """
        from sanzaru.exceptions import RealtimeAPIError

        mocker.patch(
            "sanzaru.tools.simulate_podcast.simulate_podcast",
            side_effect=RealtimeAPIError("act2: turn 1 made no progress for 90s - stalled"),
        )
        result = runner.invoke(cli, ["podcast", "simulate", "-p", "p", "-o", str(tmp_path / "ep.mp3")])
        assert result.exit_code == 1
        envelope = _envelope(result.stdout)
        assert envelope["error"]["type"] == "api_error"
        assert envelope["resume"].startswith("sanzaru podcast simulate --resume ")
        assert envelope["run_id"] == envelope["resume"].split()[-1]

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


@pytest.mark.audio
class TestSimulateRecovery:
    """The one pair of flags `docs/cli.md` documents together, end to end.

    Everything else in this file stubs `simulate_podcast` itself; this class
    deliberately does not, because the bug it guards lives in the seam between
    the CLI's `-o` path override and the tool's checkpoint bookkeeping. Only
    `run_act` is stubbed, so no session is opened and nothing is spent.
    """

    @pytest.fixture
    def media_dir(self, tmp_path, monkeypatch):
        from sanzaru.config import get_path

        path = tmp_path / "media"
        path.mkdir()
        monkeypatch.setenv("AUDIO_PATH", str(path))
        get_path.cache_clear()
        yield path
        get_path.cache_clear()

    @pytest.fixture
    def stub_run_act(self, mocker):
        """Replace recording with instant two-turn acts; returns the call log."""
        from sanzaru.audio.realtime.types import ActResult, RealtimeUsage, Turn, TurnAudio

        recorded: list[str] = []

        async def fake_run_act(brief, hosts, settings, **kwargs):
            recorded.append(brief.id)
            audio = [
                TurnAudio(
                    turn=Turn(act_id=brief.id, index=i, speaker_id="avery", speaker_name="Avery", text="hi", seconds=1),
                    pcm=b"\x10\x20" * 24000,
                )
                for i in range(2)
            ]
            return ActResult(act_id=brief.id, audio=audio, usage=RealtimeUsage(), stop_reason="complete")

        mocker.patch("sanzaru.tools.simulate_podcast.run_act", fake_run_act)
        return recorded

    @staticmethod
    def _run_id(stderr: str) -> str:
        return stderr.split("--resume ")[1].split()[0].strip()

    def test_recording_with_o_then_resuming_without_it(self, runner, tmp_path, media_dir, stub_run_act):
        rundown_file = tmp_path / "r.json"
        rundown_file.write_text(json.dumps(RUNDOWN))
        out = tmp_path / "out" / "ep1.mp3"

        first = runner.invoke(cli, ["podcast", "simulate", f"@{rundown_file}", "--no-qc", "-o", str(out)])
        assert first.exit_code == 0, first.stderr
        assert out.exists()
        assert stub_run_act == ["act1", "act2"]
        # -o carries the episode, not 2N+1 intermediates.
        assert [p.name for p in out.parent.iterdir()] == ["ep1.mp3"]

        run_id = self._run_id(first.stderr)
        stub_run_act.clear()
        # Verbatim the command the CLI just printed: no -o anywhere.
        second = runner.invoke(cli, ["podcast", "simulate", "--resume", run_id, "--no-qc"])
        assert second.exit_code == 0, second.stderr
        assert stub_run_act == []
        assert all(act["reused"] for act in _envelope(second.stdout)["result"]["acts"])

    def test_a_hand_written_rundown_without_voices_records_in_distinct_ones(
        self, runner, tmp_path, media_dir, stub_run_act
    ):
        """The workflow the docs push: plan, hand-edit the JSON, record it.

        Omitting `voice` is the obvious thing to do in that file, and it used to
        put the whole cast in one voice — audible only after the recording.
        """
        voiceless = json.loads(json.dumps(RUNDOWN))
        for host in voiceless["hosts"]:
            del host["voice"]
        rundown_file = tmp_path / "r.json"
        rundown_file.write_text(json.dumps(voiceless))

        result = runner.invoke(
            cli, ["podcast", "simulate", f"@{rundown_file}", "--no-qc", "-o", str(tmp_path / "out" / "ep.mp3")]
        )
        assert result.exit_code == 0, result.stderr
        voices = [h["voice"] for h in _envelope(result.stdout)["result"]["rundown"]["hosts"]]
        assert voices == ["marin", "cedar"]

    def test_a_bare_resume_keeps_the_container_the_first_run_asked_for(self, runner, tmp_path, media_dir, stub_run_act):
        """Every simulate option defaults to None so the manifest can fill it in.

        A click-supplied `"mp3"` is indistinguishable from the user typing it,
        so the resume the CLI itself prints would silently downgrade the
        container. No `-o` here on purpose: an `-o` basename goes into the brief
        as `filename`, which the manifest then restores and which would mask the
        format entirely.
        """
        rundown_file = tmp_path / "r.json"
        rundown_file.write_text(json.dumps(RUNDOWN))

        first = runner.invoke(cli, ["podcast", "simulate", f"@{rundown_file}", "--no-qc", "--format", "wav"])
        assert first.exit_code == 0, first.stderr
        assert _envelope(first.stdout)["result"]["file"]["path"].endswith(".wav")

        run_id = self._run_id(first.stderr)
        second = runner.invoke(cli, ["podcast", "simulate", "--resume", run_id, "--no-qc"])
        assert second.exit_code == 0, second.stderr
        assert _envelope(second.stdout)["result"]["file"]["path"].endswith(".wav")

    def test_a_bare_resume_keeps_the_bitrate_the_first_run_asked_for(self, runner, tmp_path, media_dir, stub_run_act):
        rundown_file = tmp_path / "r.json"
        rundown_file.write_text(json.dumps(RUNDOWN))

        first = runner.invoke(
            cli,
            [
                "podcast",
                "simulate",
                f"@{rundown_file}",
                "--no-qc",
                "--bitrate",
                "320k",
                "-o",
                str(tmp_path / "out" / "ep1.mp3"),
            ],
        )
        assert first.exit_code == 0, first.stderr

        run_id = self._run_id(first.stderr)
        second = runner.invoke(cli, ["podcast", "simulate", "--resume", run_id, "--no-qc"])
        assert second.exit_code == 0, second.stderr
        # The resumed run rewrites the manifest from its own effective brief.
        manifest = json.loads((media_dir / f"simrun_{run_id}.json").read_text())
        assert manifest["brief"]["output_bitrate"] == "320k"

    def test_the_ceiling_from_the_first_run_still_applies_on_resume(self, runner, tmp_path, media_dir, stub_run_act):
        rundown_file = tmp_path / "r.json"
        rundown_file.write_text(json.dumps(RUNDOWN))

        first = runner.invoke(
            cli,
            [
                "podcast",
                "simulate",
                f"@{rundown_file}",
                "--no-qc",
                "--max-cost",
                "9.5",
                "-o",
                str(tmp_path / "out" / "ep1.mp3"),
            ],
        )
        assert first.exit_code == 0, first.stderr

        run_id = self._run_id(first.stderr)
        second = runner.invoke(cli, ["podcast", "simulate", "--resume", run_id, "--no-qc"])
        assert second.exit_code == 0, second.stderr
        # Following the printed hint must not silently re-run uncapped.
        assert _envelope(second.stdout)["result"]["cost"]["limit_usd"] == 9.5


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
