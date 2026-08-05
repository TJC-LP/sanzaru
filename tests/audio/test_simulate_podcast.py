"""Tests for the simulate_podcast tool: mixdown, checkpointing, resume, budgets.

Recording itself is stubbed at the `run_act` seam — floor control is covered in
test_realtime_producer.py, and what matters here is everything around it: that
acts land on disk before anything can go wrong, that a resumed run reuses them,
that stems line up with the master, and that a cost ceiling leaves the finished
work intact.
"""

import json
import pathlib

import pytest
from pydantic import ValidationError

from sanzaru.audio.realtime import mixdown
from sanzaru.audio.realtime.pricing import ModelPrices, prices_for, project_usage, usage_cost
from sanzaru.audio.realtime.types import (
    REALTIME_SAMPLE_RATE,
    ActBrief,
    ActResult,
    HostSpec,
    RealtimeUsage,
    Rundown,
    Turn,
    TurnAudio,
    pcm_seconds,
    pcm_silence,
)
from sanzaru.tools import simulate_podcast as sim

pytestmark = pytest.mark.audio


def _pcm(seconds: float, marker: bytes = b"\x10\x20") -> bytes:
    frames = int(seconds * REALTIME_SAMPLE_RATE)
    return (marker * frames)[: frames * 2]


@pytest.fixture
def rundown():
    return Rundown(
        title="Stitch Test",
        premise="the plumbing",
        hosts=[
            HostSpec(id="avery", name="Avery", voice="marin", persona="You host."),
            HostSpec(id="rory", name="Rory", voice="cedar", persona="You engineer."),
        ],
        acts=[
            ActBrief(id=f"act{i + 1}", title=f"Act {i + 1}", topic=f"topic {i + 1}", target_seconds=4.0, max_turns=2)
            for i in range(3)
        ],
    )


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    """Point the audio storage at a temp dir for one test."""
    from sanzaru.config import get_path

    path = tmp_path / "audio"
    path.mkdir()
    monkeypatch.setenv("AUDIO_PATH", str(path))
    get_path.cache_clear()
    yield path
    get_path.cache_clear()


def _fake_act(act_id: str, *, seconds: float = 2.0, turns: int = 2) -> ActResult:
    """A recorded act with distinguishable per-speaker audio."""
    speakers = [("avery", b"\x10\x20"), ("rory", b"\x30\x40")]
    audio = []
    for index in range(turns):
        speaker_id, marker = speakers[index % 2]
        audio.append(
            TurnAudio(
                turn=Turn(
                    act_id=act_id,
                    index=index,
                    speaker_id=speaker_id,
                    speaker_name=speaker_id.title(),
                    text=f"{act_id} line {index}",
                    seconds=seconds,
                ),
                pcm=_pcm(seconds, marker),
            )
        )
    return ActResult(act_id=act_id, audio=audio, usage=RealtimeUsage(output_audio_tokens=100), stop_reason="complete")


@pytest.fixture
def stub_run_act(monkeypatch):
    """Replace recording with instant fake acts; returns the call log."""
    recorded: list[str] = []

    async def fake_run_act(brief, hosts, settings, **kwargs):
        recorded.append(brief.id)
        result = _fake_act(brief.id)
        budget = kwargs.get("budget")
        if budget is not None:
            for _ in result.audio:
                budget.charge(RealtimeUsage(output_audio_tokens=50), settings.model)
        on_turn = kwargs.get("on_turn")
        if on_turn is not None:
            for turn_audio in result.audio:
                on_turn(turn_audio.turn)
        return result

    monkeypatch.setattr(sim, "run_act", fake_run_act)
    return recorded


# ---------- pcm helpers ----------


@pytest.mark.unit
class TestPcmHelpers:
    def test_seconds_from_bytes(self):
        assert pcm_seconds(_pcm(2.5)) == pytest.approx(2.5)

    def test_silence_is_the_right_length_and_actually_silent(self):
        silence = pcm_silence(500)
        assert pcm_seconds(silence) == pytest.approx(0.5)
        assert set(silence) == {0}

    def test_zero_silence_is_empty(self):
        assert pcm_silence(0) == b""
        assert pcm_silence(-10) == b""

    def test_join_pcm_inserts_gaps_between_turns_only(self):
        result = _fake_act("act1", seconds=1.0, turns=3)
        assert pcm_seconds(result.join_pcm()) == pytest.approx(3.0)
        assert pcm_seconds(result.join_pcm(gap_ms=1000)) == pytest.approx(5.0)


@pytest.mark.unit
class TestMixdown:
    def test_pcm_becomes_a_segment_without_transcoding(self):
        segment = mixdown.pcm_to_segment(_pcm(1.0))
        assert segment.frame_rate == REALTIME_SAMPLE_RATE
        assert segment.channels == 1
        assert len(segment) == pytest.approx(1000, abs=5)

    def test_encode_decode_round_trip_preserves_duration(self):
        encoded = mixdown.encode_pcm(_pcm(1.0), "wav")
        decoded = mixdown.decode_to_pcm(encoded, "wav")
        assert pcm_seconds(decoded) == pytest.approx(1.0, abs=0.02)

    def test_stem_matches_the_master_length_and_isolates_one_speaker(self):
        timeline = [("avery", _pcm(1.0, b"\x10\x20")), ("rory", _pcm(1.0, b"\x30\x40")), (None, pcm_silence(500))]
        stem = mixdown.render_stem(timeline, "avery", "wav", "192k")
        decoded = mixdown.decode_to_pcm(stem, "wav")
        assert pcm_seconds(decoded) == pytest.approx(2.5, abs=0.05)
        # The second half (Rory + the gap) must be silent in Avery's stem.
        tail = decoded[len(decoded) // 2 :]
        assert max(tail) == 0

    def test_slice_by_durations_partitions_the_whole_buffer(self):
        pcm = _pcm(3.0)
        parts = mixdown.slice_pcm_by_durations(pcm, [1.0, 2.0])
        assert b"".join(parts) == pcm
        assert pcm_seconds(parts[0]) == pytest.approx(1.0, abs=0.01)

    def test_slice_tolerates_a_decoded_length_that_drifted(self):
        # mp3 round-trips do not preserve exact byte counts; boundaries scale.
        parts = mixdown.slice_pcm_by_durations(_pcm(3.05), [1.0, 1.0, 1.0])
        assert len(parts) == 3
        assert all(len(p) > 0 for p in parts)

    def test_slice_with_no_durations_returns_the_buffer(self):
        pcm = _pcm(1.0)
        assert mixdown.slice_pcm_by_durations(pcm, []) == [pcm]


@pytest.mark.unit
class TestStitchWithPcmDecoder:
    def test_realtime_audio_is_resampled_to_the_podcast_rate(self):
        from io import BytesIO

        from pydub import AudioSegment

        from sanzaru.tools.podcast import _stitch_audio

        data = _stitch_audio(
            segment_bytes_list=[_pcm(1.0), _pcm(1.0)],
            pause_ms_list=[500, 0],
            intro_ms=0,
            outro_ms=0,
            normalize_loudness=False,
            output_format="wav",
            output_bitrate="192k",
            decode=mixdown.pcm_to_segment,
        )
        stitched = AudioSegment.from_file(BytesIO(data), format="wav")
        assert stitched.frame_rate == 44100
        assert len(stitched) == pytest.approx(2500, abs=20)


# ---------- pricing ----------


@pytest.mark.unit
class TestPricing:
    def test_audio_and_text_output_bill_differently(self):
        audio_only = usage_cost(RealtimeUsage(output_audio_tokens=1_000_000), "gpt-realtime-2.1-mini")
        text_only = usage_cost(RealtimeUsage(output_text_tokens=1_000_000), "gpt-realtime-2.1-mini")
        assert audio_only == pytest.approx(20.0)
        assert text_only == pytest.approx(2.40)

    def test_cached_input_is_nearly_free(self):
        usage = RealtimeUsage(input_audio_tokens=1_000_000, cached_audio_tokens=1_000_000)
        assert usage_cost(usage, "gpt-realtime-2.1") == pytest.approx(0.40)

    def test_dated_snapshots_bill_like_their_base_model(self):
        assert prices_for("gpt-realtime-2.1-2026-01-01") == prices_for("gpt-realtime-2.1")

    def test_unknown_model_has_no_price_rather_than_a_wrong_one(self):
        assert prices_for("some-future-model") is None
        assert usage_cost(RealtimeUsage(output_audio_tokens=10), "some-future-model") is None

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1", "1,2,3,4,5,6")
        assert prices_for("gpt-realtime-2.1") == ModelPrices(1, 2, 3, 4, 5, 6)

    def test_malformed_env_override_is_ignored(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1", "not,a,price")
        assert prices_for("gpt-realtime-2.1") == ModelPrices(4.0, 0.40, 32.0, 0.40, 64.0, 24.0)

    def test_projection_scales_input_with_listeners_not_hosts(self):
        solo = project_usage(seconds=100, turns=10, hosts=1)
        duo = project_usage(seconds=100, turns=10, hosts=2)
        assert solo.input_audio_tokens == 0
        assert duo.input_audio_tokens > 0


# ---------- planning ----------


@pytest.mark.unit
class TestAnnotateUpcoming:
    def test_each_act_learns_what_later_acts_own(self, rundown):
        annotated = sim.annotate_upcoming(rundown)
        assert "Act 2" in annotated.acts[0].upcoming
        assert "Act 3" in annotated.acts[0].upcoming

    def test_the_last_act_has_nothing_upcoming(self, rundown):
        assert sim.annotate_upcoming(rundown).acts[-1].upcoming == ""

    def test_lookahead_is_bounded(self):
        many = Rundown(
            title="t",
            hosts=[HostSpec(id="a", name="A")],
            acts=[ActBrief(id=f"act{i}", title=f"T{i}", topic="x") for i in range(8)],
        )
        assert sim.annotate_upcoming(many).acts[0].upcoming.count(";") == sim.UPCOMING_LOOKAHEAD - 1

    def test_an_explicit_upcoming_is_left_alone(self, rundown):
        rundown.acts[0].upcoming = "mine"
        assert sim.annotate_upcoming(rundown).acts[0].upcoming == "mine"


@pytest.mark.unit
class TestMaxSessions:
    def test_explicit_wins(self):
        assert sim._max_sessions(sim.SimulationBrief(premise="p", max_concurrent_sessions=11)) == 11

    def test_env_is_honoured(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_MAX_SESSIONS", "4")
        assert sim._max_sessions(sim.SimulationBrief(premise="p")) == 4

    @pytest.mark.parametrize("value", ["", "  ", "nope", "0", "-3"])
    def test_junk_falls_back_to_the_default(self, monkeypatch, value):
        monkeypatch.setenv("SANZARU_REALTIME_MAX_SESSIONS", value)
        assert sim._max_sessions(sim.SimulationBrief(premise="p")) == sim.DEFAULT_MAX_SESSIONS


@pytest.mark.unit
class TestResolveRundown:
    """The unrecordable cases are rejected by the schema, before any API call."""

    def test_a_brief_with_nothing_to_record_is_rejected_at_parse_time(self):
        with pytest.raises(ValidationError, match="nothing to record"):
            sim.SimulationBrief()

    def test_resume_without_a_run_id_is_rejected(self):
        with pytest.raises(ValidationError, match="needs a run_id"):
            sim.SimulationBrief(resume=True)

    def test_a_rundown_with_no_acts_is_rejected_at_parse_time(self):
        with pytest.raises(ValidationError):
            Rundown(title="t", hosts=[HostSpec(id="a", name="A")], acts=[])

    async def test_passes_a_usable_rundown_straight_through(self, rundown):
        assert await sim.resolve_rundown(sim.SimulationBrief(rundown=rundown)) is rundown


@pytest.mark.unit
class TestDryRun:
    async def test_records_nothing_and_writes_nothing(self, rundown, media_dir):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, dry_run=True))
        assert result.dry_run is True
        assert result.output_file == ""
        assert list(media_dir.iterdir()) == []

    async def test_projects_cost_and_duration(self, rundown):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, dry_run=True))
        assert result.duration_seconds == pytest.approx(12.0)
        assert result.cost.estimated is True
        assert result.cost.usd and result.cost.usd > 0

    async def test_offers_a_resume_command(self, rundown):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, dry_run=True))
        assert result.run_id in result.resume_command


# ---------- recording, checkpointing, resume ----------


@pytest.mark.integration
class TestSimulate:
    async def test_records_every_act_and_writes_the_episode(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", output_format="wav")
        )
        assert stub_run_act == ["act1", "act2", "act3"]
        assert (media_dir / result.output_file).exists()
        assert result.turn_count == 6
        assert [a.act_id for a in result.acts] == ["act1", "act2", "act3"]

    async def test_act_order_survives_parallel_recording(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        # Results are read by index, not completion order.
        assert [a.act_id for a in result.acts] == ["act1", "act2", "act3"]
        assert result.transcript.index("act1 line 0") < result.transcript.index("act3 line 0")

    async def test_checkpoints_each_act_with_a_manifest(self, rundown, media_dir, stub_run_act):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        names = {p.name for p in media_dir.iterdir()}
        assert "simrun_testrun.json" in names
        for act_id in ("act1", "act2", "act3"):
            assert f"Stitch_Test_testrun_{act_id}.mp3" in names
            assert f"Stitch_Test_testrun_{act_id}.json" in names

    async def test_the_sidecar_carries_turns_and_usage(self, rundown, media_dir, stub_run_act):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        meta = json.loads((media_dir / "Stitch_Test_testrun_act1.json").read_text())
        assert [t["speaker_id"] for t in meta["turns"]] == ["avery", "rory"]
        assert meta["usage"]["output_audio_tokens"] == 100
        assert meta["stop_reason"] == "complete"

    async def test_the_run_manifest_makes_resume_self_sufficient(self, rundown, media_dir, stub_run_act):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        manifest = json.loads((media_dir / "simrun_testrun.json").read_text())
        assert manifest["rundown"]["title"] == "Stitch Test"
        assert len(manifest["rundown"]["acts"]) == 3
        # The nested copy is dropped so the manifest has one source of truth.
        assert manifest["brief"]["rundown"] is None

    async def test_resume_reuses_checkpoints_and_records_only_gaps(self, rundown, media_dir, stub_run_act):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        stub_run_act.clear()
        (media_dir / "Stitch_Test_testrun_act2.mp3").unlink()
        (media_dir / "Stitch_Test_testrun_act2.json").unlink()

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == ["act2"]
        assert [(a.act_id, a.reused) for a in result.acts] == [("act1", True), ("act2", False), ("act3", True)]
        assert result.turn_count == 6

    async def test_a_resumed_act_keeps_its_transcript_and_usage(self, rundown, media_dir, stub_run_act):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert "act1 line 0" in result.transcript
        assert result.cost.usage.output_audio_tokens == 300

    async def test_a_corrupt_checkpoint_is_re_recorded_rather_than_fatal(self, rundown, media_dir, stub_run_act):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        stub_run_act.clear()
        (media_dir / "Stitch_Test_testrun_act3.json").write_text("{not json")

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == ["act3"]
        assert len(result.acts) == 3

    async def test_resume_without_a_manifest_or_rundown_is_an_error(self, media_dir):
        with pytest.raises(ValueError, match="no run manifest"):
            await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="nope", qc=False))

    async def test_stems_are_written_per_host(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", stems=True, output_format="wav")
        )
        assert set(result.stems) == {"avery", "rory"}
        for name in result.stems.values():
            assert (media_dir / name).exists()

    async def test_stems_are_the_same_length_as_the_episode(self, rundown, media_dir, stub_run_act):
        from io import BytesIO

        from pydub import AudioSegment

        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", stems=True, output_format="wav")
        )
        master = AudioSegment.from_file(BytesIO((media_dir / result.output_file).read_bytes()), format="wav")
        stem = AudioSegment.from_file(
            BytesIO((media_dir / result.stems["avery"]).read_bytes()),
            format="wav",
        )
        assert len(stem) == pytest.approx(len(master), abs=30)

    async def test_output_filename_is_honoured(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", filename="chosen.mp3")
        )
        assert result.output_file == "chosen.mp3"
        assert (media_dir / "chosen.mp3").exists()

    async def test_upcoming_is_filled_in_before_recording(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        assert result.rundown is not None
        assert result.rundown.acts[0].upcoming
        assert result.rundown.acts[-1].upcoming == ""


@pytest.mark.integration
class TestCostCeiling:
    async def test_aborts_and_leaves_finished_acts_on_disk(self, rundown, media_dir, monkeypatch):
        from sanzaru.exceptions import CostCeilingError

        monkeypatch.setenv("SANZARU_REALTIME_MAX_SESSIONS", "2")

        calls: list[str] = []

        async def fake_run_act(brief, hosts, settings, **kwargs):
            calls.append(brief.id)
            result = _fake_act(brief.id)
            budget = kwargs["budget"]
            budget.charge(RealtimeUsage(output_audio_tokens=500_000), settings.model)
            return result

        monkeypatch.setattr(sim, "run_act", fake_run_act)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 — anyio wraps it in a group
            await sim.simulate_podcast(
                sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", max_cost_usd=0.02)
            )

        from sanzaru.cli._runtime import find_in_group

        # Several acts can trip the ceiling in one scheduling window, so the
        # group may carry more than one — the CLI has to find it either way.
        ceiling = find_in_group(excinfo.value, CostCeilingError)
        assert ceiling is not None
        assert ceiling.limit_usd == 0.02
        # The run manifest is written before any recording, so resume works.
        assert (media_dir / "simrun_testrun.json").exists()

    async def test_a_generous_ceiling_does_not_fire(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", max_cost_usd=100.0)
        )
        assert result.cost.limit_usd == 100.0
        assert len(result.acts) == 3


@pytest.mark.unit
class TestNaming:
    def test_run_ids_are_short_and_hex(self):
        run_id = sim.new_run_id()
        assert len(run_id) == 8
        int(run_id, 16)

    def test_checkpoint_names_group_by_run(self):
        assert sim._act_audio_name("show", "abc", "act1") == "show_abc_act1.mp3"
        assert sim._act_meta_name("show", "abc", "act1") == "show_abc_act1.json"
        assert sim._manifest_name("abc") == "simrun_abc.json"

    def test_checkpoint_paths_stay_inside_the_media_dir(self, media_dir):
        # Names are basenames by construction; the storage layer sanitizes anyway.
        for name in (sim._act_audio_name("show", "abc", "act1"), sim._manifest_name("abc")):
            assert pathlib.Path(name).name == name


# ---------- schema validation ----------


@pytest.mark.unit
class TestRundownValidation:
    """Guards that reject a rundown before it can cost anything."""

    def _rundown(self, **overrides):
        payload = {
            "title": "t",
            "hosts": [HostSpec(id="a", name="A"), HostSpec(id="b", name="B")],
            "acts": [ActBrief(id="act1", title="One", topic="x"), ActBrief(id="act2", title="Two", topic="y")],
        }
        payload.update(overrides)
        return Rundown(**payload)

    def test_duplicate_act_ids_are_rejected(self):
        # Two acts with one id overwrite each other's checkpoint, so a resumed
        # run would silently lose one.
        with pytest.raises(ValidationError, match="duplicate act id"):
            self._rundown(acts=[ActBrief(id="act1", title="A", topic="x"), ActBrief(id="act1", title="B", topic="y")])

    def test_duplicate_host_ids_are_rejected(self):
        # Two hosts with one id collapse into one speaker, breaking stems.
        with pytest.raises(ValidationError, match="duplicate host id"):
            self._rundown(hosts=[HostSpec(id="a", name="A"), HostSpec(id="a", name="B")])

    def test_speaking_order_naming_an_unknown_host_is_rejected(self):
        act = ActBrief(id="act1", title="One", topic="x", speaking_order=["a", "ghost"])
        with pytest.raises(ValidationError, match="ghost"):
            self._rundown(acts=[act])

    def test_a_valid_speaking_order_passes(self):
        act = ActBrief(id="act1", title="One", topic="x", speaking_order=["a", "b", "b"])
        assert self._rundown(acts=[act]).acts[0].speaking_order == ["a", "b", "b"]

    def test_an_empty_rundown_is_rejected(self):
        with pytest.raises(ValidationError):
            self._rundown(acts=[])
        with pytest.raises(ValidationError):
            self._rundown(hosts=[])

    def test_ids_that_would_escape_the_media_dir_are_rejected(self):
        # Ids reach filenames; storage sanitizes too, but fail early and clearly.
        for bad in ("../etc", "a/b", "a\\b", ""):
            with pytest.raises(ValidationError):
                ActBrief(id=bad, title="t", topic="x")

    def test_turn_notes_outside_the_act_are_rejected(self):
        with pytest.raises(ValidationError, match="would never fire"):
            ActBrief(id="act1", title="t", topic="x", max_turns=4, turn_notes={9: "never runs"})

    def test_absurd_act_budgets_are_rejected(self):
        with pytest.raises(ValidationError):
            ActBrief(id="act1", title="t", topic="x", target_seconds=99_999)
        with pytest.raises(ValidationError):
            ActBrief(id="act1", title="t", topic="x", max_turns=0)

    def test_an_unknown_voice_warns_but_is_allowed(self, caplog):
        # OpenAI ships voices faster than we do, so this must not hard-fail.
        with caplog.at_level("WARNING", logger="sanzaru"):
            host = HostSpec(id="a", name="A", voice="brand-new-voice")
        assert host.voice == "brand-new-voice"
        assert "not a known realtime voice" in caplog.text

    def test_an_underbudgeted_act_warns(self, caplog):
        with caplog.at_level("WARNING", logger="sanzaru"):
            ActBrief(id="act1", title="t", topic="x", target_seconds=600, max_turns=4)
        assert "stop on the duration budget" in caplog.text


@pytest.mark.unit
class TestSimulationBriefValidation:
    def test_run_id_cannot_escape_the_media_dir(self):
        for bad in ("../../etc/passwd", "a/b", "with space"):
            with pytest.raises(ValidationError):
                sim.SimulationBrief(premise="p", run_id=bad)

    def test_filename_cannot_contain_a_separator(self):
        with pytest.raises(ValidationError):
            sim.SimulationBrief(premise="p", filename="../out.mp3")

    def test_bitrate_must_look_like_a_bitrate(self):
        with pytest.raises(ValidationError):
            sim.SimulationBrief(premise="p", output_bitrate="loud")
        assert sim.SimulationBrief(premise="p", output_bitrate="320k").output_bitrate == "320k"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"acts": 0},
            {"acts": 999},
            {"target_minutes": 0},
            {"turn_seconds": 0},
            {"turn_tokens": -1},
            {"max_cost_usd": 0},
            {"max_cost_usd": -5},
            {"act_gap_ms": -1},
            {"max_concurrent_sessions": -1},
        ],
    )
    def test_out_of_range_values_are_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            sim.SimulationBrief(premise="p", **kwargs)

    def test_a_ceiling_on_an_unpriceable_model_warns_loudly(self, caplog):
        with caplog.at_level("WARNING", logger="sanzaru"):
            sim.SimulationBrief(premise="p", model="some-future-model", max_cost_usd=5.0)
        assert "ceiling will never fire" in caplog.text

    def test_an_explicit_cap_too_small_for_one_act_is_rejected(self, rundown):
        with pytest.raises(ValidationError, match="cannot fit one act"):
            sim.SimulationBrief(rundown=rundown, max_concurrent_sessions=1)

    def test_the_default_cap_defers_to_the_environment(self, rundown, monkeypatch):
        """A validator must not reject what SANZARU_REALTIME_MAX_SESSIONS allows."""
        monkeypatch.setenv("SANZARU_REALTIME_MAX_SESSIONS", "16")
        brief = sim.SimulationBrief(rundown=rundown, max_concurrent_sessions=0)
        assert sim._max_sessions(brief) == 16

    def test_the_run_manifest_brief_is_itself_valid(self, rundown):
        """The stored brief has the rundown stripped, so it must stand alone."""
        brief = sim.SimulationBrief(rundown=rundown, run_id="testrun")
        stored = brief.model_copy(update={"rundown": None, "resume": True})
        manifest = sim.RunManifest(run_id="testrun", slug="s", created=0.0, rundown=rundown, brief=stored)
        assert sim.RunManifest.model_validate_json(manifest.model_dump_json()).brief.resume is True
