"""Tests for the simulate_podcast tool: mixdown, checkpointing, resume, budgets.

Recording itself is stubbed at the `run_act` seam — floor control is covered in
test_realtime_producer.py, and what matters here is everything around it: that
acts land on disk before anything can go wrong, that a resumed run reuses them,
that stems line up with the master, and that a cost ceiling leaves the finished
work intact.
"""

import json
import pathlib

import anyio
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
    extension_cap,
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


@pytest.fixture
def output_override():
    """Repoint the whole "audio" path type, exactly the way the CLI's `-o` does.

    `plan_output`/`install_overrides` turn `-o ./out/ep.mp3` into a
    `LocalStorageBackend(path_overrides={"audio": ./out})`, so this is the seam
    where checkpoints could get dragged out of the media dir.
    """
    from sanzaru.storage import set_storage_backend
    from sanzaru.storage.local import LocalStorageBackend

    def point_at(directory=None):
        set_storage_backend(None if directory is None else LocalStorageBackend(path_overrides={"audio": directory}))

    yield point_at
    set_storage_backend(None)


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

    def test_a_dated_mini_snapshot_bills_at_the_mini_tier(self):
        """The longest matching base wins, not the first one in the table.

        "gpt-realtime-2.1" is a prefix of "gpt-realtime-2.1-mini", so scanning in
        insertion order priced every dated -mini snapshot 3.2x over — silently,
        in both the dry-run projection and the cost ceiling.
        """
        prices = prices_for("gpt-realtime-2.1-mini-2026-01-01")
        assert prices == prices_for("gpt-realtime-2.1-mini")
        assert prices is not None
        assert prices.audio_output == pytest.approx(20.0)

    @pytest.mark.parametrize(
        ("model", "base"),
        [
            ("gpt-realtime-2.1-2026-01-01", "gpt-realtime-2.1"),
            ("gpt-realtime-2.1-mini-2026-01-01", "gpt-realtime-2.1-mini"),
            ("gpt-realtime-mini-2025-08-28", "gpt-realtime-mini"),
            ("gpt-realtime-2-2025-08-28", "gpt-realtime-2"),
        ],
    )
    def test_every_dated_snapshot_finds_its_own_tier(self, model, base):
        assert prices_for(model) is prices_for(base)

    def test_unknown_model_has_no_price_rather_than_a_wrong_one(self):
        assert prices_for("some-future-model") is None
        assert usage_cost(RealtimeUsage(output_audio_tokens=10), "some-future-model") is None

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1", "1,2,3,4,5,6")
        assert prices_for("gpt-realtime-2.1") == ModelPrices(1, 2, 3, 4, 5, 6)

    def test_malformed_env_override_is_ignored(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1", "not,a,price")
        assert prices_for("gpt-realtime-2.1") == ModelPrices(4.0, 0.40, 32.0, 0.40, 64.0, 24.0)

    @pytest.mark.parametrize("raw", ["not,a,price", "1,2,3,4,5", "1,2,3,4,5,6,7", "4,0.4,32,0.4,64,cheap"])
    def test_a_malformed_env_override_says_so(self, monkeypatch, caplog, raw):
        # Someone who set this variable wanted it to take effect; billing them at
        # list price without a word is the outcome they were trying to avoid.
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1", raw)
        with caplog.at_level("WARNING", logger="sanzaru"):
            assert prices_for("gpt-realtime-2.1") == ModelPrices(4.0, 0.40, 32.0, 0.40, 64.0, 24.0)
        assert "SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1" in caplog.text

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

    async def test_a_supplied_rundown_still_gets_its_voices_assigned(self):
        """The hand-edited-rundown path is the one the docs push hardest.

        It never goes through `_merge_hosts`, so before this it was the one path
        where every voiceless host recorded in the same voice — and the rundown
        came back from here completely untouched, so nothing later could fix it.
        """
        supplied = Rundown.model_validate(
            {
                "title": "Stitch",
                "hosts": [{"id": "avery", "name": "Avery"}, {"id": "rory", "name": "Rory"}],
                "acts": [{"id": "act1", "title": "One", "topic": "x"}],
            }
        )
        resolved = await sim.resolve_rundown(sim.SimulationBrief(rundown=supplied))
        assert [h.voice for h in resolved.hosts] == ["marin", "cedar"]

    async def test_a_voice_the_author_chose_is_left_alone(self):
        supplied = Rundown(
            title="Stitch",
            hosts=[HostSpec(id="avery", name="Avery", voice="cedar"), HostSpec(id="rory", name="Rory")],
            acts=[ActBrief(id="act1", title="One", topic="x")],
        )
        resolved = await sim.resolve_rundown(sim.SimulationBrief(rundown=supplied))
        assert [h.voice for h in resolved.hosts] == ["cedar", "marin"]


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

    async def test_projects_turns_against_the_extended_ceiling(self, rundown):
        """#50: an act runs to `extension_cap`, so the projection must say so.

        Quoting `max_turns` handed callers a number the tool's own description
        tells them to budget past — and the same projection drives the resume
        refusal, where reading low is the dangerous direction.
        """
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, dry_run=True))

        assert result.turn_count == sum(extension_cap(act.max_turns) for act in rundown.acts)
        assert result.turn_count > rundown.total_max_turns(), "the fixture must actually extend"
        assert [a.turns for a in result.acts] == [extension_cap(act.max_turns) for act in rundown.acts]

    async def test_the_extended_projection_costs_more_than_the_planned_one(self, rundown):
        """Small but real: `text_in` is the one turn-scaled term."""
        planned = sim.project_run(
            rundown.model_copy(update={"acts": [a.model_copy(update={"max_turns": 1}) for a in rundown.acts]}),
            sim.SimulationBrief(rundown=rundown, dry_run=True),
        )
        extended = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, dry_run=True))

        assert extended.cost.usage.input_text_tokens > planned.usage.input_text_tokens


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

    async def test_undecodable_audio_is_re_recorded_rather_than_fatal(self, rundown, media_dir, stub_run_act):
        # LocalStorageBackend.write is a plain open+write, so a crash mid-write
        # leaves a truncated mp3 that `exists()` happily reports.
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        stub_run_act.clear()
        (media_dir / "Stitch_Test_testrun_act2.mp3").write_bytes(b"\xff\xfb" + b"truncated" * 8)

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == ["act2"]
        assert result.turn_count == 6

    async def test_a_zero_audio_checkpoint_is_re_recorded_rather_than_fatal(self, rundown, media_dir, stub_run_act):
        # What a turn that produced no audio used to leave behind: a ~500-byte
        # mp3 with no decodable frame. One of those must not kill every resume.
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        stub_run_act.clear()
        (media_dir / "Stitch_Test_testrun_act1.mp3").write_bytes(mixdown.encode_pcm(b"", "mp3"))

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == ["act1"]
        assert result.turn_count == 6

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

    async def test_stems_stay_aligned_when_the_episode_has_head_and_tail_silence(
        self, rundown, media_dir, stub_run_act
    ):
        """`render_stem` promises sample-for-sample alignment with the master.

        The stitch step prepends `intro_silence_ms` to the master, so a timeline
        built from turns and act gaps alone shifts every stem earlier by exactly
        that much — drift you only find once the tracks are on an editor
        timeline. The older length test only ever ran with intro=0.
        """
        from io import BytesIO

        from pydub import AudioSegment

        result = await sim.simulate_podcast(
            sim.SimulationBrief(
                rundown=rundown,
                qc=False,
                run_id="testrun",
                stems=True,
                output_format="wav",
                intro_silence_ms=1500,
                outro_silence_ms=800,
            )
        )

        def milliseconds(name):
            return len(AudioSegment.from_file(BytesIO((media_dir / name).read_bytes()), format="wav"))

        master = milliseconds(result.output_file)
        for name in result.stems.values():
            assert milliseconds(name) == pytest.approx(master, abs=30)

    async def test_the_intro_silence_is_silent_in_a_stem_too(self, rundown, media_dir, stub_run_act):
        # Alignment by length alone would also pass if the intro were filled with
        # the first turn's audio, which is the opposite of a stem.
        from io import BytesIO

        from pydub import AudioSegment

        result = await sim.simulate_podcast(
            sim.SimulationBrief(
                rundown=rundown, qc=False, run_id="testrun", stems=True, output_format="wav", intro_silence_ms=1500
            )
        )
        stem = AudioSegment.from_file(BytesIO((media_dir / result.stems["avery"]).read_bytes()), format="wav")
        assert max(stem[:1400].raw_data) == 0

    async def test_output_filename_is_honoured(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", filename="chosen.mp3")
        )
        assert result.output_file == "chosen.mp3"
        assert (media_dir / "chosen.mp3").exists()

    async def test_the_briefs_producer_knobs_reach_every_act(self, rundown, media_dir, monkeypatch):
        """`turn_timeout_s` and friends are documented brief-level overrides.

        Nothing else in the suite crosses the brief → SimulationSettings seam:
        the producer tests build `SimulationSettings` directly, so dropping the
        forwarding line here left every one of them green.
        """
        seen = []

        async def capture_settings(brief, hosts, settings, **kwargs):
            seen.append(settings)
            return _fake_act(brief.id)

        monkeypatch.setattr(sim, "run_act", capture_settings)
        await sim.simulate_podcast(
            sim.SimulationBrief(
                rundown=rundown,
                qc=False,
                run_id="testrun",
                turn_timeout_s=12.0,
                turn_seconds=9.0,
                turn_tokens=77,
                model="gpt-realtime-2.1-mini",
            )
        )
        assert len(seen) == 3
        assert {(s.turn_timeout_s, s.turn_seconds, s.max_turn_tokens, s.model) for s in seen} == {
            (12.0, 9.0, 77, "gpt-realtime-2.1-mini")
        }

    async def test_upcoming_is_filled_in_before_recording(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        assert result.rundown is not None
        assert result.rundown.acts[0].upcoming
        assert result.rundown.acts[-1].upcoming == ""


@pytest.mark.integration
class TestOutputOverrideAndResume:
    """`-o` and `--resume RUN_ID` are documented as one workflow; they compose.

    `-o` repoints the whole "audio" path type for one invocation, but the resume
    hint it prints carries no `-o` — so anything a later run has to find by run
    id alone belongs in the media dir regardless.
    """

    async def test_the_episode_follows_the_override(self, rundown, media_dir, tmp_path, stub_run_act, output_override):
        out = tmp_path / "out"
        out.mkdir()
        output_override(out)
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        assert (out / result.output_file).exists()

    async def test_the_manifest_and_checkpoints_stay_in_the_media_dir(
        self, rundown, media_dir, tmp_path, stub_run_act, output_override
    ):
        out = tmp_path / "out"
        out.mkdir()
        output_override(out)
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        assert (media_dir / "simrun_testrun.json").exists()
        assert (media_dir / "Stitch_Test_testrun_act1.mp3").exists()
        # And the user's output directory holds the episode, not 2N+1 intermediates.
        assert {p.name for p in out.iterdir()} == {result.output_file}

    async def test_stems_still_follow_the_override(self, rundown, media_dir, tmp_path, stub_run_act, output_override):
        out = tmp_path / "out"
        out.mkdir()
        output_override(out)
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", stems=True, output_format="wav")
        )
        for name in result.stems.values():
            assert (out / name).exists()

    async def test_a_run_recorded_with_an_override_resumes_without_one(
        self, rundown, media_dir, tmp_path, stub_run_act, output_override
    ):
        out = tmp_path / "out"
        out.mkdir()
        output_override(out)
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))

        output_override(None)  # the recovery command carries no -o
        stub_run_act.clear()
        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == []
        assert all(act.reused for act in result.acts)
        assert (media_dir / result.output_file).exists()

    async def test_no_media_dir_at_all_still_records(self, rundown, tmp_path, monkeypatch, stub_run_act):
        """The CLI falls back to cwd when nothing is configured; so do we."""
        from sanzaru.config import get_path

        monkeypatch.delenv("AUDIO_PATH", raising=False)
        monkeypatch.delenv("SANZARU_MEDIA_PATH", raising=False)
        get_path.cache_clear()
        out = tmp_path / "cwd"
        out.mkdir()
        try:
            from sanzaru.storage import set_storage_backend
            from sanzaru.storage.local import LocalStorageBackend

            set_storage_backend(LocalStorageBackend(path_overrides={"audio": out}))
            result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        finally:
            set_storage_backend(None)
            get_path.cache_clear()
        # Nowhere more durable exists, so the checkpoints land beside the episode.
        assert (out / result.output_file).exists()
        assert (out / "simrun_testrun.json").exists()


@pytest.mark.integration
class TestResumeRestoresSettings:
    """What `--resume RUN_ID` alone must reinstate, and what it must not."""

    async def _record(self, rundown, **kwargs):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", **kwargs))

    async def test_the_cost_ceiling_survives_a_bare_resume(self, rundown, media_dir, stub_run_act):
        # The ceiling abort's own resume hint carries no flags, so a resume that
        # dropped max_cost_usd would re-run uncapped — disabling the exact
        # safety that produced the hint.
        await self._record(rundown, max_cost_usd=100.0)
        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert result.cost.limit_usd == 100.0

    async def test_a_caller_supplied_ceiling_still_wins(self, rundown, media_dir, stub_run_act):
        await self._record(rundown, max_cost_usd=100.0)
        result = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=5.0)
        )
        assert result.cost.limit_usd == 5.0

    async def test_the_output_format_can_be_changed_on_resume(self, rundown, media_dir, stub_run_act):
        await self._record(rundown, output_format="mp3")
        result = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, output_format="wav")
        )
        assert result.output_file.endswith(".wav")
        assert (media_dir / result.output_file).exists()

    async def test_the_act_gap_can_be_changed_on_resume(self, rundown, media_dir, stub_run_act):
        from io import BytesIO

        from pydub import AudioSegment

        await self._record(rundown, act_gap_ms=0, output_format="wav")

        def milliseconds(name):
            return len(AudioSegment.from_file(BytesIO((media_dir / name).read_bytes()), format="wav"))

        tight = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, output_format="wav", filename="tight.wav")
        )
        wide = await sim.simulate_podcast(
            sim.SimulationBrief(
                resume=True,
                run_id="testrun",
                qc=False,
                output_format="wav",
                filename="wide.wav",
                act_gap_ms=2000,
            )
        )
        assert wide.duration_seconds == tight.duration_seconds  # the act audio itself is unchanged
        # Two 2s gaps between three acts, which the manifest's act_gap_ms=0 omits.
        assert milliseconds(wide.output_file) - milliseconds(tight.output_file) == pytest.approx(4000, abs=60)

    async def test_a_new_container_renames_the_restored_filename(self, rundown, media_dir, stub_run_act):
        """`-o ep1.mp3` puts a name in the manifest that `--format wav` outdates.

        The filename is restored independently of the format, so the resumed run
        wrote RIFF bytes into a file still called `.mp3` — newly reachable the
        moment `--format` started taking effect on resume at all.
        """
        await self._record(rundown, filename="ep1.mp3")
        result = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, output_format="wav")
        )
        assert result.output_file == "ep1.wav"
        assert (media_dir / "ep1.wav").read_bytes()[:4] == b"RIFF"

    async def test_a_filename_passed_on_the_resume_is_left_alone(self, rundown, media_dir, stub_run_act):
        # Only a *restored* name is corrected; one the caller typed this time is
        # theirs, extension and all.
        await self._record(rundown, filename="ep1.mp3")
        result = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, output_format="wav", filename="mine.mp3")
        )
        assert result.output_file == "mine.mp3"

    async def test_a_manifest_written_before_voices_were_assigned_still_gets_them(
        self, rundown, media_dir, stub_run_act
    ):
        """Manifests outlive the version that wrote them.

        The resume path never calls `resolve_rundown`, so a manifest whose hosts
        have no voice — every one written before voice assignment existed —
        would replay with the whole cast in the default voice.
        """
        await self._record(rundown)
        manifest_path = media_dir / "simrun_testrun.json"
        manifest = json.loads(manifest_path.read_text())
        for host in manifest["rundown"]["hosts"]:
            host["voice"] = ""
        manifest_path.write_text(json.dumps(manifest))

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert result.rundown is not None
        assert [h.voice for h in result.rundown.hosts] == ["marin", "cedar"]

    async def test_settings_the_caller_did_not_pass_come_from_the_manifest(self, rundown, media_dir, stub_run_act):
        await self._record(rundown, output_format="wav", turn_seconds=42.0)
        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert result.output_file.endswith(".wav")
        manifest = json.loads((media_dir / "simrun_testrun.json").read_text())
        assert manifest["brief"]["turn_seconds"] == 42.0


@pytest.mark.integration
class TestReportedCost:
    """`cost.usd` is the number a user reconciles against their bill.

    It used to be recomputed from pooled usage at the episode model, which is a
    different number from the one the budget charged the moment either a host
    overrides the model or `--qc-retry` throws a paid take away.
    """

    @pytest.fixture
    def mixed_model_rundown(self):
        return Rundown(
            title="Stitch Test",
            premise="the plumbing",
            hosts=[
                HostSpec(id="avery", name="Avery", voice="marin"),
                # ~3x cheaper: pooling the two and pricing at the episode model
                # cannot produce the right answer for this episode.
                HostSpec(id="rory", name="Rory", voice="cedar", model="gpt-realtime-2.1-mini"),
            ],
            acts=[
                ActBrief(
                    id=f"act{i + 1}", title=f"Act {i + 1}", topic=f"topic {i + 1}", target_seconds=4.0, max_turns=2
                )
                for i in range(3)
            ],
        )

    async def test_a_mixed_model_episode_reports_what_was_actually_charged(
        self, mixed_model_rundown, media_dir, monkeypatch
    ):
        async def fake_run_act(brief, hosts, settings, **kwargs):
            result = _fake_act(brief.id)
            by_id = {h.id: h for h in hosts}
            # Mirrors run_act: each turn is charged at the model that spoke it.
            for turn_audio in result.audio:
                host = by_id[turn_audio.turn.speaker_id]
                kwargs["budget"].charge(RealtimeUsage(output_audio_tokens=500_000), host.model or settings.model)
            return result

        monkeypatch.setattr(sim, "run_act", fake_run_act)
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=mixed_model_rundown, qc=False, run_id="testrun")
        )

        # Per act: 0.5M audio-out at $64/1M (avery) + 0.5M at $20/1M (rory).
        assert result.cost.usd == pytest.approx(3 * (32.0 + 10.0))

    async def test_a_take_discarded_by_qc_retry_is_still_billed(self, rundown, media_dir, monkeypatch):
        """--qc-retry drops the bad take from `recorded`, but not from the bill."""
        from sanzaru.audio.realtime.qc import ActVerdict, QCReport

        takes: list[str] = []

        async def fake_run_act(brief, hosts, settings, **kwargs):
            takes.append(brief.id)
            kwargs["budget"].charge(RealtimeUsage(output_audio_tokens=1_000_000), settings.model)
            return _fake_act(brief.id)

        reports = iter(
            [
                QCReport(verdict="warn", acts=[ActVerdict(act_id="act2", similarity=0.1)], flagged_acts=["act2"]),
                QCReport(verdict="pass"),
            ]
        )

        async def fake_run_qc(*args, **kwargs):
            return next(reports)

        monkeypatch.setattr(sim, "run_act", fake_run_act)
        monkeypatch.setattr(sim, "run_qc", fake_run_qc)

        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, run_id="testrun", qc=True, qc_retry=True)
        )

        assert takes == ["act1", "act2", "act3", "act2"]
        assert len(result.acts) == 3
        # Four takes at 1M audio-out tokens each, $64/1M — the discarded one included.
        assert result.cost.usd == pytest.approx(4 * 64.0)

    async def test_qc_retry_preserves_the_take_it_replaces(self, rundown, media_dir, monkeypatch):
        """Re-recording a flagged act must not destroy the only copy of the prior take.

        QC verdicts have been observed to disagree run-to-run on the same
        material, so the replaced take has to stay recoverable for the caller
        to choose from.
        """
        from sanzaru.audio.realtime.qc import ActVerdict, QCReport

        async def fake_run_act(brief, hosts, settings, **kwargs):
            kwargs["budget"].charge(RealtimeUsage(output_audio_tokens=1_000), settings.model)
            return _fake_act(brief.id)

        reports = iter(
            [
                QCReport(verdict="warn", acts=[ActVerdict(act_id="act2", similarity=0.1)], flagged_acts=["act2"]),
                QCReport(verdict="pass"),
            ]
        )

        async def fake_run_qc(*args, **kwargs):
            return next(reports)

        monkeypatch.setattr(sim, "run_act", fake_run_act)
        monkeypatch.setattr(sim, "run_qc", fake_run_qc)

        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, run_id="testrun", qc=True, qc_retry=True))

        names = {f.name for f in media_dir.iterdir()}
        slug = sim._safe_title("Stitch Test")
        # The retried act's first take survives under a _take1 suffix...
        assert f"{slug}_testrun_act2_take1.mp3" in names
        assert f"{slug}_testrun_act2_take1.json" in names
        # ...the live checkpoint names still hold exactly one truth for resume...
        assert f"{slug}_testrun_act2.mp3" in names
        # ...and unflagged acts were not touched.
        assert f"{slug}_testrun_act1_take1.mp3" not in names

    async def test_a_second_retry_does_not_clobber_the_first_preserved_take(self, media_dir):
        """`_take1` is a floor, not a fixed name — the storage layer has no delete."""
        storage = sim.checkpoint_storage()
        await storage.write("audio", "show_run_act2.mp3", b"take-1")
        assert await sim._next_take_number(storage, "show_run_act2.mp3") == 1

        await storage.write("audio", "show_run_act2_take1.mp3", b"take-1")
        assert await sim._next_take_number(storage, "show_run_act2.mp3") == 2

    async def test_the_take_number_is_shared_by_the_audio_and_its_sidecar(self, media_dir):
        """A `_take2.mp3` beside a `_take1.json` describes two different takes.

        Deriving the number per name lets a half-written pair drift apart, and
        with no delete op the mis-pairing is permanent.
        """
        storage = sim.checkpoint_storage()
        # The audio already has a take1; the sidecar's is missing (an interrupt
        # between the two writes, or a checkpoint that never had one).
        await storage.write("audio", "show_run_act2.mp3", b"live")
        await storage.write("audio", "show_run_act2.json", b"{}")
        await storage.write("audio", "show_run_act2_take1.mp3", b"take-1")

        take = await sim._next_take_number(storage, "show_run_act2.mp3", "show_run_act2.json")

        assert sim._take_name("show_run_act2.mp3", take) == "show_run_act2_take2.mp3"
        assert sim._take_name("show_run_act2.json", take) == "show_run_act2_take2.json"

    async def test_the_take_scan_covers_the_sidecar_too(self, media_dir):
        """The mirror case: a `_take1.json` whose `.mp3` is gone must not be overwritten."""
        storage = sim.checkpoint_storage()
        await storage.write("audio", "show_run_act2.mp3", b"live")
        await storage.write("audio", "show_run_act2.json", b"{}")
        await storage.write("audio", "show_run_act2_take1.json", b"{}")

        assert await sim._next_take_number(storage, "show_run_act2.mp3", "show_run_act2.json") == 2

    async def test_recording_over_an_existing_run_id_is_refused(self, rundown, media_dir, stub_run_act):
        """Naming a run is the recommended workflow, so reusing one must not overwrite it.

        Re-recording under an existing id would replace the manifest and every
        act checkpoint beneath it, stranding the first run's audio against a
        manifest that no longer describes it — and storage has no delete.
        """
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))

        with pytest.raises(ValueError, match="already exists"):
            await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))

        # ...and resuming it, which is what the error tells you to do, works.
        resumed = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert resumed.run_id == "testrun"

    async def test_planning_against_an_existing_run_id_is_allowed(self, rundown, media_dir, stub_run_act):
        """A dry run writes no manifest, so re-planning a recorded run costs nothing.

        Blocking it would break the obvious way to decide whether to resume.
        """
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))

        planned = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", dry_run=True)
        )
        assert planned.dry_run

    async def test_a_resumed_run_reports_the_spend_it_replayed(self, rundown, media_dir, stub_run_act):
        first = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        resumed = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert first.cost.usd is not None
        assert first.cost.usd > 0
        # Every act came off disk, so the run cost nothing new — but reporting
        # zero for a recovered episode would be just as wrong as double-counting.
        assert resumed.cost.usd == pytest.approx(first.cost.usd)

    async def test_an_unpriceable_model_reports_no_figure_rather_than_zero(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", model="some-future-model")
        )
        assert result.cost.usd is None
        assert result.cost.unpriced_models == ["some-future-model"]

    async def test_the_token_breakdown_still_describes_what_shipped(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        assert result.cost.usage.output_audio_tokens == 300


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

    async def test_reused_acts_count_as_checkpointed_and_safe(self, rundown, media_dir, stub_run_act, monkeypatch):
        """The ceiling message is "N act(s) checkpointed and safe" — N must be true.

        A resumed act is on disk by definition; reporting 0 tells the user their
        recovered work is gone and invites them to start over.
        """
        from sanzaru.exceptions import CostCeilingError

        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))
        for suffix in ("mp3", "json"):
            (media_dir / f"Stitch_Test_testrun_act2.{suffix}").unlink()

        async def blow_the_budget(brief, hosts, settings, **kwargs):
            kwargs["budget"].charge(RealtimeUsage(output_audio_tokens=500_000), settings.model)
            return _fake_act(brief.id)

        monkeypatch.setattr(sim, "run_act", blow_the_budget)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 — anyio wraps it in a group
            await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=0.02))

        from sanzaru.cli._runtime import find_in_group

        ceiling = find_in_group(excinfo.value, CostCeilingError)
        assert ceiling is not None
        assert sorted(ceiling.completed_acts) == ["act1", "act3"]

    async def test_a_generous_ceiling_does_not_fire(self, rundown, media_dir, stub_run_act):
        result = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", max_cost_usd=100.0)
        )
        assert result.cost.limit_usd == 100.0
        assert len(result.acts) == 3

    async def test_every_checkpointed_act_is_named_when_the_replay_trips_the_ceiling(
        self, rundown, media_dir, stub_run_act
    ):
        """ "N act(s) checkpointed and safe" is what the user decides to resume on.

        Replaying the checkpointed spend can trip the ceiling partway through
        the list, and marking each act as it was charged left every act after
        the one that raised out of the count — while its checkpoint sat on disk.
        """
        from sanzaru.exceptions import CostCeilingError

        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun"))

        # Below even one act's replayed spend, so the first charge aborts.
        with pytest.raises(CostCeilingError) as excinfo:
            await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=0.001))
        assert sorted(excinfo.value.completed_acts) == ["act1", "act2", "act3"]

    async def test_a_finished_act_keeps_its_whole_checkpoint_when_a_sibling_aborts(
        self, rundown, media_dir, monkeypatch
    ):
        """A checkpoint is two files, and a resume needs both.

        The ceiling cancels the acts recording alongside the one that tripped
        it, and an act cancelled between its mp3 and its sidecar leaves an
        orphan that `_load_checkpoint` discards — audio that was recorded,
        billed, and then paid for a second time on the resume that was supposed
        to recover it.
        """
        from sanzaru.cli._runtime import find_in_group
        from sanzaru.exceptions import CostCeilingError
        from sanzaru.infrastructure import FileSystemRepository

        act1_audio_written = anyio.Event()
        write_audio_file = FileSystemRepository.write_audio_file

        async def instrumented(self, filename, data):
            if filename.endswith("act1.json"):
                # Widen the gap between act1's two writes to a window the
                # sibling's abort is certain to land inside.
                await anyio.sleep(0.05)
            written = await write_audio_file(self, filename, data)
            if filename.endswith("act1.mp3"):
                act1_audio_written.set()
            return written

        calls: list[str] = []

        async def fake_run_act(brief, hosts, settings, **kwargs):
            calls.append(brief.id)
            if brief.id == "act3":
                await act1_audio_written.wait()
                kwargs["budget"].charge(RealtimeUsage(output_audio_tokens=500_000), settings.model)
            return _fake_act(brief.id)

        monkeypatch.setattr(FileSystemRepository, "write_audio_file", instrumented)
        monkeypatch.setattr(sim, "run_act", fake_run_act)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 — anyio wraps it in a group
            await sim.simulate_podcast(
                sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", max_cost_usd=0.02)
            )
        ceiling = find_in_group(excinfo.value, CostCeilingError)
        assert ceiling is not None
        assert (media_dir / "Stitch_Test_testrun_act1.mp3").exists()
        assert (media_dir / "Stitch_Test_testrun_act1.json").exists()
        # act1 finished checkpointing *after* the ceiling snapshotted its list,
        # and "N act(s) checkpointed and safe" has to count it anyway.
        assert "act1" in ceiling.completed_acts

        # The point of the pair landing together: the resume reuses act1 instead
        # of paying for it again.
        monkeypatch.setattr(FileSystemRepository, "write_audio_file", write_audio_file)
        calls.clear()
        resumed = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=100.0)
        )
        assert "act1" not in calls
        assert next(a.reused for a in resumed.acts if a.act_id == "act1") is True


@pytest.mark.integration
class TestResumeUnderTheRestoredCeiling:
    """A resume replays the checkpointed spend against the restored ceiling.

    That is what makes the ceiling survive a bare `--resume` — and what makes
    the abort's own hint a money-burning loop unless the ceiling is raised with
    it: the same acts are charged again, the missing ones are re-recorded, and
    the run trips at the same total having checkpointed nothing new.
    """

    async def _record(self, rundown, **kwargs):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, qc=False, run_id="testrun", **kwargs))

    def _drop_act(self, media_dir, act_id):
        for suffix in ("mp3", "json"):
            (media_dir / f"Stitch_Test_testrun_{act_id}.{suffix}").unlink()

    async def test_a_resume_that_cannot_finish_refuses_before_recording(self, rundown, media_dir, stub_run_act):
        from sanzaru.exceptions import CostCeilingError

        await self._record(rundown, max_cost_usd=100.0)
        self._drop_act(media_dir, "act2")
        stub_run_act.clear()

        # Above the $0.0128 the two surviving acts replay, below what the run
        # needs in total — exactly the shape a restored ceiling has after an abort.
        with pytest.raises(CostCeilingError) as excinfo:
            await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=0.015))
        assert stub_run_act == []  # not one session opened, not one dollar spent
        assert "max_cost_usd" in str(excinfo.value)
        assert excinfo.value.suggested_limit_usd is not None
        assert excinfo.value.suggested_limit_usd > 0.015

    async def test_the_suggested_ceiling_is_one_the_resume_completes_under(self, rundown, media_dir, stub_run_act):
        """The whole point of the number: following it has to end the loop."""
        from sanzaru.exceptions import CostCeilingError

        await self._record(rundown, max_cost_usd=100.0)
        self._drop_act(media_dir, "act2")

        with pytest.raises(CostCeilingError) as excinfo:
            await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=0.015))
        result = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="testrun", qc=False, max_cost_usd=excinfo.value.suggested_limit_usd)
        )
        assert len(result.acts) == 3

    async def test_a_ceiling_that_still_fits_records_the_missing_act(self, rundown, media_dir, stub_run_act):
        """The guard must not block a resume that would have finished."""
        await self._record(rundown, max_cost_usd=100.0)
        self._drop_act(media_dir, "act2")
        stub_run_act.clear()

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == ["act2"]
        assert result.cost.limit_usd == 100.0

    async def test_an_unpriceable_model_is_not_blocked_by_a_projection_it_cannot_make(
        self, rundown, media_dir, stub_run_act
    ):
        await self._record(rundown, max_cost_usd=100.0, model="some-future-model")
        self._drop_act(media_dir, "act2")
        stub_run_act.clear()

        result = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="testrun", qc=False))
        assert stub_run_act == ["act2"]
        assert result.cost.usd is None


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
        # The threshold accounts for extension: 4 planned turns reach 6, and 6
        # turns still cannot fill 600s.
        assert "stop on the turn cap short of its target" in caplog.text

    def test_turn_notes_may_address_the_turns_extension_creates(self):
        # 6 planned turns reach 9, so a note on turn 7 does fire — rejecting it
        # as out of range would be a validator disagreeing with the loop.
        brief = ActBrief(id="act1", title="t", topic="x", max_turns=6, turn_notes={7: "push harder"})
        assert brief.turn_notes[7] == "push harder"
        with pytest.raises(ValidationError):
            ActBrief(id="act1", title="t", topic="x", max_turns=6, turn_notes={9: "never fires"})

    def test_a_single_turn_act_does_not_warn_about_extension_it_cannot_use(self, caplog):
        # extension_cap(1) == 1, so the warning must not promise two turns.
        with caplog.at_level("WARNING", logger="sanzaru"):
            ActBrief(id="act1", title="t", topic="x", target_seconds=180, max_turns=1)
        assert "extended to 1 turns" in caplog.text

    def test_an_act_that_extension_can_fill_does_not_warn(self, caplog):
        # 8 turns extend to 12, which covers 150s at ~15s a turn. Warning here
        # would train the planner to over-budget turns it does not need.
        with caplog.at_level("WARNING", logger="sanzaru"):
            ActBrief(id="act1", title="t", topic="x", target_seconds=150, max_turns=8)
        assert "turn cap" not in caplog.text


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

    def test_an_act_length_no_act_could_hold_is_rejected_up_front(self):
        # Same cross-field guard RundownRequest carries: legal apart, impossible
        # together, and only discovered after the planner call is paid for.
        with pytest.raises(ValidationError, match="per act"):
            sim.SimulationBrief(premise="p", acts=2, target_minutes=200.0)

    def test_enough_acts_for_the_same_episode_is_accepted(self):
        assert sim.SimulationBrief(premise="p", acts=6, target_minutes=200.0).acts == 6

    def test_a_supplied_rundown_owns_its_own_act_lengths(self, rundown):
        # acts/target_minutes only feed pre-production, which a rundown skips.
        assert sim.SimulationBrief(rundown=rundown, acts=2, target_minutes=200.0).rundown is rundown

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

    def test_a_long_episodes_manifest_brief_survives_the_strip(self, rundown):
        """acts/target_minutes are pre-production inputs a rundown makes moot.

        Pydantic revalidates the nested brief when the manifest is built, and
        stripping the rundown is what re-arms the guard: a brief that was legal
        when it carried its own rundown died with a raw ValidationError at
        manifest-write time, after pre-production and before any recording.
        """
        brief = sim.SimulationBrief(rundown=rundown, run_id="testrun", acts=2, target_minutes=200.0)
        stored = brief.model_copy(update={"rundown": None, "resume": True})
        manifest = sim.RunManifest(run_id="testrun", slug="s", created=0.0, rundown=rundown, brief=stored)
        assert sim.RunManifest.model_validate_json(manifest.model_dump_json()).brief.target_minutes == 200.0
