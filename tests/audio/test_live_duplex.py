"""Tests for duplex recording of gpt-live-1 tables, and the cued loop's seam fixes.

Everything runs against `FakeLiveConnection` in its scripted mode — each host
speaks on a timeline, or answers once it hears the other stop — so mixing,
alignment, transcript grouping, silent steering, collisions and checkpointing
are covered without a socket or a cent of spend.
"""

import contextlib
import hashlib
import json
from io import BytesIO

import anyio
import pytest
from conftest import LIVE_LOUD_FRAME, LIVE_SILENT_FRAME

from sanzaru.audio.realtime import duplex, producer
from sanzaru.audio.realtime.duplex import build_duplex_rules, group_utterances, plan_notes, run_duplex_act
from sanzaru.audio.realtime.live_agent import (
    FALSE_START_MAX_TRIM_S,
    LiveAgent,
    mix_pcm16,
    trim_false_start,
)
from sanzaru.audio.realtime.producer import SimulationSettings, run_act
from sanzaru.audio.realtime.types import ActBrief, ActResult, HostSpec, RealtimeUsage, Turn, TurnAudio
from sanzaru.exceptions import RealtimeAPIError
from sanzaru.tools import simulate_podcast as sim

pytestmark = pytest.mark.audio

SILENCE = 0.05
BPS = 24000 * 2
FRAME = len(LIVE_SILENT_FRAME)


@pytest.fixture
def hosts():
    return [
        HostSpec(id="avery", name="Avery", voice="marin", persona="You host."),
        HostSpec(id="rory", name="Rory", voice="cedar", persona="You engineer."),
    ]


@pytest.fixture
def brief():
    return ActBrief(
        id="act1",
        title="Open",
        topic="the topic",
        talking_points=["first", "second", "third"],
        target_seconds=2.0,
        max_turns=2,
    )


def _settings(**overrides):  # type: ignore[no-untyped-def]
    kwargs = {"model": "gpt-live-1", "show_title": "Duplex", "turn_seconds": 5.0, "live_turn_silence_s": SILENCE}
    kwargs.update(overrides)
    return SimulationSettings(**kwargs)


# ---------- pure pieces ----------


@pytest.mark.unit
class TestMixPcm16:
    def test_sums_and_clips(self):
        loud = LIVE_LOUD_FRAME
        mixed = mix_pcm16([loud, loud, loud, loud, loud])  # 5 x 8000 clips at 32767
        assert len(mixed) == len(loud)
        assert int.from_bytes(mixed[:2], "little", signed=True) == 32767

    def test_pads_to_the_longest_and_passes_a_single_stream_through(self):
        short, long = LIVE_LOUD_FRAME[:100], LIVE_SILENT_FRAME
        mixed = mix_pcm16([short, long])
        assert len(mixed) == len(long)
        assert mixed[:100] == short
        assert mix_pcm16([short]) == short
        assert mix_pcm16([]) == b""


@pytest.mark.unit
class TestTrimFalseStart:
    def test_drops_a_short_burst_before_sustained_voice(self):
        pcm = LIVE_LOUD_FRAME * 2 + LIVE_SILENT_FRAME * 3 + LIVE_LOUD_FRAME * 10
        trimmed = trim_false_start(pcm, BPS, FRAME)
        assert trimmed == LIVE_LOUD_FRAME * 10

    def test_keeps_a_turn_that_starts_with_sustained_voice(self):
        pcm = LIVE_LOUD_FRAME * 10 + LIVE_SILENT_FRAME * 2
        assert trim_false_start(pcm, BPS, FRAME) == pcm

    def test_never_trims_past_the_cap(self):
        # Sustained voice only begins 1.5s in: past FALSE_START_MAX_TRIM_S, so
        # the head is kept — that is a real opener, not a stutter.
        head = (LIVE_LOUD_FRAME + LIVE_SILENT_FRAME) * 7 + LIVE_SILENT_FRAME
        assert len(head) / BPS > FALSE_START_MAX_TRIM_S
        pcm = head + LIVE_LOUD_FRAME * 10
        assert trim_false_start(pcm, BPS, FRAME) == pcm


@pytest.mark.unit
class TestGroupUtterances:
    def test_groups_by_gap_and_orders_across_hosts(self):
        fragments = {
            "a": [(0, 500, "Hello "), (600, 1200, "there."), (5000, 5500, "Later.")],
            "b": [(2000, 2600, "Hi "), (2700, 3000, "back.")],
        }
        turns = group_utterances(fragments, {"a": 0.0, "b": 0.0})
        assert [(t[0], t[3]) for t in turns] == [("a", "Hello there."), ("b", "Hi back."), ("a", "Later.")]
        assert turns[0][1:3] == (0.0, 1.2)
        assert turns[2][1:3] == (5.0, 5.5)

    def test_offsets_move_a_late_session_onto_the_act_clock(self):
        # b's session started 1s after the act clock, so its 0ms is act 1.0s.
        fragments = {"a": [(1500, 2000, "second")], "b": [(0, 400, "first")]}
        turns = group_utterances(fragments, {"a": 0.0, "b": 1.0})
        assert [t[3] for t in turns] == ["first", "second"]
        assert turns[0][1] == pytest.approx(1.0)

    def test_empty_text_is_dropped(self):
        assert group_utterances({"a": [(0, 100, "  ")]}, {"a": 0.0}) == []


@pytest.mark.unit
class TestPlanNotes:
    def test_opener_close_and_wrap_land_where_the_clock_says(self, brief, hosts):
        notes = plan_notes(brief, hosts, [0, 1], is_first_act=True, is_last_act=False)
        assert [n.at_s for n in notes] == sorted(n.at_s for n in notes)
        opens = [n for n in notes if n.kind == "open"]
        assert opens[0].host_ids == ("avery",) and opens[0].at_s == 0.0
        assert "Open the episode" in opens[0].text
        assert opens[1].host_ids == ("rory",) and "Avery opens" in opens[1].text
        close = [n for n in notes if n.kind == "close"]
        # max_turns=2: the closer is whoever takes the last planned turn — rory.
        assert close[0].host_ids == ("rory",) and close[0].at_s == pytest.approx(1.7)
        assert "Start landing" in close[0].text and "Do not sign off" in close[0].text
        assert close[1].host_ids == ("avery",) and "Rory is landing" in close[1].text
        wrap = [n for n in notes if n.kind == "wrap"]
        assert wrap[0].at_s == pytest.approx(2.0) and wrap[0].host_ids == ("avery", "rory")
        assert "segment is over" in wrap[0].text

    def test_talking_points_are_spread_before_the_close_with_rotating_bringers(self, brief, hosts):
        notes = plan_notes(brief, hosts, [0, 1], is_first_act=False, is_last_act=False)
        points = [n for n in notes if n.kind == "point"]
        assert [n.text.split(":")[1].split(" —")[0].strip() for n in points] == ["second", "third"]
        assert all(0 < n.at_s < 1.7 for n in points)
        assert "Rory raises it" in points[0].text and "Avery raises it" in points[1].text
        assert all(n.host_ids == ("avery", "rory") for n in points)

    def test_turn_notes_go_to_their_hosts_and_closing_note_overrides(self, hosts):
        brief = ActBrief(
            id="a",
            title="t",
            topic="x",
            target_seconds=100.0,
            max_turns=4,
            turn_notes={1: "push back hard", 3: "takeover"},
        )
        notes = plan_notes(brief, hosts, [0, 1], is_first_act=False, is_last_act=True)
        turn_notes = [n for n in notes if n.kind == "turn_note"]
        assert [(n.host_ids, n.text, n.at_s) for n in turn_notes] == [(("rory",), "push back hard", 25.0)]
        close = next(n for n in notes if n.kind == "close" and len(n.host_ids) == 1)
        # Index max_turns-1 without closing_note is the takeover: it is the close.
        assert close.text.endswith("takeover")
        with_closing = plan_notes(
            brief.model_copy(update={"closing_note": "land it on cats"}),
            hosts,
            [0, 1],
            is_first_act=False,
            is_last_act=True,
        )
        close = next(n for n in with_closing if n.kind == "close" and len(n.host_ids) == 1)
        assert close.text.endswith("land it on cats")
        # ...and index 3 is an ordinary turn note again.
        assert any(n.kind == "turn_note" and n.text == "takeover" for n in with_closing)

    def test_duplex_rules_name_the_opener_and_forbid_reading_notes(self, hosts):
        rules = build_duplex_rules(hosts[0], hosts[1:], hosts[0], 12.0)
        assert "Avery opens the segment" in rules
        assert "You open" in rules
        assert "never read them aloud" in rules
        assert "around 12 seconds" in rules
        assert "You open" not in build_duplex_rules(hosts[1], hosts[:1], hosts[0], 12.0)


# ---------- the duplex act ----------


class TestDuplexAct:
    async def test_hosts_hear_each_other_live_and_the_act_is_a_mix(self, fake_live, connect_factory, brief, hosts):
        a = fake_live.Connection(timeline=[(0.2, 0.5)], transcripts=["Welcome, Rory."])
        b = fake_live.Connection(reply_seconds=0.5, transcripts=["Thanks, Avery."])
        factory, _ = connect_factory(a, b)

        result = await run_duplex_act(brief, hosts, _settings(), connect=factory, is_first_act=True)

        assert result.is_mixed
        assert set(result.stems) == {"avery", "rory"}
        assert all(len(stem) == len(result.mixed_pcm or b"") for stem in result.stems.values())
        assert result.mixed_pcm == mix_pcm16(list(result.stems.values()))
        assert result.seconds > 1.0
        # The transcript comes from the timestamps, in order, one turn each.
        assert [(t.speaker_id, t.text) for t in result.turns] == [
            ("avery", "Welcome, Rory."),
            ("rory", "Thanks, Avery."),
        ]
        assert result.turns[0].seconds == pytest.approx(0.5, abs=0.11)
        assert result.turns[1].seconds == pytest.approx(0.5, abs=0.11)
        assert all(ta.pcm == b"" for ta in result.audio)
        assert result.stop_reason == "target_seconds"
        assert result.collision_seconds == 0.0
        # Each heard the other's speech through its own clock, exactly once.
        assert b.heard_speech_bytes == pytest.approx(0.5 * BPS, abs=2 * FRAME)
        assert a.heard_speech_bytes == pytest.approx(0.5 * BPS, abs=2 * FRAME)

    async def test_steering_is_silent_thinking_only(self, fake_live, connect_factory, brief, hosts):
        a = fake_live.Connection(timeline=[(0.2, 0.4)])
        b = fake_live.Connection(reply_seconds=0.3)
        factory, _ = connect_factory(a, b)

        await run_duplex_act(brief, hosts, _settings(), connect=factory, is_first_act=True)

        for conn in (a, b):
            assert conn.sent_of("session.instructions.append") == []
            assert conn.sent_of("session.commentary.append") == []
            kinds = [str(e["content"]) for e in conn.sent_of("session.thinking.append")]
            assert kinds, "no producer notes reached the host"
        assert any(
            "You open the segment now" in c for c in (str(e["content"]) for e in a.sent_of("session.thinking.append"))
        )
        assert any("Avery opens" in c for c in (str(e["content"]) for e in b.sent_of("session.thinking.append")))
        assert any("Start landing" in c for c in (str(e["content"]) for e in b.sent_of("session.thinking.append")))
        # The session instructions carry the duplex contract, not the cued one.
        instructions = str(a.sent_of("session.start")[0]["session"]["instructions"])  # type: ignore[index]
        assert "LIVE CONVERSATION" in instructions
        assert "Speak ONLY after the producer" not in instructions

    async def test_sustained_overlap_is_counted_and_the_interrupter_is_told_to_yield(
        self, fake_live, connect_factory, hosts
    ):
        brief = ActBrief(id="act1", title="t", topic="x", target_seconds=3.0, max_turns=2)
        a = fake_live.Connection(timeline=[(0.2, 2.5)])
        b = fake_live.Connection(timeline=[(0.6, 2.4)])
        factory, _ = connect_factory(a, b)

        result = await run_duplex_act(brief, hosts, _settings(), connect=factory)

        assert result.collision_seconds == pytest.approx(2.0, abs=0.5)
        yields_b = [e for e in b.sent_of("session.thinking.append") if "talking over Avery" in str(e["content"])]
        yields_a = [e for e in a.sent_of("session.thinking.append") if "talking over" in str(e["content"])]
        assert len(yields_b) == 1 and yields_a == []

    async def test_a_table_that_never_lands_is_cut_at_the_hard_cap(self, fake_live, connect_factory, hosts):
        brief = ActBrief(id="act1", title="t", topic="x", target_seconds=1.0, max_turns=2)
        # Talks straight through the close and the wrap.
        a = fake_live.Connection(timeline=[(0.1, 60.0)])
        b = fake_live.Connection()
        factory, _ = connect_factory(a, b)

        result = await run_duplex_act(brief, hosts, _settings(act_budget_s=2.0), connect=factory)

        assert result.stop_reason == "wall_clock"
        assert 1.8 <= result.seconds <= 3.0

    async def test_usage_is_charged_per_host_including_the_tail(self, fake_live, connect_factory, brief, hosts):
        from sanzaru.audio.realtime.budget import CostBudget

        a = fake_live.Connection(timeline=[(0.2, 0.3)], final_usage_seconds=30.0)
        b = fake_live.Connection(reply_seconds=0.3, final_usage_seconds=20.0)
        factory, _ = connect_factory(a, b)
        budget = CostBudget(limit_usd=5.0)

        result = await run_duplex_act(brief, hosts, _settings(), connect=factory, budget=budget)

        assert result.usage.live_seconds == pytest.approx(50.0)
        assert budget.spent_usd == pytest.approx(50.0 / 60 * 0.05)

    async def test_run_act_dispatches_by_mode_and_table(
        self, fake_live, fake_realtime, connect_factory, brief, hosts, mocker
    ):
        spy = mocker.spy(duplex, "run_duplex_act")

        factory, _ = connect_factory(
            fake_live.Connection(timeline=[(0.2, 0.3)]), fake_live.Connection(reply_seconds=0.3)
        )
        await run_act(brief, hosts, _settings(live_mode="duplex"), connect=factory)
        assert spy.call_count == 1

        factory, _ = connect_factory(fake_live.Connection(seconds=0.3), fake_live.Connection(seconds=0.3))
        await run_act(brief, hosts, _settings(live_mode="cued"), connect=factory)
        assert spy.call_count == 1

        mixed = [hosts[0].model_copy(update={"model": "gpt-live-1"}), hosts[1]]
        factory, _ = connect_factory(fake_live.Connection(seconds=0.3), fake_realtime.Connection(seconds=0.3))
        await run_act(brief, mixed, _settings(model="gpt-realtime-2.1", live_mode="duplex"), connect=factory)
        assert spy.call_count == 1

    async def test_hard_cap_interrupts_stalled_steering(self, fake_live, connect_factory, brief, hosts):
        """Finding 4: a `thinking.append` that never returns must not hold the loop past the cap."""

        class StalledSteer(fake_live.Connection):  # type: ignore[misc,name-defined]
            async def send(self, event):  # type: ignore[no-untyped-def]
                if event.get("type") == "session.thinking.append":
                    await anyio.sleep_forever()
                await super().send(event)

        a = StalledSteer(timeline=[(0.2, 0.4)])
        b = fake_live.Connection(reply_seconds=0.3)
        factory, _ = connect_factory(a, b)

        with anyio.move_on_after(1.0) as watchdog:
            with contextlib.suppress(RealtimeAPIError):
                await run_duplex_act(brief, hosts, _settings(act_budget_s=0.2), connect=factory)
        assert not watchdog.cancel_called, "the 0.2-second act cap never interrupted the blocked send"

    async def test_the_opening_is_kept_while_a_slower_peer_starts(self, fake_live, connect_factory, brief, hosts):
        """Finding 5: no host's timeline moves until every host is up, wired, and recording."""

        class SlowStart(fake_live.Connection):  # type: ignore[misc,name-defined]
            async def send(self, event):  # type: ignore[no-untyped-def]
                if event.get("type") == "session.start":
                    await anyio.sleep(1.0)
                await super().send(event)

        a = fake_live.Connection(timeline=[(0.2, 0.4)], transcripts=["Opening words."])
        b = SlowStart(reply_seconds=0.3)
        factory, _ = connect_factory(a, b)

        result = await run_duplex_act(brief, hosts, _settings(), connect=factory)

        assert result.turns[0].text == "Opening words."
        assert any(result.mixed_pcm or b""), "the returned act contains only silence despite an opening transcript"
        assert b.heard_speech_bytes > 0, "the co-host never heard the opening"
        # Nobody streamed a frame before the slow host was ready.
        assert a.input_frames <= b.input_frames + 2

    async def test_a_disconnect_during_the_close_is_an_error(self, fake_live, connect_factory, brief, hosts):
        """Finding 6: two sessions dying during the wrap look exactly like "everyone went quiet"."""

        class DropAfterWrap(fake_live.Connection):  # type: ignore[misc,name-defined]
            drop_next = False

            async def send(self, event):  # type: ignore[no-untyped-def]
                await super().send(event)
                if event.get("type") == "session.thinking.append" and "segment is over" in str(event.get("content")):
                    self.drop_next = True
                elif event.get("type") == "session.input_audio.append" and self.drop_next:
                    self._end_stream()

        a = DropAfterWrap(timeline=[(0.2, 4.0)])
        b = DropAfterWrap(timeline=[(0.4, 4.0)])
        factory, _ = connect_factory(a, b)

        with pytest.raises(RealtimeAPIError):
            await run_duplex_act(brief, hosts, _settings(act_budget_s=3.0), connect=factory)

    async def test_duplex_needs_two_hosts(self, fake_live, connect_factory, brief, hosts):
        factory, _ = connect_factory(fake_live.Connection())
        with pytest.raises(ValueError, match="two hosts"):
            await run_duplex_act(brief, hosts[:1], _settings(), connect=factory)


# ---------- cued-loop seam fixes ----------


class TestCuedSeams:
    async def test_steer_uses_the_silent_channel(self, fake_live, hosts):
        conn = fake_live.Connection()
        agent = LiveAgent(hosts[0], conn, model="gpt-live-1", turn_seconds=5.0, sample_rate=24000)
        async with agent:
            await agent.configure("persona")
            await agent.steer("land the plane")
        assert [e["content"] for e in conn.sent_of("session.thinking.append")] == ["land the plane"]
        assert conn.sent_of("session.instructions.append") == []

    async def test_a_settle_gap_precedes_the_cue_after_hearing_a_turn(self, fake_live, hosts):
        conn = fake_live.Connection(seconds=0.3)
        agent = LiveAgent(
            hosts[0],
            conn,
            model="gpt-live-1",
            turn_seconds=5.0,
            sample_rate=24000,
            end_of_turn_silence_s=SILENCE,
            settle_s=0.4,
        )
        async with agent:
            await agent.configure("persona")
            await agent.hear(LIVE_LOUD_FRAME * 2)
            heard_at = anyio.current_time()
            await agent.speak()
        cue_index = next(i for i, e in enumerate(conn.sent) if e.get("type") == "session.instructions.append")
        assert conn.sent_at[cue_index] - heard_at >= 0.4

    async def test_the_first_cue_of_an_act_does_not_wait(self, fake_live, hosts):
        conn = fake_live.Connection(seconds=0.3)
        agent = LiveAgent(
            hosts[0],
            conn,
            model="gpt-live-1",
            turn_seconds=5.0,
            sample_rate=24000,
            end_of_turn_silence_s=SILENCE,
            settle_s=5.0,
        )
        async with agent:
            await agent.configure("persona")
            started = anyio.current_time()
            await agent.speak()
            assert anyio.current_time() - started < 3.0


# ---------- the tool: a duplex act through checkpoint and resume ----------


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    from sanzaru.config import get_path

    path = tmp_path / "audio"
    path.mkdir()
    monkeypatch.setenv("AUDIO_PATH", str(path))
    get_path.cache_clear()
    yield path
    get_path.cache_clear()


def _duplex_act(act_id: str) -> ActResult:
    stems = {
        "avery": LIVE_LOUD_FRAME * 10 + LIVE_SILENT_FRAME * 10,
        "rory": LIVE_SILENT_FRAME * 10 + LIVE_LOUD_FRAME * 10,
    }
    turns = [
        Turn(
            act_id=act_id, index=0, speaker_id="avery", speaker_name="Avery", text=f"{act_id} avery says", seconds=1.0
        ),
        Turn(
            act_id=act_id, index=1, speaker_id="rory", speaker_name="Rory", text=f"{act_id} rory replies", seconds=1.0
        ),
    ]
    result = ActResult(
        act_id=act_id,
        audio=[TurnAudio(turn=t, pcm=b"") for t in turns],
        stop_reason="target_seconds",
        mixed_pcm=mix_pcm16(list(stems.values())),
        stems=stems,
        collision_seconds=0.3,
    )
    result.add_usage("gpt-live-1", RealtimeUsage(live_seconds=40.0))
    return result


class TestDuplexCheckpoints:
    @pytest.fixture
    def rundown(self, hosts):
        from sanzaru.audio.realtime.types import Rundown

        return Rundown(
            title="Duplex Test",
            hosts=hosts,
            acts=[ActBrief(id="act1", title="One", topic="t", target_seconds=2.0, max_turns=2)],
        )

    @pytest.fixture
    def stub_duplex(self, monkeypatch):
        calls: list[str] = []

        async def fake_run_act(brief, hosts, settings, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(brief.id)
            result = _duplex_act(brief.id)
            budget = kwargs.get("budget")
            if budget is not None:
                budget.charge(result.usage, settings.model)
            return result

        monkeypatch.setattr(sim, "run_act", fake_run_act)
        return calls

    async def test_a_duplex_act_checkpoints_as_a_mix_and_resumes_whole(self, rundown, media_dir, stub_duplex):
        brief = sim.SimulationBrief(rundown=rundown, model="gpt-live-1", qc=False, run_id="dup", stems=True)
        first = await sim.simulate_podcast(brief)

        meta = json.loads((media_dir / "Duplex_Test_dup_act1.json").read_text())
        assert meta["mode"] == "duplex"
        assert meta["collision_seconds"] == 0.3
        assert [t["speaker_id"] for t in meta["turns"]] == ["avery", "rory"]
        # Finding 2 applies to duplex too: the spend is on disk per model.
        assert meta["usage_by_model"] == {"gpt-live-1": meta["usage"]}
        assert meta["usage_by_model"]["gpt-live-1"]["live_seconds"] == 40.0
        # The per-host streams are checkpointed beside the mix, digested and signed.
        assert meta["stems"] == {
            "avery": "Duplex_Test_dup_act1_stem_avery.mp3",
            "rory": "Duplex_Test_dup_act1_stem_rory.mp3",
        }
        for host, name in meta["stems"].items():
            data = (media_dir / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == meta["stem_sha256"][host]
        checkpoint = sim.ActCheckpoint.model_validate(meta)
        swapped = checkpoint.model_copy(update={"stem_sha256": {**checkpoint.stem_sha256, "rory": "0" * 64}})
        assert swapped.signed_payload() != checkpoint.signed_payload()
        assert first.acts[0].mode == "duplex" and first.acts[0].collision_seconds == 0.3
        assert first.duration_seconds == pytest.approx(2.0, abs=0.1)
        assert "act1 avery says" in first.transcript and "act1 rory replies" in first.transcript
        assert set(first.stems) == {"avery", "rory"}

        stub_duplex.clear()
        resumed = await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="dup", qc=False, stems=True))
        assert stub_duplex == []
        assert resumed.acts[0].reused is True
        assert resumed.acts[0].mode == "duplex"
        assert resumed.turn_count == 2
        assert resumed.duration_seconds == pytest.approx(2.0, abs=0.15)
        assert resumed.cost.usage.live_seconds == pytest.approx(40.0)
        assert resumed.cost.usd == pytest.approx(first.cost.usd)
        assert "act1 rory replies" in resumed.transcript

    async def test_resume_preserves_previously_exported_duplex_stems(self, rundown, media_dir, stub_duplex):
        """The reviewer's probe: a resumed duplex act must not render silence over recorded stems."""
        from pydub import AudioSegment

        first = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, model="gpt-live-1", qc=False, stems=True, run_id="stemreview")
        )
        stem_path = media_dir / first.stems["avery"]
        before = stem_path.read_bytes()
        assert AudioSegment.from_file(stem_path).rms > 0

        stub_duplex.clear()
        resumed = await sim.simulate_podcast(
            sim.SimulationBrief(resume=True, run_id="stemreview", qc=False, stems=True)
        )

        assert stub_duplex == []
        assert resumed.acts[0].reused
        assert resumed.stems["avery"] == first.stems["avery"]
        after_segment = AudioSegment.from_file(stem_path)
        after_rms = after_segment.rms
        assert after_rms > 0, f"resume overwrote {len(before)} bytes of recorded speech with a silent stem"
        # Re-rendered from the checkpointed stem — one more mp3 generation, so
        # not byte-identical: the same speech, same length, still audible. (The
        # fixture's +/-8000 alternating samples are a 12kHz tone that mp3 at
        # 24kHz mostly filters, so absolute levels here are tiny either way.)
        before_segment = AudioSegment.from_file(BytesIO(before))
        assert after_rms > before_segment.rms * 0.5
        assert len(after_segment) == pytest.approx(len(before_segment), abs=60)

    async def test_a_legacy_duplex_checkpoint_keeps_the_exported_stem_rather_than_silencing_it(
        self, rundown, media_dir, stub_duplex, caplog
    ):
        from pydub import AudioSegment

        first = await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, model="gpt-live-1", qc=False, stems=True, run_id="legacy")
        )
        # A checkpoint from before stems were kept: strip the stem entries and files.
        meta_path = media_dir / "Duplex_Test_legacy_act1.json"
        meta = json.loads(meta_path.read_text())
        for name in meta.pop("stems").values():
            (media_dir / name).unlink()
        meta.pop("stem_sha256")
        meta_path.write_text(json.dumps(meta))
        stem_path = media_dir / first.stems["avery"]
        before = stem_path.read_bytes()

        with caplog.at_level("WARNING", logger="sanzaru"):
            resumed = await sim.simulate_podcast(
                sim.SimulationBrief(resume=True, run_id="legacy", qc=False, stems=True)
            )

        assert resumed.acts[0].reused
        assert stem_path.read_bytes() == before
        assert AudioSegment.from_file(stem_path).rms > 0
        assert resumed.stems["avery"] == first.stems["avery"]
        assert first.stems["avery"] in caplog.text and "keeping the existing file" in caplog.text

    async def test_a_swapped_checkpoint_stem_is_caught(self, rundown, media_dir, stub_duplex, caplog):
        await sim.simulate_podcast(sim.SimulationBrief(rundown=rundown, model="gpt-live-1", qc=False, run_id="swap"))
        (media_dir / "Duplex_Test_swap_act1_stem_rory.mp3").write_bytes(
            (media_dir / "Duplex_Test_swap_act1_stem_avery.mp3").read_bytes()
        )
        stub_duplex.clear()
        with caplog.at_level("WARNING", logger="sanzaru"):
            await sim.simulate_podcast(sim.SimulationBrief(resume=True, run_id="swap", qc=False))
        # Not shipped: the act was re-recorded instead.
        assert stub_duplex == ["act1"]
        assert "does not match the digest" in caplog.text

    async def test_live_mode_is_restored_from_the_manifest(self, rundown, media_dir, stub_duplex):
        await sim.simulate_podcast(
            sim.SimulationBrief(rundown=rundown, model="gpt-live-1", live_mode="cued", qc=False, run_id="cm")
        )
        manifest = json.loads((media_dir / "simrun_cm.json").read_text())
        assert manifest["brief"]["live_mode"] == "cued"
        assert "live_mode" in sim._SIGNED_BRIEF_FIELDS

    def test_the_settings_carry_the_mode(self):
        assert SimulationSettings().live_mode == "duplex"
        assert sim.SimulationBrief(premise="p").live_mode == "duplex"
        assert producer.SimulationSettings(live_mode="cued").live_mode == "cued"
