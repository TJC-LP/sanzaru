"""Tests for the gpt-live-1 host: the Live API agent, its billing, and its seat in `run_act`.

Everything runs against `FakeLiveConnection` (tests/audio/conftest.py) — no SDK,
no websocket, no spend. The fake answers the agent's own client events with
scripted server events, so these cover the protocol the agent actually speaks.
"""

import anyio
import pytest

from sanzaru.audio.realtime import producer
from sanzaru.audio.realtime.agent import RealtimeAgent
from sanzaru.audio.realtime.budget import CostBudget
from sanzaru.audio.realtime.live_agent import (
    FRAME_MS,
    SPEECH_RMS_THRESHOLD,
    STEER_MAX_CHARS,
    STOP_CUE,
    TURN_CUE,
    TURN_NUDGE,
    LiveAgent,
    frame_rms,
)
from sanzaru.audio.realtime.pricing import ModelPrices, prices_for, project_usage, usage_cost
from sanzaru.audio.realtime.producer import SimulationSettings, run_act
from sanzaru.audio.realtime.types import (
    LIVE_VOICES,
    REALTIME_VOICES,
    ActBrief,
    HostSpec,
    RealtimeUsage,
    is_live_model,
)
from sanzaru.exceptions import CostCeilingError, RealtimeAPIError

pytestmark = pytest.mark.audio

SILENCE = 0.05
"""A silence gap short enough to keep the suite fast; the default is 1.2s."""

BYTES_PER_SECOND = 24000 * 2
FRAME_BYTES = BYTES_PER_SECOND * FRAME_MS // 1000


@pytest.fixture
def host():
    return HostSpec(id="avery", name="Avery", voice="marin", persona="You host.")


def _agent(conn, host, **overrides):  # type: ignore[no-untyped-def]
    kwargs = {
        "model": "gpt-live-1",
        "turn_seconds": 10.0,
        "sample_rate": 24000,
        "end_of_turn_silence_s": SILENCE,
        "start_wait_s": 0.5,
    }
    kwargs.update(overrides)
    return LiveAgent(host, conn, **kwargs)


# ---------- model routing ----------


@pytest.mark.unit
class TestIsLiveModel:
    @pytest.mark.parametrize("model", ["gpt-live-1", "gpt-live-1-2026-09-01", "gpt-live"])
    def test_live_prefix(self, model):
        assert is_live_model(model)

    @pytest.mark.parametrize("model", ["gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime", "live-1"])
    def test_realtime_and_strangers_are_not(self, model):
        assert not is_live_model(model)


@pytest.mark.unit
class TestVoices:
    def test_live_voices_are_a_superset_of_realtime_voices(self):
        assert set(REALTIME_VOICES) <= set(LIVE_VOICES)

    def test_a_live_only_voice_does_not_warn(self, caplog):
        with caplog.at_level("WARNING", logger="sanzaru"):
            HostSpec(id="h", name="H", voice="vesper")
        assert "not a known" not in caplog.text

    def test_an_unknown_voice_still_warns(self, caplog):
        with caplog.at_level("WARNING", logger="sanzaru"):
            HostSpec(id="h", name="H", voice="not-a-voice")
        assert "not a known realtime voice" in caplog.text


# ---------- the agent ----------


class TestConfigure:
    async def test_sends_session_start_and_waits_for_started(self, fake_live, host):
        conn = fake_live.Connection()
        async with _agent(conn, host) as agent:
            await agent.configure("You are Avery.")

            start = conn.sent_of("session.start")
            assert len(start) == 1
            session = start[0]["session"]
            assert isinstance(session, dict)
            assert session["model"] == "gpt-live-1"
            assert session["audio"] == {
                "format": {"type": "audio/pcm", "rate": 24000},
                "output": {"voice": "marin"},
            }
            instructions = session["instructions"]
            assert isinstance(instructions, str)
            assert instructions.startswith("You are Avery.")
            # The turn-taking contract the producer relies on rides along.
            assert "Speak ONLY after the producer" in instructions
            assert "under 10 seconds" in instructions

    async def test_an_error_before_started_raises(self, fake_live, host):
        conn = fake_live.Connection(start_error="unknown voice")
        async with _agent(conn, host) as agent:
            with pytest.raises(RealtimeAPIError, match="unknown voice"):
                await agent.configure("persona")

    async def test_configure_without_entering_is_a_programming_error(self, fake_live, host):
        agent = _agent(fake_live.Connection(), host)
        with pytest.raises(RuntimeError, match="enter"):
            await agent.configure("persona")


@pytest.mark.unit
class TestFrameRms:
    def test_silence_is_zero_and_speech_is_loud(self):
        from conftest import LIVE_LOUD_FRAME, LIVE_SILENT_FRAME

        assert frame_rms(LIVE_SILENT_FRAME) == 0.0
        assert frame_rms(LIVE_LOUD_FRAME) == pytest.approx(8000.0)
        assert frame_rms(b"") == 0.0
        assert frame_rms(b"\x01") == 0.0
        assert 200 < SPEECH_RMS_THRESHOLD < 500


class TestSpeak:
    async def test_collects_speech_and_transcript_and_ends_on_silence(self, fake_live, host):
        conn = fake_live.Connection(seconds=1.5, transcripts=["hello there friend"], lead_silence_s=0.3)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            spoken = await agent.speak()

        # Leading and trailing silence are trimmed: `seconds` is speech.
        assert spoken.seconds == pytest.approx(1.5)
        assert spoken.pcm[:2] != b"\x00\x00"
        assert spoken.pcm[-2:] != b"\x00\x00"
        assert spoken.text == "hello there friend"
        assert spoken.truncated is False
        # The cue is an instruction, the nudge is commentary; both go out,
        # each with an event_id so the ack (or the loss) can be matched.
        cues = conn.sent_of("session.instructions.append")
        assert [e["content"] for e in cues] == [TURN_CUE]
        assert [e["content"] for e in conn.sent_of("session.commentary.append")] == [TURN_NUDGE]
        assert all(e["delegation_id"] is None and e["event_id"] for e in cues)

    async def test_the_turn_only_starts_once_input_frames_are_flowing(self, fake_live, host):
        # Mirrors the real server: nothing comes out until audio goes in. The
        # clock is what makes the cue land; without it this turn would be empty.
        conn = fake_live.Connection(seconds=0.5)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            before = conn.input_frames
            spoken = await agent.speak()
            assert conn.input_frames > before
        assert spoken.seconds == pytest.approx(0.5)

    async def test_hard_cap_truncates_and_sends_a_stop_cue(self, fake_live, host):
        # 3s of audio against a 1s turn: the cap is 2 x turn_seconds.
        conn = fake_live.Connection(seconds=3.0)
        async with _agent(conn, host, turn_seconds=1.0) as agent:
            await agent.configure("persona")
            spoken = await agent.speak()

        assert spoken.truncated is True
        assert spoken.seconds == pytest.approx(2.0)
        assert STOP_CUE in [e["content"] for e in conn.sent_of("session.instructions.append")]

    async def test_off_floor_speech_is_discarded_but_counted(self, fake_live, host):
        conn = fake_live.Connection(seconds=1.0, pre_cue_seconds=0.7)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            # Let the clock tick a few times: the out-of-turn speech plays on
            # the first frame, then idle zero frames follow — those must not
            # count.
            await anyio.sleep(0.35)
            assert agent.off_floor_seconds == pytest.approx(0.7)
            spoken = await agent.speak()

            assert spoken.seconds == pytest.approx(1.0)
            assert agent.off_floor_seconds == pytest.approx(0.7)

    async def test_silence_after_the_cue_is_an_empty_turn_not_an_error(self, fake_live, host, caplog):
        # The stream keeps flowing (zero frames), the model just says nothing.
        conn = fake_live.Connection(silent=True)
        async with _agent(conn, host, start_wait_s=0.3) as agent:
            await agent.configure("persona")
            with caplog.at_level("WARNING", logger="sanzaru"):
                spoken = await agent.speak()

        assert spoken.pcm == b""
        assert spoken.truncated is False
        assert "no speech within" in caplog.text

    async def test_a_session_closed_mid_turn_raises(self, fake_live, host):
        conn = fake_live.Connection(seconds=2.0, close_mid_turn=True)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            with pytest.raises(RealtimeAPIError, match="closed before the turn finished"):
                await agent.speak()

    async def test_a_stream_that_ends_mid_turn_raises(self, fake_live, host):
        # No session.closed at all — the socket just went away.
        conn = fake_live.Connection(seconds=2.0, end_stream_mid_turn=True)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            with pytest.raises(RealtimeAPIError, match="without session.closed"):
                await agent.speak()

    async def test_a_faulted_session_refuses_the_next_turn(self, fake_live, host):
        conn = fake_live.Connection(seconds=1.0)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            await agent.speak()
            conn.emit_error("rate limited")
            await anyio.sleep(0.01)
            with pytest.raises(RealtimeAPIError, match="rate limited"):
                await agent.speak()


class TestUsage:
    async def test_usage_is_the_delta_of_cumulative_session_seconds(self, fake_live, host):
        conn = fake_live.Connection(seconds=1.0, usage_seconds=[30.0, 90.0], final_usage_seconds=100.0)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            first = await agent.speak()
            second = await agent.speak()
            tail = await agent.finish()

        # Both turns carry speech — a second turn that returned on the first
        # turn's speech count would read as empty here.
        assert first.seconds == pytest.approx(1.0)
        assert second.seconds == pytest.approx(1.0)
        assert first.usage.live_seconds == pytest.approx(30.0)
        assert second.usage.live_seconds == pytest.approx(60.0)
        assert tail.live_seconds == pytest.approx(10.0)
        assert first.usage.output_audio_tokens == 0
        # finish() closed the session and the server answered.
        assert conn.sent[-1] == {"type": "session.close"}
        assert conn.closed

    async def test_finish_is_idempotent(self, fake_live, host):
        conn = fake_live.Connection(seconds=1.0, usage_seconds=[30.0])
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            await agent.speak()
            assert (await agent.finish()).live_seconds == pytest.approx(0.0)
            assert (await agent.finish()).live_seconds == 0.0
        assert len(conn.sent_of("session.close")) == 1

    async def test_leaving_the_scope_closes_the_session(self, fake_live, host):
        conn = fake_live.Connection(seconds=1.0)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            await agent.speak()
        assert conn.sent_of("session.close")


class TestSteerAndHear:
    async def test_steer_sends_an_instructions_append_and_truncates(self, fake_live, host, caplog):
        conn = fake_live.Connection()
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            await agent.steer("short note")
            with caplog.at_level("WARNING", logger="sanzaru"):
                await agent.steer("x" * (STEER_MAX_CHARS + 500))

        notes = conn.sent_of("session.instructions.append")
        assert notes[0]["content"] == "short note"
        assert notes[0]["delegation_id"] is None
        assert notes[0]["event_id"]
        assert len(str(notes[1]["content"])) == STEER_MAX_CHARS
        assert "truncated" in caplog.text

    async def test_lost_injections_at_close_warn_by_name_and_do_not_raise(self, fake_live, host, caplog):
        conn = fake_live.Connection(seconds=0.3, lose_injections_at_close=True)
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            await agent.steer("land the plane")
            spoken = await agent.speak()
            with caplog.at_level("WARNING", logger="sanzaru"):
                tail = await agent.finish()

        assert spoken.seconds == pytest.approx(0.3)
        assert tail.live_seconds >= 0.0
        assert "lost: steer: 'land the plane'" in caplog.text
        assert "lost: turn cue" in caplog.text
        assert "lost: turn nudge" in caplog.text


class TestClock:
    async def test_frames_keep_flowing_while_idle(self, fake_live, host):
        conn = fake_live.Connection()
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            await anyio.sleep(0.35)
            assert agent.frames_sent >= 3
            assert conn.input_frames == agent.frames_sent
            # Idle frames are silence, one frame each.
            frames = conn.sent_of("session.input_audio.append")
            assert frames
            assert conn.heard_speech_bytes == 0
        # The clock stops with the session: no frames after close.
        after_close = [
            e for e in conn.sent[conn.sent.index({"type": "session.close"}) :] if e.get("type") != "session.close"
        ]
        assert after_close == []

    async def test_the_clock_starts_only_after_session_started(self, fake_live, host):
        conn = fake_live.Connection(start_error="bad voice")
        async with _agent(conn, host) as agent:
            with pytest.raises(RealtimeAPIError):
                await agent.configure("persona")
            await anyio.sleep(0.25)
        assert conn.input_frames == 0

    async def test_hear_plays_out_at_pace_and_returns_once_heard(self, fake_live, host):
        from conftest import LIVE_LOUD_FRAME

        conn = fake_live.Connection()
        async with _agent(conn, host) as agent:
            await agent.configure("persona")
            speech = LIVE_LOUD_FRAME * 3  # 0.3s
            started = anyio.current_time()
            await agent.hear(speech)
            await agent.hear(b"")
            elapsed = anyio.current_time() - started

        # Three frames at 100ms each, not one burst.
        assert 0.2 <= elapsed < 1.5
        assert agent.inbox_seconds == 0.0
        assert conn.heard_speech_bytes == len(speech)
        loud_frames = [
            e
            for e in conn.sent_of("session.input_audio.append")
            if any(__import__("base64").b64decode(str(e["audio"])))
        ]
        assert len(loud_frames) == 3
        assert all(len(__import__("base64").b64decode(str(e["audio"]))) == FRAME_BYTES for e in loud_frames)


class TestFanOut:
    async def test_listeners_hear_the_speaker_live(self, fake_live, host):
        speaker_conn = fake_live.Connection(seconds=0.3)
        listener_conn = fake_live.Connection()
        listener = LiveAgent(
            HostSpec(id="rory", name="Rory", voice="cedar"),
            listener_conn,
            model="gpt-live-1",
            turn_seconds=10.0,
            sample_rate=24000,
            end_of_turn_silence_s=SILENCE,
        )
        async with _agent(speaker_conn, host) as speaker, listener:
            await speaker.configure("persona")
            await listener.configure("persona")
            speaker.set_listeners([speaker, listener])  # self is dropped

            spoken = await speaker.speak()

            # By the time the turn is over, the listener's clock has played it.
            assert listener.inbox_seconds == 0.0
            assert listener_conn.heard_speech_bytes == len(spoken.pcm) == int(0.3 * BYTES_PER_SECOND)
            # And the listener recorded none of it as its own speech.
            assert listener.off_floor_seconds == 0.0

    async def test_only_frames_from_the_first_speech_frame_on_are_forwarded(self, fake_live, host):
        speaker_conn = fake_live.Connection(seconds=0.2, lead_silence_s=0.5, trail_silence_s=0.1)
        listener_conn = fake_live.Connection()
        listener = LiveAgent(
            HostSpec(id="rory", name="Rory", voice="cedar"),
            listener_conn,
            model="gpt-live-1",
            turn_seconds=10.0,
            sample_rate=24000,
            end_of_turn_silence_s=SILENCE,
        )
        async with _agent(speaker_conn, host) as speaker, listener:
            await speaker.configure("persona")
            await listener.configure("persona")
            speaker.set_listeners([listener])
            fed_before = listener_conn.input_frames
            await speaker.speak()
            # 0.2s speech + 0.1s trailing silence forwarded; 0.5s lead dropped.
            # The listener's idle silence frames are indistinguishable from
            # forwarded trailing silence, so count only the speech.
            assert listener_conn.heard_speech_bytes == int(0.2 * BYTES_PER_SECOND)
            assert listener_conn.input_frames > fed_before


# ---------- pricing ----------


@pytest.mark.unit
class TestLivePricing:
    def test_gpt_live_1_bills_per_minute(self):
        assert usage_cost(RealtimeUsage(live_seconds=90.0), "gpt-live-1") == pytest.approx(0.075)

    def test_tokens_are_free_on_a_live_model(self):
        assert usage_cost(RealtimeUsage(output_audio_tokens=1_000_000), "gpt-live-1") == 0.0

    def test_live_seconds_are_free_on_a_realtime_model(self):
        assert usage_cost(RealtimeUsage(live_seconds=600.0), "gpt-realtime-2.1") == 0.0

    def test_dated_snapshots_price_like_the_base(self):
        assert prices_for("gpt-live-1-2026-09-01") == prices_for("gpt-live-1")

    def test_seven_value_env_override(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_LIVE_1", "0,0,0,0,0,0,0.10")
        assert prices_for("gpt-live-1") == ModelPrices(0, 0, 0, 0, 0, 0, per_minute=0.10)
        assert usage_cost(RealtimeUsage(live_seconds=60.0), "gpt-live-1") == pytest.approx(0.10)

    def test_six_value_env_override_still_works(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1", "1,2,3,4,5,6")
        assert prices_for("gpt-realtime-2.1") == ModelPrices(1, 2, 3, 4, 5, 6, per_minute=0.0)

    def test_eight_values_are_rejected(self, monkeypatch, caplog):
        monkeypatch.setenv("SANZARU_REALTIME_PRICE_GPT_LIVE_1", "0,0,0,0,0,0,0.05,9")
        with caplog.at_level("WARNING", logger="sanzaru"):
            assert prices_for("gpt-live-1") == ModelPrices(0, 0, 0, 0, 0, 0, per_minute=0.05)
        assert "expected 6 or 7" in caplog.text

    def test_usage_addition_carries_live_seconds(self):
        total = RealtimeUsage(live_seconds=1.5) + RealtimeUsage(live_seconds=2.5, output_audio_tokens=3)
        assert total.live_seconds == pytest.approx(4.0)
        assert total.output_audio_tokens == 3

    def test_negative_live_seconds_are_rejected(self):
        with pytest.raises(ValueError):
            RealtimeUsage(live_seconds=-1.0)


@pytest.mark.unit
class TestLiveProjection:
    def test_a_live_model_projects_session_seconds_per_host_and_no_tokens(self):
        projected = project_usage(seconds=120.0, turns=8, hosts=2, model="gpt-live-1")
        assert projected.live_seconds == pytest.approx(240.0)
        assert projected.input_tokens == 0
        assert projected.output_tokens == 0

    def test_a_realtime_model_projects_tokens_as_before(self):
        projected = project_usage(seconds=120.0, turns=8, hosts=2, model="gpt-realtime-2.1")
        assert projected.live_seconds == 0.0
        assert projected.output_audio_tokens > 0
        assert projected == project_usage(seconds=120.0, turns=8, hosts=2)

    def test_project_run_prices_a_live_episode_by_the_minute(self):
        from sanzaru.audio.realtime.types import Rundown
        from sanzaru.tools.simulate_podcast import SimulationBrief, project_run

        rundown = Rundown(
            title="Live",
            hosts=[HostSpec(id="a", name="A"), HostSpec(id="b", name="B")],
            acts=[
                ActBrief(id="act1", title="One", topic="t", target_seconds=180.0, max_turns=8),
                ActBrief(id="act2", title="Two", topic="t", target_seconds=120.0, max_turns=8),
            ],
        )
        report = project_run(rundown, SimulationBrief(rundown=rundown, model="gpt-live-1"))
        assert report.usage.live_seconds == pytest.approx(600.0)
        assert report.usd == pytest.approx(0.5)
        assert report.unpriced_models == []


# ---------- the producer's seat ----------


@pytest.fixture
def hosts():
    return [
        HostSpec(id="avery", name="Avery", voice="marin", persona="You host."),
        HostSpec(id="rory", name="Rory", voice="cedar", persona="You engineer."),
    ]


@pytest.fixture
def brief():
    # Two 0.3s turns land it: the close is due once one more average turn
    # would reach the target.
    return ActBrief(id="act1", title="Open", topic="the topic", target_seconds=0.6, max_turns=2)


class TestRunActRouting:
    async def test_run_act_seats_live_agents_for_gpt_live_1(self, fake_live, connect_factory, brief, hosts, mocker):
        one = fake_live.Connection(seconds=0.3, usage_seconds=[20.0, 40.0], final_usage_seconds=45.0)
        two = fake_live.Connection(seconds=0.3, usage_seconds=[20.0], final_usage_seconds=30.0)
        factory, handed = connect_factory(one, two)
        seated = mocker.spy(producer, "_make_agent")
        settings = SimulationSettings(
            model="gpt-live-1", show_title="Live", turn_seconds=5.0, live_turn_silence_s=SILENCE
        )
        budget = CostBudget(limit_usd=10.0)

        result = await run_act(brief, hosts, settings, connect=factory, budget=budget, is_first_act=True)

        assert [type(agent) for agent in seated.spy_return_list] == [LiveAgent, LiveAgent]
        assert [turn.speaker_id for turn in result.turns] == ["avery", "rory"]
        assert result.seconds == pytest.approx(0.6)
        # Both hosts' sessions billed for the whole act: 45 + 30 seconds.
        assert result.usage.live_seconds == pytest.approx(75.0)
        assert budget.spent_usd == pytest.approx(75.0 / 60 * 0.05)
        for conn in handed:
            assert conn.sent[0]["type"] == "session.start"
            assert conn.sent[-1] == {"type": "session.close"}
        # Rory heard Avery's 0.3s turn exactly once — live, not live *and*
        # replayed — and vice versa.
        assert two.heard_speech_bytes == int(0.3 * BYTES_PER_SECOND)
        assert one.heard_speech_bytes == int(0.3 * BYTES_PER_SECOND)

    async def test_run_act_still_seats_realtime_agents_for_gpt_realtime(
        self, fake_realtime, connect_factory, brief, hosts, mocker
    ):
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        seated = mocker.spy(producer, "_make_agent")
        settings = SimulationSettings(model="gpt-realtime-2.1", show_title="RT", turn_seconds=5.0)

        result = await run_act(brief, hosts, settings, connect=factory)

        assert [type(agent) for agent in seated.spy_return_list] == [RealtimeAgent, RealtimeAgent]
        assert len(result.turns) == 2
        assert result.usage.live_seconds == 0.0

    async def test_a_per_host_override_can_seat_one_live_host(
        self, fake_live, fake_realtime, connect_factory, brief, mocker
    ):
        hosts = [
            HostSpec(id="avery", name="Avery", voice="marin", model="gpt-live-1"),
            HostSpec(id="rory", name="Rory", voice="cedar"),
        ]
        live_conn = fake_live.Connection(seconds=0.3)
        rt_conn = fake_realtime.Connection(seconds=0.3)
        factory, _ = connect_factory(live_conn, rt_conn)
        seated = mocker.spy(producer, "_make_agent")
        settings = SimulationSettings(model="gpt-realtime-2.1", turn_seconds=5.0, live_turn_silence_s=SILENCE)

        result = await run_act(brief, hosts, settings, connect=factory)

        assert [type(agent) for agent in seated.spy_return_list] == [LiveAgent, RealtimeAgent]
        assert len(result.turns) == 2
        # The realtime host got the live turn replayed; the live host heard
        # the realtime turn through its clock, at pace.
        assert rt_conn.heard_bytes > 0
        assert live_conn.heard_speech_bytes == int(0.3 * BYTES_PER_SECOND)

    def test_default_connect_dials_the_live_api_for_live_models(self, mocker):
        client = mocker.Mock()
        mocker.patch("sanzaru.config.get_client", return_value=client)

        producer._default_connect("gpt-live-1")
        client.live.connect.assert_called_once_with()
        client.realtime.connect.assert_not_called()

        producer._default_connect("gpt-realtime-2.1")
        client.realtime.connect.assert_called_once_with(model="gpt-realtime-2.1")


class TestCuedBudgetCountsListeners:
    """A Live session bills while it listens; the ceiling has to see that between turns."""

    async def test_listening_hosts_are_charged_before_another_turn_starts(
        self, fake_live, connect_factory, brief, hosts, monkeypatch
    ):
        # Both sessions have accrued a minute — the listener as much as the
        # speaker — so after Avery's turn the table stands at 2 x $0.05.
        monkeypatch.setattr(LiveAgent, "_billable_seconds", lambda self: 60.0)
        a = fake_live.Connection(seconds=0.3)
        b = fake_live.Connection(seconds=0.3)
        factory, _ = connect_factory(a, b)
        settings = SimulationSettings(model="gpt-live-1", turn_seconds=5.0, live_turn_silence_s=SILENCE)

        with pytest.raises(CostCeilingError):
            await run_act(brief, hosts, settings, connect=factory, budget=CostBudget(limit_usd=0.075))

        assert b.turn == 0, "a second turn started with $0.10 already accrued against a $0.075 ceiling"

    async def test_the_checkpoint_view_matches_what_the_budget_saw(self, fake_live, connect_factory, brief, hosts):
        a = fake_live.Connection(seconds=0.3, usage_seconds=[20.0, 40.0], final_usage_seconds=45.0)
        b = fake_live.Connection(seconds=0.3, usage_seconds=[20.0], final_usage_seconds=30.0)
        factory, _ = connect_factory(a, b)
        settings = SimulationSettings(model="gpt-live-1", turn_seconds=5.0, live_turn_silence_s=SILENCE)
        budget = CostBudget(limit_usd=10.0)

        result = await run_act(brief, hosts, settings, connect=factory, budget=budget)

        # Listening charges, turn charges and the tail all land in both views,
        # and nothing is charged twice: the per-model slices price to exactly
        # the budget's total, and sum to the pooled usage.
        assert set(result.usage_by_model) == {"gpt-live-1"}
        assert result.usage_by_model["gpt-live-1"].live_seconds == pytest.approx(result.usage.live_seconds)
        assert result.usage.live_seconds == pytest.approx(75.0)
        priced = sum(usage_cost(u, m) or 0.0 for m, u in result.usage_by_model.items())
        assert priced == pytest.approx(budget.spent_usd)
