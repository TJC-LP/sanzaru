"""Tests for realtime floor control, steering, and act budgets.

Everything here runs against a fake connection object — no SDK, no websocket, no
spend — which is possible because the agent only ever touches five methods and
async iteration.
"""

import pytest

from sanzaru.audio.realtime.agent import RealtimeAgent
from sanzaru.audio.realtime.budget import CostBudget
from sanzaru.audio.realtime.producer import (
    SimulationSettings,
    _point_schedule,
    build_instructions,
    run_act,
    turn_timeout_seconds,
    turn_token_cap,
)
from sanzaru.audio.realtime.types import ActBrief, HostSpec, RealtimeUsage
from sanzaru.exceptions import CostCeilingError, RealtimeAPIError

pytestmark = pytest.mark.audio


@pytest.fixture
def hosts():
    return [
        HostSpec(id="avery", name="Avery", voice="marin", persona="You host."),
        HostSpec(id="rory", name="Rory", voice="cedar", persona="You engineer."),
    ]


@pytest.fixture
def brief():
    return ActBrief(
        id="act2",
        title="Middle",
        topic="the middle of it",
        talking_points=["first point", "second point", "third point"],
        target_seconds=60.0,
        max_turns=6,
        prior_context="Act 1 covered the setup.",
        upcoming="Act 3 owns the conclusion.",
        handoff="on the open question",
    )


@pytest.fixture
def settings():
    return SimulationSettings(model="gpt-realtime-2.1-mini", show_title="Stitch", turn_seconds=10.0)


# ---------- prompts ----------


@pytest.mark.unit
class TestBuildInstructions:
    def test_carries_the_cross_act_wiring(self, brief, hosts, settings):
        text = build_instructions(brief, hosts[0], hosts[1:], settings, is_first_act=False, is_last_act=False)
        assert "Act 1 covered the setup." in text
        assert "Act 3 owns the conclusion." in text
        assert "on the open question" in text
        assert "Rory" in text
        assert "You host." in text

    def test_later_acts_do_not_cold_open(self, brief, hosts, settings):
        text = build_instructions(brief, hosts[0], hosts[1:], settings, is_first_act=False, is_last_act=False)
        assert "Welcome listeners" not in text
        assert "Pick up mid-conversation" in text

    def test_first_act_opens_the_show(self, hosts, settings):
        first = ActBrief(id="act1", title="Open", topic="the top", max_turns=4)
        text = build_instructions(first, hosts[0], hosts[1:], settings, is_first_act=True, is_last_act=False)
        assert "top of the episode" in text

    def test_last_act_signs_off_and_ignores_handoff(self, brief, hosts, settings):
        text = build_instructions(brief, hosts[0], hosts[1:], settings, is_first_act=False, is_last_act=True)
        assert "sign off" in text
        assert "LEAVE THE CONVERSATION HERE" not in text

    def test_length_rule_uses_the_configured_target(self, brief, hosts, settings):
        text = build_instructions(brief, hosts[0], hosts[1:], settings, is_first_act=False, is_last_act=False)
        assert "Under 10 seconds" in text

    def test_caller_direction_reaches_the_agents(self, brief, hosts, settings):
        brief.direction = "Let this one get heated."
        text = build_instructions(brief, hosts[0], hosts[1:], settings, is_first_act=False, is_last_act=False)
        assert "Let this one get heated." in text

    def test_no_direction_adds_no_line(self, brief, hosts, settings):
        text = build_instructions(brief, hosts[0], hosts[1:], settings, is_first_act=False, is_last_act=False)
        assert "How to play it" not in text


@pytest.mark.unit
class TestTurnTokenCap:
    def test_scales_with_the_target_turn_length(self):
        assert turn_token_cap(30.0) == 2 * turn_token_cap(15.0)

    def test_budgets_transcript_as_well_as_audio(self):
        # 20 audio tok/s alone would give 300 for a 15s turn with 1.0 headroom;
        # the transcript is billed too, which is what a naive cap misses.
        assert turn_token_cap(15.0) > 15 * 20

    def test_has_a_floor(self):
        assert turn_token_cap(0.1) == 256


@pytest.mark.unit
class TestPointSchedule:
    def test_spreads_points_over_the_act(self, brief):
        schedule = _point_schedule(brief)
        # The opening note already puts point 0 on the table.
        assert 0 not in schedule
        assert sorted(schedule.values()) == [1, 2]
        assert len(set(schedule)) == len(schedule)

    def test_no_schedule_for_a_single_point(self):
        assert _point_schedule(ActBrief(id="a", title="t", topic="x", talking_points=["one"], max_turns=8)) == {}

    def test_never_collides_two_points_on_one_turn(self):
        brief = ActBrief(id="a", title="t", topic="x", talking_points=[f"p{i}" for i in range(5)], max_turns=6)
        schedule = _point_schedule(brief)
        assert len(schedule) == len(set(schedule.values()))


# ---------- the agent ----------


@pytest.mark.unit
class TestRealtimeAgent:
    async def test_speak_collects_audio_transcript_and_usage(self, fake_realtime, hosts):
        conn = fake_realtime.Connection(seconds=1.5, transcripts=["hello there"])
        agent = RealtimeAgent(hosts[0], conn, model="m", max_turn_tokens=400, sample_rate=24000)
        spoken = await agent.speak()
        assert spoken.text == "hello there"
        assert spoken.seconds == pytest.approx(1.5, abs=0.01)
        assert spoken.usage.output_audio_tokens == 100
        assert spoken.truncated is False

    async def test_incomplete_response_marks_the_turn_truncated(self, fake_realtime, hosts):
        conn = fake_realtime.Connection(status="incomplete")
        agent = RealtimeAgent(hosts[0], conn, model="m", max_turn_tokens=10, sample_rate=24000)
        assert (await agent.speak()).truncated is True

    async def test_failed_response_raises(self, fake_realtime, hosts):
        conn = fake_realtime.Connection(status="failed")
        agent = RealtimeAgent(hosts[0], conn, model="m", max_turn_tokens=400, sample_rate=24000)
        with pytest.raises(RealtimeAPIError, match="response failed"):
            await agent.speak()

    async def test_error_event_raises(self, fake_realtime, hosts):
        conn = fake_realtime.Connection(error="rate limited")
        agent = RealtimeAgent(hosts[0], conn, model="m", max_turn_tokens=400, sample_rate=24000)
        with pytest.raises(RealtimeAPIError, match="rate limited"):
            await agent.speak()

    async def test_a_stream_that_ends_without_response_done_raises(self, fake_realtime, hosts):
        # The SDK returns cleanly from __aiter__ on ConnectionClosedOK, so a
        # graceful mid-act close would otherwise be a silent zero-usage turn.
        conn = fake_realtime.Connection(seconds=1.0, end_early=True)
        agent = RealtimeAgent(hosts[0], conn, model="m", max_turn_tokens=400, sample_rate=24000)
        with pytest.raises(RealtimeAPIError, match="ended before response.done"):
            await agent.speak()

    async def test_hear_is_a_noop_for_empty_audio(self, fake_realtime, hosts):
        conn = fake_realtime.Connection()
        agent = RealtimeAgent(hosts[0], conn, model="m", max_turn_tokens=400, sample_rate=24000)
        await agent.hear(b"")
        assert conn.calls == []

    async def test_configure_disables_vad_and_sets_the_voice(self, fake_realtime, hosts):
        conn = fake_realtime.Connection()
        agent = RealtimeAgent(hosts[1], conn, model="m", max_turn_tokens=333, sample_rate=24000)
        await agent.configure("be someone")
        session = conn.calls[0][1]["session"]
        assert session["audio"]["input"]["turn_detection"] is None
        assert session["audio"]["output"]["voice"] == "cedar"
        assert session["max_output_tokens"] == 333
        assert session["output_modalities"] == ["audio"]


# ---------- floor control ----------


@pytest.mark.unit
class TestRunAct:
    async def test_alternates_the_floor(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory)
        assert [t.speaker_id for t in result.turns] == ["avery", "rory"] * 3

    async def test_start_index_rotates_who_opens(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory, start_index=1)
        assert result.turns[0].speaker_id == "rory"

    async def test_everyone_but_the_speaker_hears_each_turn(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        one, two = fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0)
        factory, _ = connect_factory(one, two)
        await run_act(brief, hosts, settings, connect=factory)
        # Six turns, three each: each connection hears only the other's three.
        assert [name for name, _ in one.calls].count("input_audio_buffer.commit") == 3
        assert [name for name, _ in two.calls].count("input_audio_buffer.commit") == 3

    async def test_stops_at_max_turns_having_delivered_a_closing(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory)
        assert len(result.turns) == 6
        assert result.stop_reason == "complete"
        assert any("Final turn" in note for conn in handed for note in conn.steers)

    async def test_stops_on_the_duration_budget(self, fake_realtime, connect_factory, hosts, settings):
        # 40s turns against a 60s target: two turns overshoot, then it lands.
        long_act = ActBrief(id="act1", title="t", topic="x", target_seconds=60.0, max_turns=20)
        factory, _ = connect_factory(fake_realtime.Connection(seconds=40.0), fake_realtime.Connection(seconds=40.0))
        result = await run_act(long_act, hosts, settings, connect=factory)
        assert result.stop_reason == "target_seconds"
        assert len(result.turns) == 3

    async def test_opening_note_differs_by_act_position(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory, is_first_act=False)
        assert "no greeting" in handed[0].steers[0]

        factory2, handed2 = connect_factory(
            fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0)
        )
        await run_act(brief, hosts, settings, connect=factory2, is_first_act=True)
        assert "Open the episode" in handed2[0].steers[0]

    async def test_talking_points_get_pushed_during_the_act(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        all_notes = " ".join(note for conn in handed for note in conn.steers)
        assert "second point" in all_notes
        assert "third point" in all_notes

    async def test_last_act_closing_note_signs_off(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory, is_last_act=True)
        notes = " ".join(note for conn in handed for note in conn.steers)
        assert "sign off warmly" in notes

    async def test_usage_accumulates_across_turns(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory)
        assert result.usage.output_audio_tokens == 6 * 100

    async def test_on_turn_fires_per_turn(self, fake_realtime, connect_factory, brief, hosts, settings):
        seen = []
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory, on_turn=seen.append)
        assert [t.index for t in seen] == [0, 1, 2, 3, 4, 5]

    async def test_budget_stops_the_act(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        budget = CostBudget(limit_usd=0.000001)
        with pytest.raises(CostCeilingError):
            await run_act(brief, hosts, settings, connect=factory, budget=budget)

    async def test_no_hosts_is_a_value_error(self, settings, brief):
        with pytest.raises(ValueError, match="no hosts"):
            await run_act(brief, [], settings)

    async def test_derives_the_token_cap_when_unset(self, fake_realtime, connect_factory, brief, hosts):
        settings = SimulationSettings(turn_seconds=20.0, max_turn_tokens=0)
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        assert handed[0].calls[0][1]["session"]["max_output_tokens"] == turn_token_cap(20.0)

    async def test_a_caller_note_replaces_the_generated_one(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        brief.turn_notes = {0: "Start with the thing nobody wants to say."}
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        assert handed[0].steers[0] == "Start with the thing nobody wants to say."

    async def test_an_empty_caller_note_suppresses_the_turn_entirely(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        brief.turn_notes = {0: ""}
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        # Turn 0 is Avery's; without an override she would always get an opener.
        assert not any("Open the episode" in note or "no greeting" in note for note in handed[0].steers)

    async def test_a_caller_note_can_take_over_the_closing_turn(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        brief.turn_notes = {5: "Stop mid-argument. No resolution."}
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        notes = [note for conn in handed for note in conn.steers]
        assert "Stop mid-argument. No resolution." in notes
        assert not any("Final turn" in note for note in notes)

    async def test_speaking_order_overrides_round_robin(self, fake_realtime, connect_factory, brief, hosts, settings):
        brief.speaking_order = ["avery", "rory", "rory", "avery"]
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory)
        assert [t.speaker_id for t in result.turns] == ["avery", "rory", "rory", "avery", "avery", "rory"]

    async def test_speaking_order_beats_start_index(self, fake_realtime, connect_factory, brief, hosts, settings):
        brief.speaking_order = ["rory"]
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory, start_index=0)
        assert {t.speaker_id for t in result.turns} == {"rory"}

    async def test_speaking_order_with_an_unknown_host_is_rejected(self, hosts, settings, brief):
        brief.speaking_order = ["avery", "nobody"]
        with pytest.raises(ValueError, match="not in the episode"):
            await run_act(brief, hosts, settings)

    async def test_explicit_token_cap_wins(self, fake_realtime, connect_factory, brief, hosts):
        settings = SimulationSettings(turn_seconds=20.0, max_turn_tokens=99)
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        assert handed[0].calls[0][1]["session"]["max_output_tokens"] == 99

    async def test_a_hung_turn_fails_the_act_instead_of_hanging_forever(
        self, fake_realtime, connect_factory, brief, hosts
    ):
        # Without a bound this test would never return: the act holds a limiter
        # slot inside a blocking tool with no job registry to cancel it.
        settings = SimulationSettings(turn_seconds=10.0, turn_timeout_s=0.05)
        factory, _ = connect_factory(fake_realtime.Connection(hang=True), fake_realtime.Connection(seconds=1.0))
        with pytest.raises(RealtimeAPIError, match="looks stalled"):
            await run_act(brief, hosts, settings, connect=factory)

    async def test_a_stream_that_ends_early_fails_the_act(self, fake_realtime, connect_factory, brief, hosts, settings):
        factory, _ = connect_factory(
            fake_realtime.Connection(seconds=1.0, end_early=True), fake_realtime.Connection(seconds=1.0)
        )
        with pytest.raises(RealtimeAPIError, match="ended before response.done"):
            await run_act(brief, hosts, settings, connect=factory)


@pytest.mark.unit
class TestTurnTimeout:
    def test_derives_a_generous_bound_from_the_target_turn_length(self):
        assert turn_timeout_seconds(120.0) == pytest.approx(720.0)

    def test_has_a_floor_so_short_turns_still_get_room(self):
        assert turn_timeout_seconds(5.0) == 60.0

    def test_an_explicit_setting_wins(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_TURN_TIMEOUT", "300")
        assert turn_timeout_seconds(15.0, explicit=12.0) == 12.0

    def test_the_env_var_overrides_the_derived_bound(self, monkeypatch):
        monkeypatch.setenv("SANZARU_REALTIME_TURN_TIMEOUT", "300")
        assert turn_timeout_seconds(15.0) == 300.0

    @pytest.mark.parametrize("value", ["", "  ", "nope", "0", "-5"])
    def test_junk_falls_back_to_the_derived_bound(self, monkeypatch, value):
        monkeypatch.setenv("SANZARU_REALTIME_TURN_TIMEOUT", value)
        assert turn_timeout_seconds(30.0) == pytest.approx(180.0)


# ---------- budget ----------


@pytest.mark.unit
class TestCostBudget:
    def test_accumulates_spend(self):
        budget = CostBudget()
        usage = RealtimeUsage(output_audio_tokens=1_000_000)
        budget.charge(usage, "gpt-realtime-2.1-mini")
        assert budget.spent_usd == pytest.approx(20.0)

    def test_unknown_model_is_reported_not_silently_free(self):
        budget = CostBudget(limit_usd=0.01)
        budget.charge(RealtimeUsage(output_audio_tokens=10_000_000), "some-future-model")
        assert budget.spent_usd == 0.0
        assert budget.unpriced_models == ["some-future-model"]

    def test_ceiling_carries_the_completed_acts(self):
        budget = CostBudget(limit_usd=0.001)
        budget.mark_act_complete("act1")
        with pytest.raises(CostCeilingError) as excinfo:
            budget.charge(RealtimeUsage(output_audio_tokens=1_000_000), "gpt-realtime-2.1-mini")
        assert excinfo.value.completed_acts == ["act1"]
        assert excinfo.value.limit_usd == 0.001

    def test_marking_one_act_twice_does_not_double_count_it(self):
        # --qc-retry re-records and re-checkpoints an act that is already there;
        # "2 act(s) checkpointed and safe" for one act is a number users act on.
        budget = CostBudget()
        budget.mark_act_complete("act1")
        budget.mark_act_complete("act1")
        assert budget.completed_acts == ["act1"]

    def test_completed_acts_keeps_the_order_they_landed_in(self):
        budget = CostBudget()
        for act_id in ("act2", "act1", "act2"):
            budget.mark_act_complete(act_id)
        assert budget.completed_acts == ["act2", "act1"]
