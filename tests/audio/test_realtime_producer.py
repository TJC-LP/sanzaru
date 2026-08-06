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
from sanzaru.audio.realtime.types import MAX_ACT_TURNS, ActBrief, HostSpec, RealtimeUsage
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

    @pytest.mark.parametrize("max_turns", range(3, 13))
    @pytest.mark.parametrize("point_count", range(2, 9))
    def test_every_point_is_either_scheduled_or_reported(self, caplog, max_turns, point_count):
        """The one guarantee: no point disappears without anyone being told.

        `len(schedule) == len(set(values))` cannot see the old failure — an
        overwrite keeps both sides of that equal while losing a point — so this
        checks coverage against the brief instead of against the schedule.
        """
        points = [f"talking point {i}" for i in range(point_count)]
        brief = ActBrief(id="a", title="t", topic="x", talking_points=points, max_turns=max_turns, target_seconds=30.0)

        with caplog.at_level("WARNING", logger="sanzaru"):
            schedule = _point_schedule(brief)

        # Point 0 rides the opening note, so turns only ever carry points 1..N-1.
        assert len(schedule) == len(set(schedule.values()))
        for index in set(range(1, point_count)) - set(schedule.values()):
            assert points[index] in caplog.text

    @pytest.mark.parametrize(("max_turns", "point_count"), [(4, 3), (6, 5), (8, 4), (12, 8)])
    def test_places_every_point_when_the_act_has_room(self, max_turns, point_count):
        brief = ActBrief(
            id="a",
            title="t",
            topic="x",
            talking_points=[f"p{i}" for i in range(point_count)],
            max_turns=max_turns,
            target_seconds=30.0,
        )
        assert set(_point_schedule(brief).values()) == set(range(1, point_count))

    def test_points_stay_in_brief_order(self):
        brief = ActBrief(id="a", title="t", topic="x", talking_points=[f"p{i}" for i in range(5)], max_turns=8)
        schedule = _point_schedule(brief)
        by_turn = [point for _, point in sorted(schedule.items())]
        assert by_turn == sorted(by_turn)

    def test_an_act_too_short_for_its_points_names_what_it_will_drop(self, caplog):
        # max_turns=3 leaves exactly one steerable turn: turn 0 opens, turn 2 lands.
        brief = ActBrief(
            id="crowded",
            title="t",
            topic="x",
            talking_points=["the first thing", "the second thing", "the third thing"],
            max_turns=3,
            target_seconds=30.0,
        )
        with caplog.at_level("WARNING", logger="sanzaru"):
            schedule = _point_schedule(brief)
        assert schedule == {1: 1}
        assert "the third thing" in caplog.text
        assert "crowded" in caplog.text


@pytest.mark.unit
class TestVoiceTable:
    def test_the_producer_and_the_validator_read_one_table(self):
        """Two byte-identical tuples in one package drift; then `HostSpec` warns
        about a voice `assign_voices` handed out itself."""
        from sanzaru.audio.realtime import producer, types

        assert producer.DEFAULT_VOICES is types.REALTIME_VOICES


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
        # 6 planned turns extend to the 9-turn cap: 1s turns never fill the 60s target.
        assert [t.speaker_id for t in result.turns] == ["avery", "rory"] * 4 + ["avery"]

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
        # Nine turns (extended), five for avery and four for rory: each connection
        # hears only the other's turns.
        assert [name for name, _ in one.calls].count("input_audio_buffer.commit") == 4
        assert [name for name, _ in two.calls].count("input_audio_buffer.commit") == 5

    async def test_stops_at_the_extension_cap_having_delivered_a_closing(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        # 1s turns cannot fill a 60s target, so the act borrows extension turns
        # up to 1.5x the planned budget, then lands on a cued closing.
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory)
        assert len(result.turns) == 9
        assert any("Final turn" in note for conn in handed for note in conn.steers)
        # 9s of a 60s target: it closed because it ran out of turns, not because
        # it arrived. Reporting "complete" here would hide the exact undershoot
        # extension exists to prevent.
        assert result.stop_reason == "max_turns"

    async def test_extension_turns_fill_toward_the_time_target(self, fake_realtime, connect_factory, hosts, settings):
        # 5s turns against a 40s target with a 6-turn plan: the old hard stop
        # shipped 30s; extension lands the closing once one more average turn
        # would reach the target.
        act = ActBrief(id="fill", title="t", topic="x", target_seconds=40.0, max_turns=6)
        factory, handed = connect_factory(fake_realtime.Connection(seconds=5.0), fake_realtime.Connection(seconds=5.0))
        result = await run_act(act, hosts, settings, connect=factory)
        assert result.stop_reason == "target_seconds"
        assert len(result.turns) == 8
        assert result.seconds == 40.0

    async def test_a_short_act_still_opens_before_it_closes(self, fake_realtime, connect_factory, hosts, settings):
        # target_seconds only has to be > 0. Under a target smaller than one
        # turn, the predictive cue would make turn 0 the closing turn and the
        # act would open by signing off.
        act = ActBrief(id="tiny", title="t", topic="x", target_seconds=8.0, max_turns=6)
        factory, handed = connect_factory(fake_realtime.Connection(seconds=5.0), fake_realtime.Connection(seconds=5.0))
        result = await run_act(act, hosts, settings, connect=factory)
        assert len(result.turns) == 2
        assert "Final turn" not in handed[0].steers[0]

    async def test_a_single_turn_act_records_exactly_one_turn(self, fake_realtime, connect_factory, hosts, settings):
        # Rounding 1.5x up would turn max_turns=1 into two turns and move the
        # closing note off turn 0 — a caller planning one turn is stating a
        # shape, not estimating a duration.
        act = ActBrief(id="one", title="t", topic="x", target_seconds=60.0, max_turns=1)
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(act, hosts, settings, connect=factory)
        assert len(result.turns) == 1
        assert "Final turn" in handed[0].steers[0]

    async def test_the_last_act_still_signs_off_when_the_closing_note_is_blank(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        # "" means "say nothing this turn" — but the last act's hosts are told to
        # wait for a producer cue, so honoring it there ends the episode
        # mid-conversation.
        brief.turn_notes = {5: ""}
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory, is_last_act=True)
        assert any("sign off" in note for conn in handed for note in conn.steers)

    async def test_a_blank_closing_note_still_says_nothing_mid_episode(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        # Not the last act: nothing is waiting on a cue, so "" is honored.
        brief.turn_notes = {5: ""}
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory, is_last_act=False)
        assert not any("natural rest" in note for conn in handed for note in conn.steers)

    async def test_the_last_act_is_cued_to_close_wherever_the_close_lands(
        self, fake_realtime, connect_factory, hosts, settings
    ):
        """The cue belongs to the turn that lands the act, not to an index.

        With no note on the last planned turn, an early close used to hand the
        turn whatever mid-act note it carried — and the hosts are now told not
        to wrap up until cued, so the episode simply stopped.
        """
        act = ActBrief(
            id="early",
            title="t",
            topic="x",
            target_seconds=40.0,
            max_turns=6,
            turn_notes={1: "push back on the cost claim"},
        )
        factory, handed = connect_factory(
            fake_realtime.Connection(seconds=25.0), fake_realtime.Connection(seconds=25.0)
        )
        await run_act(act, hosts, settings, connect=factory, is_last_act=True)
        assert any("sign off" in note for conn in handed for note in conn.steers)

    async def test_a_middle_act_is_cued_to_its_handoff_wherever_the_close_lands(
        self, fake_realtime, connect_factory, hosts, settings
    ):
        act = ActBrief(
            id="early",
            title="t",
            topic="x",
            target_seconds=40.0,
            max_turns=6,
            handoff="on the open question",
            turn_notes={1: "push back on the cost claim"},
        )
        factory, handed = connect_factory(
            fake_realtime.Connection(seconds=25.0), fake_realtime.Connection(seconds=25.0)
        )
        await run_act(act, hosts, settings, connect=factory, is_last_act=False)
        assert any("on the open question" in note for conn in handed for note in conn.steers)

    async def test_points_skipped_by_an_early_close_are_reported(
        self, fake_realtime, connect_factory, hosts, settings, caplog
    ):
        # QC reports these as missed only after the act is paid for.
        act = ActBrief(
            id="early",
            title="t",
            topic="x",
            target_seconds=40.0,
            max_turns=8,
            talking_points=["the first thing", "the second thing", "the third thing"],
        )
        factory, _ = connect_factory(fake_realtime.Connection(seconds=25.0), fake_realtime.Connection(seconds=25.0))
        with caplog.at_level("WARNING", logger="sanzaru"):
            await run_act(act, hosts, settings, connect=factory)
        assert "never came up" in caplog.text
        assert "the third thing" in caplog.text

    async def test_a_maxed_out_act_short_of_target_still_reports_max_turns(
        self, fake_realtime, connect_factory, hosts, settings
    ):
        # At MAX_ACT_TURNS the extension clamp collapses hard_cap onto max_turns,
        # which must not read as "no cap was hit" — burning 200 turns short of
        # target is the loudest undershoot there is. (2 turns here; same shape.)
        act = ActBrief(id="clamped", title="t", topic="x", target_seconds=600.0, max_turns=MAX_ACT_TURNS)
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(act, hosts, settings, connect=factory)
        assert len(result.turns) == MAX_ACT_TURNS
        assert result.stop_reason == "max_turns"

    async def test_a_note_displaced_by_an_early_close_is_reported(
        self, fake_realtime, connect_factory, hosts, settings, caplog
    ):
        # 25s turns against a 40s target close on turn 1, which carries a note of
        # its own. The closing note wins, but dropping the caller's in silence is
        # the same failure the rotation pinning prevents.
        act = ActBrief(
            id="early",
            title="t",
            topic="x",
            target_seconds=40.0,
            max_turns=4,
            turn_notes={1: "push back on the cost claim", 3: "land it on the open question"},
        )
        factory, _ = connect_factory(fake_realtime.Connection(seconds=25.0), fake_realtime.Connection(seconds=25.0))
        with caplog.at_level("WARNING", logger="sanzaru"):
            await run_act(act, hosts, settings, connect=factory)
        assert "push back on the cost claim" in caplog.text
        assert "displaced" in caplog.text

    async def test_a_single_turn_acts_note_is_honored(self, fake_realtime, connect_factory, hosts, settings):
        # turn 0 IS `max_turns - 1` here, so it is the caller's closing takeover
        # — exempting single-turn acts from the lookup dropped their only note.
        act = ActBrief(id="one", title="t", topic="x", max_turns=1, turn_notes={0: "Say the number and stop."})
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(act, hosts, settings, connect=factory)
        assert handed[0].steers == ["Say the number and stop."]

    async def test_extension_turns_are_steered_away_from_recap(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory)
        notes = [note for conn in handed for note in conn.steers]
        assert any("Do not summarize" in note for note in notes)

    async def test_a_closing_override_rides_with_the_moved_closing_turn(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        # The caller landed the act on turn 5 of 6; extension moves the closing
        # to turn 8, and the caller's note must land there, not fire mid-act.
        brief.turn_notes = {5: "Land it on the open question."}
        factory, handed = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory)
        assert len(result.turns) == 9
        last_speaker_conn = handed[0]  # avery speaks turn 8
        assert last_speaker_conn.steers[-1] == "Land it on the open question."

    async def test_turn_notes_pin_the_rotation_without_speaking_order(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        # Index-keyed notes are written against a rotation; the per-act start
        # rotation must not silently reassign them to the other host.
        brief.turn_notes = {2: "Object to that, concretely."}
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory, start_index=1)
        assert result.turns[0].speaker_id == "avery"

    async def test_an_explicit_speaking_order_still_beats_turn_notes(
        self, fake_realtime, connect_factory, brief, hosts, settings
    ):
        # Pinning is a fallback for un-choreographed acts; a caller who wrote
        # the order down owns it, notes or not.
        brief.turn_notes = {2: "Object to that, concretely."}
        brief.speaking_order = ["rory", "avery"]
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        result = await run_act(brief, hosts, settings, connect=factory, start_index=1)
        assert result.turns[0].speaker_id == "rory"

    async def test_stops_on_the_duration_budget(self, fake_realtime, connect_factory, hosts, settings):
        # 40s turns against a 60s target: turn 1 is cued to close, because by
        # then the measured average says a third turn would land 60s over.
        long_act = ActBrief(id="act1", title="t", topic="x", target_seconds=60.0, max_turns=20)
        factory, _ = connect_factory(fake_realtime.Connection(seconds=40.0), fake_realtime.Connection(seconds=40.0))
        result = await run_act(long_act, hosts, settings, connect=factory)
        assert result.stop_reason == "target_seconds"
        assert len(result.turns) == 2

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
        assert result.usage.output_audio_tokens == 9 * 100

    async def test_on_turn_fires_per_turn(self, fake_realtime, connect_factory, brief, hosts, settings):
        seen = []
        factory, _ = connect_factory(fake_realtime.Connection(seconds=1.0), fake_realtime.Connection(seconds=1.0))
        await run_act(brief, hosts, settings, connect=factory, on_turn=seen.append)
        assert [t.index for t in seen] == [0, 1, 2, 3, 4, 5, 6, 7, 8]

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
        assert [t.speaker_id for t in result.turns] == [
            "avery",
            "rory",
            "rory",
            "avery",
            "avery",
            "rory",
            "rory",
            "avery",
            "avery",
        ]

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
