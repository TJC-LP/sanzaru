"""Tests for pre-production: premise → parallel-recordable acts."""

import pytest
from pydantic import ValidationError

from sanzaru.audio.realtime.rundown import (
    PlannedAct,
    PlannedHost,
    PlannedRundown,
    RundownRequest,
    _merge_hosts,
    _prompt,
    assign_rundown_voices,
    assign_voices,
    build_rundown,
    plan_rundown,
    slugify,
    turns_for,
)
from sanzaru.audio.realtime.types import MAX_ACT_SECONDS, MAX_ACTS, ActBrief, HostSpec, Rundown

pytestmark = pytest.mark.audio


def _planned(count: int = 3) -> PlannedRundown:
    return PlannedRundown(
        title="Stitch",
        style="dry and fast",
        hosts=[
            PlannedHost(name="Avery", role="host", persona="You host."),
            PlannedHost(name="Rory", role="engineer", persona="You engineer."),
        ],
        acts=[
            PlannedAct(
                title=f"Act {i + 1}",
                topic=f"topic {i + 1}",
                talking_points=[f"point {i}a", f"point {i}b"],
                prior_context=f"earlier acts covered {i}",
                handoff=f"leave it at {i}",
            )
            for i in range(count)
        ],
    )


@pytest.mark.unit
class TestTurnsFor:
    def test_budgets_one_turn_per_target_length_plus_one(self):
        # An act that runs out of seconds before turns stops on the duration
        # budget instead of on its closing turn, so this errs upward.
        assert turns_for(120.0, 15.0) == 9

    def test_has_a_floor(self):
        assert turns_for(1.0, 15.0) == 4

    def test_scales_with_duration(self):
        assert turns_for(600.0, 15.0) > turns_for(120.0, 15.0)


@pytest.mark.unit
class TestSlugify:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("Avery", "avery"), ("Dr. Rory Vale", "dr_rory_vale"), ("  ", "host"), ("!!!", "host")],
    )
    def test_slugs(self, name, expected):
        assert slugify(name) == expected


@pytest.mark.unit
class TestAssignVoices:
    def test_fills_gaps_without_collisions(self):
        hosts = [HostSpec(id="a", name="A", voice=""), HostSpec(id="b", name="B", voice="")]
        assigned = assign_voices(hosts)
        assert len({h.voice for h in assigned}) == 2
        assert all(h.voice for h in assigned)

    def test_keeps_an_explicit_voice_and_does_not_reuse_it(self):
        hosts = [HostSpec(id="a", name="A", voice="cedar"), HostSpec(id="b", name="B", voice="")]
        assigned = assign_voices(hosts)
        assert assigned[0].voice == "cedar"
        assert assigned[1].voice != "cedar"

    def test_a_host_who_states_no_voice_is_not_a_host_who_chose_marin(self):
        """The default `voice` has to be empty, or nothing here can fire.

        A named default made every unspecified host truthy, so `assign_voices`
        read all three as deliberate choices and the whole cast recorded in one
        voice — a failure you can only hear, after paying for the audio.
        """
        hosts = [HostSpec(id=f"h{i}", name=f"H{i}") for i in range(3)]
        assert [h.voice for h in assign_voices(hosts)] == ["marin", "cedar", "alloy"]


@pytest.mark.unit
class TestAssignRundownVoices:
    def _rundown(self, hosts):
        return Rundown(title="t", hosts=hosts, acts=[ActBrief(id="act1", title="One", topic="x")])

    def test_a_hand_written_rundown_gets_distinct_voices(self):
        # Exactly what `podcast rundown -o plan.json`, hand-edited, parses to:
        # names and personas, no voices.
        rundown = Rundown.model_validate(
            {
                "title": "Stitch",
                "hosts": [{"id": "avery", "name": "Avery"}, {"id": "rory", "name": "Rory"}],
                "acts": [{"id": "act1", "title": "One", "topic": "x"}],
            }
        )
        voices = [h.voice for h in assign_rundown_voices(rundown).hosts]
        assert voices == ["marin", "cedar"]

    def test_an_explicit_voice_survives_and_is_not_handed_out_twice(self):
        rundown = self._rundown([HostSpec(id="a", name="A", voice="cedar"), HostSpec(id="b", name="B")])
        voices = [h.voice for h in assign_rundown_voices(rundown).hosts]
        assert voices[0] == "cedar"
        assert voices[1] != "cedar"

    def test_a_fully_voiced_rundown_is_returned_untouched(self):
        rundown = self._rundown([HostSpec(id="a", name="A", voice="ash"), HostSpec(id="b", name="B", voice="echo")])
        assert assign_rundown_voices(rundown) is rundown


@pytest.mark.unit
class TestMergeHosts:
    def test_supplied_hosts_win(self):
        supplied = [HostSpec(id="x", name="Xavier", voice="ash", persona="mine")]
        merged = _merge_hosts(_planned().hosts, supplied)
        assert [h.name for h in merged] == ["Xavier"]
        assert merged[0].persona == "mine"

    def test_planner_fills_a_missing_persona(self):
        supplied = [HostSpec(id="x", name="Xavier", voice="ash", persona="")]
        merged = _merge_hosts(_planned().hosts, supplied)
        assert merged[0].persona == "You host."

    def test_duplicate_names_do_not_collapse_into_one_speaker(self):
        planned = [PlannedHost(name="Sam", role="a", persona="p1"), PlannedHost(name="Sam", role="b", persona="p2")]
        merged = _merge_hosts(planned, [])
        assert len({h.id for h in merged}) == 2

    def test_planner_invented_hosts_get_distinct_voices(self):
        merged = _merge_hosts(_planned().hosts, [])
        assert len({h.voice for h in merged}) == len(merged)

    def test_supplied_hosts_who_named_no_voice_get_distinct_ones(self):
        # `--host "Avery::persona"` and a hand-written brief both land here.
        supplied = [HostSpec(id="x", name="Xavier"), HostSpec(id="y", name="Yvonne")]
        merged = _merge_hosts(_planned().hosts, supplied)
        assert [h.voice for h in merged] == ["marin", "cedar"]


@pytest.mark.unit
class TestBuildRundown:
    def test_assigns_mechanics_and_splits_the_clock(self):
        request = RundownRequest(premise="p", acts=3, target_minutes=6.0)
        rundown = build_rundown(_planned(3), request)
        assert [a.id for a in rundown.acts] == ["act1", "act2", "act3"]
        assert all(a.target_seconds == pytest.approx(120.0) for a in rundown.acts)
        assert all(a.max_turns == turns_for(120.0, request.turn_seconds) for a in rundown.acts)

    def test_first_act_has_no_prior_context_and_last_has_no_handoff(self):
        rundown = build_rundown(_planned(3), RundownRequest(premise="p", acts=3))
        assert rundown.acts[0].prior_context == ""
        assert rundown.acts[1].prior_context != ""
        assert rundown.acts[-1].handoff == ""
        assert rundown.acts[0].handoff != ""

    def test_explicit_title_and_style_override_the_planner(self):
        request = RundownRequest(premise="p", acts=1, title="Mine", style="mine too")
        rundown = build_rundown(_planned(1), request)
        assert rundown.title == "Mine"
        assert rundown.style == "mine too"

    def test_extra_planned_acts_are_dropped(self):
        rundown = build_rundown(_planned(5), RundownRequest(premise="p", acts=2))
        assert len(rundown.acts) == 2

    def test_no_acts_is_an_error(self):
        empty = PlannedRundown(title="t", style="", hosts=_planned().hosts, acts=[])
        with pytest.raises(ValueError, match="no acts"):
            build_rundown(empty, RundownRequest(premise="p"))

    def test_totals(self):
        rundown = build_rundown(_planned(3), RundownRequest(premise="p", acts=3, target_minutes=6.0))
        assert rundown.total_target_seconds() == pytest.approx(360.0)
        assert rundown.total_max_turns() == 3 * rundown.acts[0].max_turns


@pytest.mark.unit
class TestPrompt:
    def test_explains_why_prior_context_matters(self):
        text = _prompt(RundownRequest(premise="p", acts=3))
        assert "SEPARATELY and IN PARALLEL" in text
        assert "prior_context" in text
        assert "handoff" in text

    def test_states_the_turn_budget_so_points_are_right_sized(self):
        text = _prompt(RundownRequest(premise="p", acts=3, target_minutes=6.0))
        assert "3-4 concrete talking points" in text
        assert "120 seconds" in text

    def test_fixed_hosts_are_named(self):
        request = RundownRequest(premise="p", hosts=[HostSpec(id="a", name="Avery", persona="You host.")])
        assert "Avery" in _prompt(request)

    def test_invents_hosts_when_none_supplied(self):
        assert "Invent two hosts" in _prompt(RundownRequest(premise="p"))


@pytest.mark.unit
class TestRundownRequestValidation:
    """Bounds live on the model, so they reject before any API call *and* show
    up in the JSON schema a caller reads."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"premise": "  "}, "premise is required"),
            ({"premise": ""}, "at least 1 character"),
            ({"premise": "p", "acts": 0}, "greater than or equal to 1"),
            ({"premise": "p", "acts": MAX_ACTS + 1}, f"less than or equal to {MAX_ACTS}"),
            ({"premise": "p", "target_minutes": 0}, "greater than 0"),
            ({"premise": "p", "target_minutes": 10_000}, "less than or equal to"),
            ({"premise": "p", "turn_seconds": 0.5}, "greater than or equal to 2"),
        ],
    )
    def test_rejects_unusable_requests(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            RundownRequest(**kwargs)

    def test_accepts_the_boundaries(self):
        assert RundownRequest(premise="p", acts=MAX_ACTS).acts == MAX_ACTS
        assert RundownRequest(premise="p", acts=1).acts == 1

    def test_an_act_length_no_act_could_hold_is_rejected_before_the_planner_call(self):
        """`acts` and `target_minutes` are each legal; their quotient is not.

        `build_rundown` divides one by the other into `ActBrief.target_seconds`,
        which is bounded — so without this the planner call is paid for and
        *then* the request turns out to have been impossible all along.
        """
        with pytest.raises(ValidationError, match="per act"):
            RundownRequest(premise="p", acts=2, target_minutes=200.0)

    def test_the_same_episode_split_into_enough_acts_is_fine(self):
        request = RundownRequest(premise="p", acts=6, target_minutes=200.0)
        assert request.target_minutes * 60.0 / request.acts <= MAX_ACT_SECONDS

    async def test_the_impossible_split_never_reaches_the_client(self, mocker):
        mocker.patch("sanzaru.config.get_client", side_effect=AssertionError("planner must not be called"))
        with pytest.raises(ValidationError):
            await plan_rundown(RundownRequest(premise="p", acts=1, target_minutes=120.0))

    async def test_plan_rundown_no_longer_needs_its_own_guards(self, mocker):
        """A valid request reaches the API; an invalid one never gets built."""
        mocker.patch(
            "sanzaru.config.get_client",
            side_effect=AssertionError("should have been rejected before the client"),
        )
        with pytest.raises(ValidationError):
            await plan_rundown(RundownRequest(premise="p", acts=0))
