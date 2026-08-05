"""Tests for dialogue render mode: run grouping, batching, and rendering."""

import pytest

from sanzaru.audio.providers import get_provider
from sanzaru.tools.podcast import _plan_render_units, _validate_script

pytestmark = pytest.mark.audio


def speaker(sid, provider="openai", model=None, voice=None):
    s = {
        "id": sid,
        "name": sid.title(),
        "voice": voice or ("ash" if provider == "openai" else f"voice_{sid}"),
        "speed": 1.0,
        "instructions": "",
    }
    if provider != "openai":
        s["provider"] = provider
    if model:
        s["model"] = model
    return s


def plan(speakers, segment_speakers, render_mode="dialogue", texts=None):
    """Resolve providers/models the way generate_podcast does, then plan units."""
    speaker_map = {s["id"]: s for s in speakers}
    providers, models = {}, {}
    for s in speakers:
        p = get_provider(s.get("provider", "openai"))
        providers[s["id"]] = p
        models[s["id"]] = p.resolve_model(s.get("model") or (None if s.get("provider") else "gpt-4o-mini-tts"))
    segments = [
        {"speaker": sid, "text": (texts[i] if texts else f"Line {i}.")} for i, sid in enumerate(segment_speakers)
    ]
    return _plan_render_units(segments, speaker_map, providers, models, render_mode)


def shape(units):
    """Compact view: 'D' for a dialogue unit, 'S' for a segment unit."""
    return [("D" if u.is_dialogue else "S", u.indices) for u in units]


@pytest.mark.unit
class TestPlanRenderUnits:
    def test_segments_mode_is_one_unit_per_segment(self):
        units = plan([speaker("a", "elevenlabs")], ["a"] * 4, render_mode="segments")
        assert shape(units) == [("S", (0,)), ("S", (1,)), ("S", (2,)), ("S", (3,))]

    def test_consecutive_elevenlabs_turns_batch_into_one_dialogue(self):
        units = plan([speaker("a", "elevenlabs"), speaker("b", "elevenlabs")], ["a", "b", "a", "b"])
        assert shape(units) == [("D", (0, 1, 2, 3))]

    def test_openai_speaker_splits_the_run(self):
        """The mixed-provider guarantee: an OpenAI turn renders alone, and the
        ElevenLabs turns on either side still get dialogue treatment."""
        speakers = [speaker("host"), speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        units = plan(speakers, ["host", "a", "b", "a", "host", "a", "b"])
        assert shape(units) == [("S", (0,)), ("D", (1, 2, 3)), ("S", (4,)), ("D", (5, 6))]

    def test_lone_elevenlabs_turn_falls_back_to_segment(self):
        # One turn is one voice, so the distinct-voice rule already excludes it;
        # it also gains nothing from the endpoint and would lose its
        # per-speaker voice_settings.
        speakers = [speaker("host"), speaker("a", "elevenlabs")]
        units = plan(speakers, ["host", "a", "host"])
        assert shape(units) == [("S", (0,)), ("S", (1,)), ("S", (2,))]

    def test_non_dialogue_model_never_batches(self):
        # Only eleven_v3 supports dialogue.
        speakers = [
            speaker("a", "elevenlabs", model="eleven_multilingual_v2"),
            speaker("b", "elevenlabs", model="eleven_multilingual_v2"),
        ]
        units = plan(speakers, ["a", "b", "a"])
        assert shape(units) == [("S", (0,)), ("S", (1,)), ("S", (2,))]

    def test_non_dialogue_model_between_runs_splits_them(self):
        """An excluded turn ends the run in progress and a new one starts after
        it — the same guarantee as the OpenAI case, one layer down."""
        speakers = [
            speaker("a", "elevenlabs", model="eleven_v3"),
            speaker("b", "elevenlabs", model="eleven_v3"),
            speaker("c", "elevenlabs", model="eleven_multilingual_v2"),
        ]
        units = plan(speakers, ["a", "b", "c", "a", "b"])
        assert shape(units) == [("D", (0, 1)), ("S", (2,)), ("D", (3, 4))]

    def test_two_dialogue_capable_models_never_share_a_run(self, monkeypatch):
        """A dialogue request carries one model_id, so turns on different models
        must not batch together — even when both models can render dialogue.

        Only eleven_v3 is dialogue-capable today, so the run key can only differ
        by model if a second one is admitted; widening the set is how we reach
        the branch at all.
        """
        monkeypatch.setattr(
            "sanzaru.audio.providers.elevenlabs_provider.ELEVENLABS_DIALOGUE_MODELS",
            frozenset({"eleven_v3", "eleven_multilingual_v2"}),
        )
        speakers = [
            speaker("a", "elevenlabs", model="eleven_v3"),
            speaker("b", "elevenlabs", model="eleven_v3"),
            speaker("c", "elevenlabs", model="eleven_multilingual_v2"),
            speaker("d", "elevenlabs", model="eleven_multilingual_v2"),
        ]
        units = plan(speakers, ["a", "b", "c", "d"])
        assert shape(units) == [("D", (0, 1)), ("D", (2, 3))]

    def test_single_speaker_run_never_batches(self):
        """Consecutive turns by ONE voice have no turn-taking for the model to
        pace, so batching them buys nothing and would swallow their pause_after."""
        speakers = [speaker("a", "elevenlabs")]
        units = plan(speakers, ["a", "a", "a"])
        assert shape(units) == [("S", (0,)), ("S", (1,)), ("S", (2,))]

    def test_single_speaker_stretch_inside_a_two_voice_run_still_batches(self):
        """The rule is per run, not per adjacent pair: a monologue in the middle
        of a real exchange is still part of one conversation."""
        speakers = [speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        units = plan(speakers, ["a", "a", "a", "b"])
        assert shape(units) == [("D", (0, 1, 2, 3))]

    def test_turn_over_the_dialogue_budget_renders_alone(self):
        """A turn that alone fills the 2000-char request budget cannot share one
        with its neighbours, so it (and they) fall back to segment units and get
        chunked the normal way."""
        speakers = [speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        units = plan(speakers, ["a", "b", "a"], texts=["short", "x" * 2500, "short"])
        assert shape(units) == [("S", (0,)), ("S", (1,)), ("S", (2,))]

    def test_run_splits_at_the_character_budget(self):
        # 2000-char dialogue budget; 4 turns of 900 chars must split 2+2, always
        # at a turn boundary.
        speakers = [speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        units = plan(speakers, ["a", "b", "a", "b"], texts=["x" * 900] * 4)
        assert shape(units) == [("D", (0, 1)), ("D", (2, 3))]

    def test_run_exactly_at_the_character_budget_still_batches(self):
        # The budget is inclusive: 1000 + 1000 == 2000 is one request, not two.
        speakers = [speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        units = plan(speakers, ["a", "b"], texts=["x" * 1000, "y" * 1000])
        assert shape(units) == [("D", (0, 1))]

    def test_two_speakers_sharing_one_voice_never_batch(self):
        """The endpoint hears voices, not speaker ids. Two entries pointing at
        the same voice are a monologue, so batching them would buy nothing and
        swallow their pause_after."""
        speakers = [
            speaker("a", "elevenlabs", voice="shared_voice"),
            speaker("b", "elevenlabs", voice="shared_voice"),
        ]
        units = plan(speakers, ["a", "b", "a"])
        assert shape(units) == [("S", (0,)), ("S", (1,)), ("S", (2,))]

    def test_oversized_single_turn_still_becomes_its_own_unit(self):
        speakers = [speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        units = plan(speakers, ["a", "b"], texts=["x" * 6000, "short"])
        # The huge turn cannot share a request; each ends up alone.
        assert all(not u.is_dialogue for u in units)
        assert [u.indices for u in units] == [(0,), (1,)]

    def test_all_segments_are_covered_exactly_once_in_order(self):
        speakers = [speaker("host"), speaker("a", "elevenlabs"), speaker("b", "elevenlabs")]
        pattern = ["a", "b", "host", "a", "a", "b", "host", "host", "a", "b"]
        units = plan(speakers, pattern)
        covered = [i for u in units for i in u.indices]
        assert covered == list(range(len(pattern)))


@pytest.mark.unit
class TestValidateRenderMode:
    def test_unknown_render_mode_raises(self, minimal_script):
        minimal_script["config"]["render_mode"] = "conversation"
        with pytest.raises(ValueError, match="render_mode' must be one of"):
            _validate_script(minimal_script)

    def test_valid_render_modes(self, minimal_script):
        for mode in ("segments", "dialogue"):
            minimal_script["config"]["render_mode"] = mode
            _validate_script(minimal_script)

    @pytest.mark.parametrize("value", [-0.1, 1.5, "high"])
    def test_invalid_dialogue_stability(self, minimal_script, value):
        minimal_script["config"]["dialogue_stability"] = value
        with pytest.raises(ValueError, match="dialogue_stability' must be between"):
            _validate_script(minimal_script)

    def test_dialogue_stability_ignored_outside_dialogue_mode(self, minimal_script, caplog):
        minimal_script["config"]["dialogue_stability"] = 0.5
        _validate_script(minimal_script)
        assert "ignored unless render_mode is 'dialogue'" in caplog.text

    def test_voice_settings_warn_in_dialogue_mode(self, minimal_script, caplog):
        minimal_script["config"]["render_mode"] = "dialogue"
        minimal_script["speakers"][0].update(
            {"provider": "elevenlabs", "voice": "v1", "voice_settings": {"stability": 0.5}}
        )
        _validate_script(minimal_script)
        assert "cannot apply per speaker" in caplog.text

    def test_no_voice_settings_warning_for_a_model_that_cannot_batch(self, minimal_script, caplog):
        """eleven_multilingual_v2 never joins a dialogue run, so its
        voice_settings are guaranteed to be honoured — warning about them is a lie."""
        minimal_script["config"]["render_mode"] = "dialogue"
        minimal_script["speakers"][0].update(
            {
                "provider": "elevenlabs",
                "voice": "v1",
                "model": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5},
            }
        )
        _validate_script(minimal_script)
        assert "cannot apply per speaker" not in caplog.text


# ==================== RENDERING ====================


@pytest.fixture
def podcast_env(mocker, tmp_audio_path):
    from sanzaru.storage.local import LocalStorageBackend

    stitch = mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=b"STITCHED")
    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=storage)
    return stitch


class _Endpoint:
    """Records each convert() call and streams back a fixed payload."""

    def __init__(self, calls: list[dict[str, object]], payload: bytes):
        self.calls = calls
        self.payload = payload

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payload

        async def _stream():
            yield payload

        return _stream()


class FakeDialogueClient:
    """Fake ElevenLabs client exposing both endpoints, so a test can assert
    which one a given render mode actually used."""

    def __init__(self):
        self.dialogue_calls: list[dict[str, object]] = []
        self.speech_calls: list[dict[str, object]] = []
        self.text_to_dialogue = _Endpoint(self.dialogue_calls, b"DIALOGUE")
        self.text_to_speech = _Endpoint(self.speech_calls, b"SPEECH")


def dialogue_script(**config_overrides):
    config = {
        "default_pause_ms": 400,
        "normalize_loudness": True,
        "output_format": "mp3",
        "render_mode": "dialogue",
    }
    config.update(config_overrides)
    return {
        "title": "dialogue_ep",
        "speakers": [
            {
                "id": "a",
                "name": "Ann",
                "voice": "voice_a",
                "speed": 1.0,
                "instructions": "",
                "provider": "elevenlabs",
            },
            {
                "id": "b",
                "name": "Bob",
                "voice": "voice_b",
                "speed": 1.0,
                "instructions": "",
                "provider": "elevenlabs",
            },
        ],
        "segments": [
            {"speaker": "a", "text": "First turn."},
            {"speaker": "b", "text": "Second turn."},
            {"speaker": "a", "text": "Third turn."},
        ],
        "config": config,
    }


@pytest.mark.integration
@pytest.mark.anyio
class TestDialogueRendering:
    async def test_one_request_carries_every_turn(self, mocker, podcast_env):
        from sanzaru.tools.podcast import generate_podcast

        client = FakeDialogueClient()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        result = await generate_podcast(dialogue_script())

        assert len(client.dialogue_calls) == 1
        assert client.speech_calls == []
        call = client.dialogue_calls[0]
        assert [i.text for i in call["inputs"]] == ["First turn.", "Second turn.", "Third turn."]
        assert [i.voice_id for i in call["inputs"]] == ["voice_a", "voice_b", "voice_a"]
        assert call["model_id"] == "eleven_v3"
        assert call["output_format"] == "mp3_44100_128"
        # The envelope must not change shape between render modes.
        assert result.segment_count == 3
        assert result.speakers == ["Ann", "Bob"]

    async def test_intra_run_pauses_are_not_applied(self, mocker, podcast_env):
        """The model owns pacing inside a run, so we must not inject gaps."""
        from sanzaru.tools.podcast import generate_podcast

        mocker.patch(
            "sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client",
            return_value=FakeDialogueClient(),
        )

        script = dialogue_script()
        script["segments"][0]["pause_after"] = 5000  # would be a 5s hole mid-conversation

        await generate_podcast(script)

        kwargs = podcast_env.call_args.kwargs
        assert kwargs["segment_bytes_list"] == [b"DIALOGUE"]
        assert kwargs["pause_ms_list"] == [0]

    async def test_estimate_excludes_pauses_the_model_paces(self, mocker, podcast_env):
        """estimated_duration_seconds is user-visible, so it must count only the
        silence actually inserted: none, for a script that plans to one unit."""
        from sanzaru.tools.podcast import generate_podcast

        mocker.patch(
            "sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client",
            return_value=FakeDialogueClient(),
        )

        # 6 words at 1.0x -> 2.4s of speech; the three 400ms pause_afters are
        # inside the run (or trailing), so none of them reach the output.
        result = await generate_podcast(dialogue_script())

        assert result.estimated_duration_seconds == pytest.approx(2.4)

    async def test_estimate_counts_real_gaps_in_segments_mode(self, mocker, podcast_env):
        """Same script rendered per segment: two 400ms gaps, and still nothing
        trailing the last one."""
        from sanzaru.tools.podcast import generate_podcast

        mocker.patch(
            "sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client",
            return_value=FakeDialogueClient(),
        )

        result = await generate_podcast(dialogue_script(render_mode="segments"))

        assert result.estimated_duration_seconds == pytest.approx(2.4 + 0.8)

    async def test_dialogue_stability_is_forwarded(self, mocker, podcast_env):
        from sanzaru.tools.podcast import generate_podcast

        client = FakeDialogueClient()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await generate_podcast(dialogue_script(dialogue_stability=0.3))

        assert client.dialogue_calls[0]["settings"].stability == 0.3

    async def test_stability_omitted_when_unset(self, mocker, podcast_env):
        from sanzaru.tools.podcast import generate_podcast

        client = FakeDialogueClient()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await generate_podcast(dialogue_script())

        # The sentinel, not None — None would serialize as an explicit null.
        assert client.dialogue_calls[0]["settings"] is ...

    async def test_mixed_episode_uses_both_endpoints(self, mocker, podcast_env):
        """An OpenAI host between ElevenLabs guests: dialogue where possible,
        per-segment where not, stitched in script order."""
        from sanzaru.tools.podcast import generate_podcast

        client = FakeDialogueClient()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)
        response = mocker.MagicMock()
        response.content = b"OPENAI"
        openai_client = mocker.MagicMock()
        openai_client.audio.speech.create = mocker.AsyncMock(return_value=response)
        mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=openai_client)

        script = dialogue_script()
        script["speakers"].append({"id": "host", "name": "Host", "voice": "ash", "speed": 1.0, "instructions": "Warm"})
        # Distinct pauses inside the run: a unit's gap comes from its LAST turn,
        # so 111 must be swallowed by the model's own pacing and 222 kept.
        script["segments"] = [
            {"speaker": "host", "text": "Intro.", "pause_after": 100},
            {"speaker": "a", "text": "First.", "pause_after": 111},
            {"speaker": "b", "text": "Second.", "pause_after": 222},
            {"speaker": "host", "text": "Outro.", "pause_after": 333},
        ]

        await generate_podcast(script)

        assert len(client.dialogue_calls) == 1
        assert openai_client.audio.speech.create.await_count == 2
        # 3 units: openai segment, dialogue run, openai segment — in order.
        kwargs = podcast_env.call_args.kwargs
        assert kwargs["segment_bytes_list"] == [b"OPENAI", b"DIALOGUE", b"OPENAI"]
        # 333 is the trailing pause of the final unit, which never applies.
        assert kwargs["pause_ms_list"] == [100, 222, 0]

    async def test_falls_back_when_no_run_qualifies(self, mocker, podcast_env, caplog):
        from sanzaru.tools.podcast import generate_podcast

        response = mocker.MagicMock()
        response.content = b"OPENAI"
        openai_client = mocker.MagicMock()
        openai_client.audio.speech.create = mocker.AsyncMock(return_value=response)
        mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=openai_client)

        script = dialogue_script()
        script["speakers"] = [{"id": "a", "name": "Ann", "voice": "ash", "speed": 1.0, "instructions": ""}]
        script["segments"] = [{"speaker": "a", "text": "Only OpenAI here."}] * 2

        await generate_podcast(script)

        assert "no run of 2+ consecutive turns" in caplog.text
        assert openai_client.audio.speech.create.await_count == 2

    async def test_unbounded_concurrency_still_renders_dialogue(self, mocker, podcast_env, monkeypatch):
        """SANZARU_ELEVENLABS_MAX_CONCURRENCY=0 is the documented unbounded mode,
        which leaves the limiter None — the dialogue path must not assume one."""
        from sanzaru.tools.podcast import generate_podcast

        monkeypatch.setenv("SANZARU_ELEVENLABS_MAX_CONCURRENCY", "0")
        client = FakeDialogueClient()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await generate_podcast(dialogue_script())

        assert len(client.dialogue_calls) == 1
        assert podcast_env.call_args.kwargs["segment_bytes_list"] == [b"DIALOGUE"]

    async def test_segments_mode_never_touches_the_dialogue_endpoint(self, mocker, podcast_env):
        from sanzaru.tools.podcast import generate_podcast

        client = FakeDialogueClient()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await generate_podcast(dialogue_script(render_mode="segments"))

        assert client.dialogue_calls == []
        assert len(client.speech_calls) == 3


@pytest.mark.unit
class TestDialogueProviderGuards:
    def test_openai_is_not_a_dialogue_provider(self):
        from sanzaru.audio.providers import as_dialogue_provider

        assert as_dialogue_provider(get_provider("openai")) is None

    def test_elevenlabs_narrows(self):
        from sanzaru.audio.providers import as_dialogue_provider

        assert as_dialogue_provider(get_provider("elevenlabs")) is not None

    @pytest.mark.anyio
    async def test_rejects_non_dialogue_model(self):
        from sanzaru.audio.providers import DialogueTurn, as_dialogue_provider

        provider = as_dialogue_provider(get_provider("elevenlabs"))
        assert provider is not None
        with pytest.raises(ValueError, match="does not support dialogue"):
            await provider.synthesize_dialogue([DialogueTurn("hi", "v1")], "eleven_multilingual_v2")

    @pytest.mark.anyio
    async def test_rejects_empty_turns(self):
        from sanzaru.audio.providers import as_dialogue_provider

        provider = as_dialogue_provider(get_provider("elevenlabs"))
        assert provider is not None
        with pytest.raises(ValueError, match="at least one turn"):
            await provider.synthesize_dialogue([], "eleven_v3")

    @pytest.mark.anyio
    async def test_rejects_a_request_over_the_character_budget(self):
        """An over-budget dialogue can terminate the stream early, which reads
        as a short-but-successful take — so refuse before sending it."""
        from sanzaru.audio.providers import DialogueTurn, as_dialogue_provider

        provider = as_dialogue_provider(get_provider("elevenlabs"))
        assert provider is not None
        turns = [DialogueTurn("x" * 1500, "v1"), DialogueTurn("y" * 600, "v2")]
        with pytest.raises(ValueError, match="over the 2000-character limit"):
            await provider.synthesize_dialogue(turns, "eleven_v3")

    @pytest.mark.anyio
    async def test_rejects_out_of_range_stability(self):
        from sanzaru.audio.providers import DialogueTurn, as_dialogue_provider

        provider = as_dialogue_provider(get_provider("elevenlabs"))
        assert provider is not None
        with pytest.raises(ValueError, match="stability must be between"):
            await provider.synthesize_dialogue([DialogueTurn("hi", "v1")], "eleven_v3", stability=2.0)
