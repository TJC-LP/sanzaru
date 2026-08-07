"""Tests for the multi-voice podcast generation tool."""

import pytest

from sanzaru.tools.podcast import (
    PodcastResult,
    RenderUnit,
    _build_pause_list,
    _estimate_duration,
    _safe_title,
    _validate_script,
)

pytestmark = pytest.mark.audio


@pytest.fixture
def two_speaker_script():
    """Two-speaker PodcastScript for testing."""
    return {
        "title": "tech_talk",
        "speakers": [
            {
                "id": "host",
                "name": "Alex",
                "voice": "ash",
                "speed": 1.0,
                "instructions": "Confident host",
            },
            {
                "id": "cohost",
                "name": "Sam",
                "voice": "nova",
                "speed": 1.05,
                "instructions": "Curious co-host",
            },
        ],
        "segments": [
            {"speaker": "host", "text": "Welcome to Tech Talk."},
            {"speaker": "cohost", "text": "Great to be here.", "pause_after": 1000},
            {"speaker": "host", "text": "Today we discuss Haskell."},
        ],
        "config": {
            "default_pause_ms": 600,
            "intro_silence_ms": 500,
            "outro_silence_ms": 1000,
            "normalize_loudness": True,
            "output_format": "mp3",
            "output_bitrate": "192k",
        },
    }


@pytest.mark.unit
class TestValidateScript:
    """Unit tests for _validate_script."""

    def test_valid_minimal_script(self, minimal_script):
        """Valid minimal script passes validation."""
        title, speakers, segments, config = _validate_script(minimal_script)
        assert title == "test_podcast"
        assert len(speakers) == 1
        assert len(segments) == 1

    def test_valid_two_speaker_script(self, two_speaker_script):
        """Valid two-speaker script passes validation."""
        title, speakers, segments, config = _validate_script(two_speaker_script)
        assert title == "tech_talk"
        assert len(speakers) == 2
        assert len(segments) == 3

    def test_missing_required_top_level_key(self, minimal_script):
        """Missing top-level key raises ValueError."""
        del minimal_script["title"]
        with pytest.raises(ValueError, match="missing required field: 'title'"):
            _validate_script(minimal_script)

    def test_missing_speakers_key(self, minimal_script):
        """Missing speakers raises ValueError."""
        del minimal_script["speakers"]
        with pytest.raises(ValueError, match="missing required field: 'speakers'"):
            _validate_script(minimal_script)

    def test_missing_segments_key(self, minimal_script):
        """Missing segments raises ValueError."""
        del minimal_script["segments"]
        with pytest.raises(ValueError, match="missing required field: 'segments'"):
            _validate_script(minimal_script)

    def test_missing_config_key(self, minimal_script):
        """Missing config raises ValueError."""
        del minimal_script["config"]
        with pytest.raises(ValueError, match="missing required field: 'config'"):
            _validate_script(minimal_script)

    def test_empty_title_raises(self, minimal_script):
        """Empty title string raises ValueError."""
        minimal_script["title"] = "   "
        with pytest.raises(ValueError, match="'title' must not be empty"):
            _validate_script(minimal_script)

    def test_no_speakers_raises(self, minimal_script):
        """Empty speakers list raises ValueError."""
        minimal_script["speakers"] = []
        with pytest.raises(ValueError, match="at least 1 speaker"):
            _validate_script(minimal_script)

    def test_too_many_speakers_raises(self, minimal_script):
        """More than 4 speakers raises ValueError."""
        base_speaker = minimal_script["speakers"][0]
        extra = [{**base_speaker, "id": f"s{i}", "name": f"Speaker{i}"} for i in range(5)]
        minimal_script["speakers"] = extra
        with pytest.raises(ValueError, match="at most 4 speakers"):
            _validate_script(minimal_script)

    def test_speaker_missing_field_raises(self, minimal_script):
        """Speaker missing required field raises ValueError."""
        del minimal_script["speakers"][0]["voice"]
        with pytest.raises(ValueError, match="missing required field: 'voice'"):
            _validate_script(minimal_script)

    def test_empty_segments_raises(self, minimal_script):
        """Empty segments list raises ValueError."""
        minimal_script["segments"] = []
        with pytest.raises(ValueError, match="at least 1 segment"):
            _validate_script(minimal_script)

    def test_segment_unknown_speaker_raises(self, minimal_script):
        """Segment referencing unknown speaker id raises ValueError."""
        minimal_script["segments"][0]["speaker"] = "nonexistent"
        with pytest.raises(ValueError, match="unknown speaker id"):
            _validate_script(minimal_script)

    def test_segment_missing_text_raises(self, minimal_script):
        """Segment missing text field raises ValueError."""
        del minimal_script["segments"][0]["text"]
        with pytest.raises(ValueError, match="missing required field: 'text'"):
            _validate_script(minimal_script)

    def test_segment_text_too_long_raises(self, minimal_script):
        """Segment text exceeding 40000 chars raises ValueError."""
        minimal_script["segments"][0]["text"] = "x" * 40001
        with pytest.raises(ValueError, match="text exceeds 40000 characters"):
            _validate_script(minimal_script)

    def test_invalid_output_format_raises(self, minimal_script):
        """Invalid output_format raises ValueError."""
        minimal_script["config"]["output_format"] = "ogg"
        with pytest.raises(ValueError, match="output_format"):
            _validate_script(minimal_script)

    def test_config_missing_required_field_raises(self, minimal_script):
        """Config missing required field raises ValueError."""
        del minimal_script["config"]["normalize_loudness"]
        with pytest.raises(ValueError, match="PodcastConfig missing required field"):
            _validate_script(minimal_script)

    def test_wav_output_format_is_valid(self, minimal_script):
        """WAV output format passes validation."""
        minimal_script["config"]["output_format"] = "wav"
        title, speakers, segments, config = _validate_script(minimal_script)
        assert config["output_format"] == "wav"

    def test_four_speakers_is_valid(self, minimal_script):
        """Exactly 4 speakers passes validation."""
        voices = ["ash", "nova", "onyx", "alloy"]
        minimal_script["speakers"] = [
            {"id": f"s{i}", "name": f"Speaker{i}", "voice": voices[i], "speed": 1.0, "instructions": "test"}
            for i in range(4)
        ]
        minimal_script["segments"][0]["speaker"] = "s0"
        _validate_script(minimal_script)  # Should not raise

    def test_segment_empty_text_raises(self, minimal_script):
        """Segment with empty/whitespace-only text raises ValueError."""
        minimal_script["segments"][0]["text"] = "   "
        with pytest.raises(ValueError, match="text must not be empty"):
            _validate_script(minimal_script)

    def test_speaker_speed_out_of_range_raises(self, minimal_script):
        """Speaker speed outside 0.25–4.0 raises ValueError."""
        minimal_script["speakers"][0]["speed"] = 5.0
        with pytest.raises(ValueError, match="speed must be between 0.25 and 4.0"):
            _validate_script(minimal_script)

    def test_speaker_speed_too_low_raises(self, minimal_script):
        """Speaker speed below 0.25 raises ValueError."""
        minimal_script["speakers"][0]["speed"] = 0.1
        with pytest.raises(ValueError, match="speed must be between 0.25 and 4.0"):
            _validate_script(minimal_script)

    def test_segment_speed_override_out_of_range_raises(self, minimal_script):
        """Segment speed_override outside 0.25–4.0 raises ValueError."""
        minimal_script["segments"][0]["speed_override"] = 0.1
        with pytest.raises(ValueError, match="speed_override must be between 0.25 and 4.0"):
            _validate_script(minimal_script)

    def test_speed_at_boundaries_valid(self, minimal_script):
        """Speed values at exactly 0.25 and 4.0 pass validation."""
        minimal_script["speakers"][0]["speed"] = 0.25
        _validate_script(minimal_script)
        minimal_script["speakers"][0]["speed"] = 4.0
        _validate_script(minimal_script)


@pytest.mark.unit
class TestEstimateDuration:
    """Unit tests for _estimate_duration."""

    def test_basic_duration_estimate(self):
        """Single 150-word segment at 1.0x speed estimates to ~60s.

        Formula: word_count * 60 / (150 * speed) = 150 * 60 / 150 = 60s.
        """
        text = " ".join(["word"] * 150)
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {"default_pause_ms": 0}

        duration = _estimate_duration(segments, speakers, [0], config)
        assert duration == pytest.approx(60.0)

    def test_faster_speed_reduces_duration(self):
        """Higher speed multiplier reduces estimated duration proportionally.

        Formula: 150 * 60 / (150 * 2.0) = 30s.
        """
        text = " ".join(["word"] * 150)
        speakers = [{"id": "host", "speed": 2.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {"default_pause_ms": 0}

        duration = _estimate_duration(segments, speakers, [0], config)
        assert duration == pytest.approx(30.0)

    def test_pause_contributes_to_duration(self):
        """Only the pauses actually handed to the stitch step count."""
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": "short"}] * 2
        config = {"default_pause_ms": 1000}

        duration_with_pause = _estimate_duration(segments, speakers, [1000, 0], config)
        duration_no_pause = _estimate_duration(segments, speakers, [0, 0], config)

        assert duration_with_pause - duration_no_pause == pytest.approx(1.0)

    def test_pause_after_is_never_read_behind_the_pause_lists_back(self):
        """The stitch step zeroes the trailing pause, so the estimate must too.
        The pause list is the single source of truth: a `pause_after` the list
        does not carry is silence the output never contains."""
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": "word", "pause_after": 5000}]
        config = {"default_pause_ms": 5000}

        # One word at 1.0x is 60/150 = 0.4s, and nothing else.
        assert _estimate_duration(segments, speakers, [0], config) == pytest.approx(0.4)

    def test_intro_outro_silence_included(self):
        """Intro and outro silence contribute to total duration."""
        text = "short"
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {
            "default_pause_ms": 0,
            "intro_silence_ms": 500,
            "outro_silence_ms": 1000,
        }

        duration = _estimate_duration(segments, speakers, [0], config)
        config_no_silence = {"default_pause_ms": 0}
        duration_no_silence = _estimate_duration(segments, speakers, [0], config_no_silence)

        assert duration - duration_no_silence == pytest.approx(1.5)

    def test_speed_override_on_segment(self):
        """speed_override on a segment overrides speaker default speed.

        Formula: 150 * 60 / (150 * 2.0) = 30s.
        """
        text = " ".join(["word"] * 150)
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text, "speed_override": 2.0}]
        config = {"default_pause_ms": 0}

        duration = _estimate_duration(segments, speakers, [0], config)
        assert duration == pytest.approx(30.0)

    def test_speed_override_zero_not_ignored(self):
        """speed_override=0.25 (minimum) is not treated as falsy and ignored."""
        text = " ".join(["word"] * 150)
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text, "speed_override": 0.25}]
        config = {"default_pause_ms": 0}

        duration = _estimate_duration(segments, speakers, [0], config)
        # At 0.25x speed: 150 * 60 / (150 * 0.25) = 240s
        assert duration == pytest.approx(240.0)

    def test_empty_intro_outro_defaults_to_zero(self):
        """Missing intro/outro silence defaults to zero without error."""
        text = "hello"
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {"default_pause_ms": 0}  # No intro/outro keys

        duration = _estimate_duration(segments, speakers, [0], config)
        assert duration >= 0


@pytest.mark.unit
class TestBuildPauseList:
    """One pause per render unit, taken from that unit's LAST segment."""

    def test_one_pause_per_segment_and_no_trailing_pause(self):
        segments = [{"speaker": "a", "text": "x"} for _ in range(3)]
        units = [RenderUnit((0,), "a", False), RenderUnit((1,), "a", False), RenderUnit((2,), "a", False)]

        assert _build_pause_list(units, segments, 400) == [400, 400, 0]

    def test_dialogue_unit_takes_its_last_segments_pause(self):
        segments = [
            {"speaker": "a", "text": "x", "pause_after": 111},
            {"speaker": "b", "text": "x", "pause_after": 222},
            {"speaker": "a", "text": "x"},
        ]
        units = [RenderUnit((0, 1), "a", True), RenderUnit((2,), "a", False)]

        # 111 belongs to a turn inside the run; the model paces that gap.
        assert _build_pause_list(units, segments, 400) == [222, 0]

    def test_single_unit_gets_no_pause_at_all(self):
        segments = [{"speaker": "a", "text": "x", "pause_after": 5000}]

        assert _build_pause_list([RenderUnit((0,), "a", False)], segments, 400) == [0]


@pytest.mark.unit
class TestSafeTitle:
    """Unit tests for _safe_title."""

    def test_alphanumeric_unchanged(self):
        """Alphanumeric titles pass through unchanged."""
        assert _safe_title("mypodcast123") == "mypodcast123"

    def test_spaces_replaced_with_underscores(self):
        """Spaces are replaced with underscores."""
        assert _safe_title("my podcast title") == "my_podcast_title"

    def test_hyphens_and_underscores_preserved(self):
        """Hyphens and underscores are preserved."""
        assert _safe_title("my-podcast_ep1") == "my-podcast_ep1"

    def test_special_chars_replaced(self):
        """Special characters are replaced with underscores."""
        result = _safe_title("podcast: vol.1 (2024)")
        assert ":" not in result
        assert "(" not in result
        assert ")" not in result

    def test_empty_string_returns_fallback(self):
        """Empty-ish string returns 'podcast' fallback."""
        assert _safe_title("!!!") == "podcast"

    def test_unicode_preserved(self):
        """Unicode letters (accented chars) are treated as alphanumeric and preserved."""
        result = _safe_title("caf\u00e9 podcast")
        # \u00e9 is alphanumeric per str.isalnum(), so it is kept
        assert result == "caf\u00e9_podcast"


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_happy_path(mocker, tmp_audio_path):
    """generate_podcast produces a PodcastResult when TTS and storage succeed.

    The _stitch_audio and TTS calls are mocked so ffmpeg is not required.
    """
    from sanzaru.storage.local import LocalStorageBackend

    # Stub TTS response — raw bytes content doesn't matter (stitching is mocked)
    fake_audio_bytes = b"FAKE_MP3_SEGMENT"
    mock_response = mocker.MagicMock()
    mock_response.content = fake_audio_bytes

    mock_client = mocker.MagicMock()
    mock_client.audio.speech.create = mocker.AsyncMock(return_value=mock_response)
    # TTS now routes through the provider layer, which resolves the client there.
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=mock_client)

    # Stub _stitch_audio so pydub/ffmpeg is never invoked
    fake_stitched = b"FAKE_STITCHED_OUTPUT"
    mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=fake_stitched)

    # Use local storage backend pointing at tmp dir
    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=storage)

    script = {
        "title": "test_ep1",
        "speakers": [
            {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "Confident"},
            {"id": "cohost", "name": "Sam", "voice": "nova", "speed": 1.0, "instructions": "Curious"},
        ],
        "segments": [
            {"speaker": "host", "text": "Welcome to the show."},
            {"speaker": "cohost", "text": "Great to be here."},
            {"speaker": "host", "text": "Let us get started."},
        ],
        "config": {
            "default_pause_ms": 300,
            "intro_silence_ms": 200,
            "outro_silence_ms": 200,
            "normalize_loudness": True,
            "output_format": "mp3",
            "output_bitrate": "128k",
        },
    }

    from sanzaru.tools.podcast import generate_podcast

    result = await generate_podcast(script)

    assert isinstance(result, PodcastResult)
    assert result.title == "test_ep1"
    assert result.segment_count == 3
    assert result.speakers == ["Alex", "Sam"]
    assert result.output_file.startswith("test_ep1_")
    assert result.output_file.endswith(".mp3")
    assert result.estimated_duration_seconds >= 0

    # TTS was called once per segment
    assert mock_client.audio.speech.create.call_count == 3

    # Output file was written to storage
    output_path = tmp_audio_path / result.output_file
    assert output_path.exists()
    assert output_path.read_bytes() == fake_stitched

    # Transcript includes all speakers with correct format
    assert result.transcript == (
        "**Alex:** Welcome to the show.\n\n**Sam:** Great to be here.\n\n**Alex:** Let us get started."
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_honors_an_explicit_filename(mocker, tmp_audio_path):
    """`filename` is written verbatim and reported back — #54.

    Before this, the CLI let the tool auto-name the episode and renamed it
    afterwards, so `output_file` named a file that no longer existed.
    """
    from sanzaru.storage.local import LocalStorageBackend

    mock_response = mocker.MagicMock()
    mock_response.content = b"FAKE"
    mock_client = mocker.MagicMock()
    mock_client.audio.speech.create = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=mock_client)
    mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=b"STITCHED")
    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=storage)

    script = {
        "title": "Some Long Title",
        "speakers": [{"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "host", "text": "Hello."}],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    from sanzaru.tools.podcast import generate_podcast

    result = await generate_podcast(script, filename="eleven-demo.mp3")

    assert result.output_file == "eleven-demo.mp3"
    assert (tmp_audio_path / "eleven-demo.mp3").read_bytes() == b"STITCHED"
    # The autogenerated name must not also have been written.
    assert not list(tmp_audio_path.glob("Some_Long_Title_*.mp3"))


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_warns_when_filename_suffix_contradicts_format(mocker, tmp_audio_path, caplog):
    """A .wav name on an mp3 render is the caller's call, but it gets said out loud."""
    from sanzaru.storage.local import LocalStorageBackend

    mock_response = mocker.MagicMock()
    mock_response.content = b"FAKE"
    mock_client = mocker.MagicMock()
    mock_client.audio.speech.create = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=mock_client)
    mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=b"STITCHED")
    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=storage)

    script = {
        "title": "ep",
        "speakers": [{"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "host", "text": "Hello."}],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    from sanzaru.tools.podcast import generate_podcast

    with caplog.at_level("WARNING"):
        result = await generate_podcast(script, filename="episode.wav")

    assert result.output_file == "episode.wav"
    assert "output_format" in caplog.text


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_single_speaker_transcript(mocker, tmp_audio_path):
    """generate_podcast transcript for a single segment has correct format."""
    from sanzaru.storage.local import LocalStorageBackend

    mock_response = mocker.MagicMock()
    mock_response.content = b"FAKE"
    mock_client = mocker.MagicMock()
    mock_client.audio.speech.create = mocker.AsyncMock(return_value=mock_response)
    # TTS now routes through the provider layer, which resolves the client there.
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=mock_client)
    mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=b"STITCHED")

    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=storage)

    script = {
        "title": "solo",
        "speakers": [{"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "host", "text": "Hello there."}],
        "config": {
            "default_pause_ms": 600,
            "normalize_loudness": True,
            "output_format": "mp3",
        },
    }

    from sanzaru.tools.podcast import generate_podcast

    result = await generate_podcast(script)
    assert result.transcript == "**Alex:** Hello there."


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_validation_error():
    """generate_podcast raises ValueError on invalid script."""
    from sanzaru.tools.podcast import generate_podcast

    with pytest.raises(ValueError, match="missing required field"):
        await generate_podcast({"title": "oops"})  # Missing speakers, segments, config


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_unknown_speaker_raises():
    """generate_podcast raises ValueError when segment references unknown speaker."""
    from sanzaru.tools.podcast import generate_podcast

    script = {
        "title": "bad",
        "speakers": [{"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "nobody", "text": "This will fail."}],
        "config": {
            "default_pause_ms": 600,
            "normalize_loudness": True,
            "output_format": "mp3",
        },
    }

    with pytest.raises(ValueError, match="unknown speaker id"):
        await generate_podcast(script)


# ==================== PROVIDER SUPPORT ====================


@pytest.mark.unit
class TestValidateScriptProviders:
    """Provider-aware validation. Every existing OpenAI script must still pass."""

    def test_openai_is_the_default(self, minimal_script):
        # No provider anywhere — the OpenAI speed range applies.
        minimal_script["speakers"][0]["speed"] = 3.0
        _validate_script(minimal_script)

    def test_unknown_speaker_provider_raises(self, minimal_script):
        minimal_script["speakers"][0]["provider"] = "azure"
        with pytest.raises(ValueError, match="Speaker 0 'provider': unknown provider"):
            _validate_script(minimal_script)

    def test_unknown_config_provider_raises(self, minimal_script):
        minimal_script["config"]["provider"] = "azure"
        with pytest.raises(ValueError, match="PodcastConfig 'provider': unknown provider"):
            _validate_script(minimal_script)

    def test_elevenlabs_speaker_needs_a_voice_id(self, minimal_script):
        minimal_script["speakers"][0]["provider"] = "elevenlabs"
        minimal_script["speakers"][0]["voice"] = ""
        with pytest.raises(ValueError, match="requires an explicit voice id"):
            _validate_script(minimal_script)

    def test_elevenlabs_speed_range_is_narrower(self, minimal_script):
        minimal_script["speakers"][0]["provider"] = "elevenlabs"
        minimal_script["speakers"][0]["voice"] = "voice_abc"
        minimal_script["speakers"][0]["speed"] = 3.0
        with pytest.raises(ValueError, match="between 0.7 and 1.2 for provider='elevenlabs'"):
            _validate_script(minimal_script)

    def test_v3_rejects_speed(self, minimal_script):
        minimal_script["speakers"][0].update({"provider": "elevenlabs", "voice": "v1", "speed": 1.1})
        with pytest.raises(ValueError, match="does not support speed"):
            _validate_script(minimal_script)

    def test_v3_accepts_neutral_speed(self, minimal_script):
        minimal_script["speakers"][0].update({"provider": "elevenlabs", "voice": "v1", "speed": 1.0})
        _validate_script(minimal_script)

    def test_config_provider_applies_to_all_speakers(self, minimal_script):
        minimal_script["config"]["provider"] = "elevenlabs"
        minimal_script["speakers"][0]["voice"] = "v1"
        minimal_script["speakers"][0]["speed"] = 2.0
        with pytest.raises(ValueError, match="provider='elevenlabs'"):
            _validate_script(minimal_script)

    def test_speaker_provider_overrides_config(self, minimal_script):
        # config says elevenlabs, the speaker says openai — 2.0 is legal there.
        minimal_script["config"]["provider"] = "elevenlabs"
        minimal_script["speakers"][0]["provider"] = "openai"
        minimal_script["speakers"][0]["speed"] = 2.0
        _validate_script(minimal_script)

    def test_default_provider_argument_applies(self, minimal_script):
        minimal_script["speakers"][0]["voice"] = "v1"
        minimal_script["speakers"][0]["speed"] = 2.0
        with pytest.raises(ValueError, match="provider='elevenlabs'"):
            _validate_script(minimal_script, default_provider="elevenlabs")

    def test_speed_override_uses_the_speakers_provider_range(self, minimal_script):
        minimal_script["speakers"][0].update({"provider": "elevenlabs", "voice": "v1", "speed": 1.0})
        minimal_script["segments"][0]["speed_override"] = 3.0
        with pytest.raises(ValueError, match="speed_override must be between 0.7 and 1.2"):
            _validate_script(minimal_script)

    def test_speed_override_is_probed_against_the_speakers_model(self, minimal_script):
        """0.9 is inside ElevenLabs' 0.7-1.2 range, so only the per-model rule can
        reject it — and the speaker pass only ever probed speaker['speed'].

        Without the per-segment probe this raises inside the task group, after
        sibling segments have already spent API calls.
        """
        minimal_script["speakers"][0].update({"provider": "elevenlabs", "voice": "v1", "model": "eleven_v3"})
        minimal_script["segments"][0]["speed_override"] = 0.9
        with pytest.raises(ValueError, match="Segment 0 speed_override: eleven_v3 does not support speed"):
            _validate_script(minimal_script)

    def test_supported_speed_override_passes_the_probe(self, minimal_script):
        minimal_script["speakers"][0].update(
            {"provider": "elevenlabs", "voice": "v1", "model": "eleven_multilingual_v2"}
        )
        minimal_script["segments"][0]["speed_override"] = 0.9
        _validate_script(minimal_script)

    def test_cross_provider_model_rejected(self, minimal_script):
        minimal_script["speakers"][0].update({"provider": "elevenlabs", "voice": "v1", "model": "tts-1"})
        with pytest.raises(ValueError, match="not an ElevenLabs model"):
            _validate_script(minimal_script)

    def test_voice_settings_unknown_key(self, minimal_script):
        minimal_script["speakers"][0].update({"provider": "elevenlabs", "voice": "v1", "voice_settings": {"warmth": 1}})
        with pytest.raises(ValueError, match="unknown key 'warmth'"):
            _validate_script(minimal_script)

    def test_voice_settings_wrong_type(self, minimal_script):
        minimal_script["speakers"][0].update(
            {"provider": "elevenlabs", "voice": "v1", "voice_settings": {"stability": "high"}}
        )
        with pytest.raises(ValueError, match="voice_settings\\['stability'\\] must be a number"):
            _validate_script(minimal_script)

    @pytest.mark.parametrize("value", [True, False])
    def test_voice_settings_bool_is_not_a_number(self, minimal_script, value):
        """isinstance(True, int) is True, so bool has to be excluded explicitly —
        otherwise `{"stability": true}` clears both the type and range checks."""
        minimal_script["speakers"][0].update(
            {"provider": "elevenlabs", "voice": "v1", "voice_settings": {"stability": value}}
        )
        with pytest.raises(ValueError, match="voice_settings\\['stability'\\] must be a number"):
            _validate_script(minimal_script)

    def test_voice_settings_out_of_range(self, minimal_script):
        minimal_script["speakers"][0].update(
            {"provider": "elevenlabs", "voice": "v1", "voice_settings": {"style": 1.5}}
        )
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            _validate_script(minimal_script)

    def test_valid_voice_settings_pass(self, minimal_script):
        minimal_script["speakers"][0].update(
            {
                "provider": "elevenlabs",
                "voice": "v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.9, "use_speaker_boost": True},
            }
        )
        _validate_script(minimal_script)

    @pytest.mark.parametrize("value", [0, -1, "three"])
    def test_invalid_max_concurrency(self, minimal_script, value):
        minimal_script["config"]["max_concurrency"] = value
        with pytest.raises(ValueError, match="max_concurrency' must be a positive integer"):
            _validate_script(minimal_script)

    def test_valid_max_concurrency(self, minimal_script):
        minimal_script["config"]["max_concurrency"] = 2
        _validate_script(minimal_script)


@pytest.fixture
def podcast_env(mocker, tmp_audio_path):
    """Mock storage + stubbed stitching so ffmpeg/pydub is never invoked."""
    from sanzaru.storage.local import LocalStorageBackend

    mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=b"STITCHED")
    storage = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=storage)
    return storage


def _openai_client(mocker, content=b"OPENAI_MP3"):
    response = mocker.MagicMock()
    response.content = content
    client = mocker.MagicMock()
    client.audio.speech.create = mocker.AsyncMock(return_value=response)
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=client)
    return client


@pytest.mark.integration
@pytest.mark.anyio
async def test_generate_podcast_mixed_providers(mocker, podcast_env, fake_elevenlabs):
    """One episode, one speaker per provider — each must hit its own client."""
    from sanzaru.tools.podcast import generate_podcast

    openai_client = _openai_client(mocker)
    el_client = fake_elevenlabs.Client(chunks=(b"EL_MP3",))
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=el_client)

    script = {
        "title": "mixed_ep",
        "speakers": [
            {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "Confident"},
            {
                "id": "guest",
                "name": "Sam",
                "voice": "voice_xyz",
                "speed": 1.0,
                "instructions": "ignored by elevenlabs",
                "provider": "elevenlabs",
                "voice_settings": {"stability": 0.4},
            },
        ],
        "segments": [
            {"speaker": "host", "text": "Welcome back."},
            {"speaker": "guest", "text": "Glad to be here."},
            {"speaker": "host", "text": "Let us dig in."},
        ],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    result = await generate_podcast(script)

    # Two OpenAI segments, one ElevenLabs segment.
    assert openai_client.audio.speech.create.await_count == 2
    assert len(el_client.text_to_speech.calls) == 1

    el_call = el_client.text_to_speech.calls[0]
    assert el_call["voice_id"] == "voice_xyz"
    assert el_call["model_id"] == "eleven_v3"
    assert el_call["voice_settings"].stability == 0.4

    # Envelope shape is unchanged, so downstream QC tooling doesn't fork.
    assert isinstance(result, PodcastResult)
    assert result.segment_count == 3
    assert result.speakers == ["Alex", "Sam"]
    assert "**Sam:** Glad to be here." in result.transcript


@pytest.mark.integration
@pytest.mark.anyio
async def test_openai_segments_carry_voice_speed_and_instructions(mocker, podcast_env):
    """The per-segment kwargs actually reaching the API were previously untested."""
    from sanzaru.tools.podcast import generate_podcast

    client = _openai_client(mocker)
    script = {
        "title": "kwargs_ep",
        "speakers": [{"id": "host", "name": "Alex", "voice": "onyx", "speed": 1.25, "instructions": "Be warm"}],
        "segments": [
            {"speaker": "host", "text": "First."},
            {"speaker": "host", "text": "Second.", "speed_override": 0.8, "instruction_override": "Be terse"},
        ],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    await generate_podcast(script)

    calls = {c.kwargs["input"]: c.kwargs for c in client.audio.speech.create.await_args_list}
    assert calls["First."]["voice"] == "onyx"
    assert calls["First."]["speed"] == 1.25
    assert calls["First."]["instructions"] == "Be warm"
    assert calls["Second."]["speed"] == 0.8
    assert calls["Second."]["instructions"] == "Be terse"


@pytest.mark.integration
@pytest.mark.anyio
async def test_empty_instruction_override_is_honored(mocker, podcast_env):
    """An intentional empty override must not fall back to the speaker's instructions."""
    from sanzaru.tools.podcast import generate_podcast

    client = _openai_client(mocker)
    script = {
        "title": "override_ep",
        "speakers": [{"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "Be dramatic"}],
        "segments": [{"speaker": "host", "text": "Plainly.", "instruction_override": ""}],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    await generate_podcast(script)

    assert client.audio.speech.create.await_args.kwargs["instructions"] == ""


@pytest.mark.integration
@pytest.mark.anyio
async def test_speed_override_beats_the_speakers_voice_settings_speed(mocker, podcast_env, fake_elevenlabs):
    """The segment override is the most specific value in the script, so it must
    reach the API — a speaker-level voice_settings['speed'] would otherwise
    swallow it, since that is the knob ElevenLabs actually reads."""
    from sanzaru.tools.podcast import generate_podcast

    el_client = fake_elevenlabs.Client(chunks=(b"EL_MP3",))
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=el_client)

    script = {
        "title": "override_speed_ep",
        "speakers": [
            {
                "id": "guest",
                "name": "Sam",
                "voice": "voice_xyz",
                "speed": 1.0,
                "instructions": "",
                "provider": "elevenlabs",
                "model": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.4, "speed": 1.1},
            }
        ],
        "segments": [
            {"speaker": "guest", "text": "Normal pace."},
            {"speaker": "guest", "text": "Slower now.", "speed_override": 0.8},
        ],
        "config": {"default_pause_ms": 300, "normalize_loudness": True, "output_format": "mp3"},
    }

    await generate_podcast(script)

    calls = {call["text"]: call["voice_settings"] for call in el_client.text_to_speech.calls}
    assert calls["Normal pace."].speed == 1.1
    assert calls["Slower now."].speed == 0.8
    # Only speed is replaced; the rest of the speaker's tuning still applies.
    assert calls["Slower now."].stability == 0.4
    # And the speaker's own dict is untouched, so segment order cannot matter.
    assert script["speakers"][0]["voice_settings"]["speed"] == 1.1


@pytest.mark.integration
@pytest.mark.anyio
async def test_elevenlabs_fan_out_respects_max_concurrency(mocker, podcast_env, fake_elevenlabs):
    import anyio

    from sanzaru.tools.podcast import generate_podcast

    in_flight = 0
    peak = 0

    async def track():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await anyio.sleep(0)
        in_flight -= 1

    el_client = fake_elevenlabs.Client(chunks=(b"EL",), on_call=track)
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=el_client)

    script = {
        "title": "concurrency_ep",
        "speakers": [
            {
                "id": "host",
                "name": "Alex",
                "voice": "v1",
                "speed": 1.0,
                "instructions": "",
                "provider": "elevenlabs",
            }
        ],
        "segments": [{"speaker": "host", "text": f"Segment {i}."} for i in range(8)],
        "config": {
            "default_pause_ms": 200,
            "normalize_loudness": True,
            "output_format": "mp3",
            "max_concurrency": 2,
        },
    }

    result = await generate_podcast(script)

    assert peak <= 2
    assert len(el_client.text_to_speech.calls) == 8
    # Order is by index, not completion — the limiter only delays task entry.
    assert [c["text"] for c in el_client.text_to_speech.calls] != [] and result.segment_count == 8


@pytest.mark.integration
@pytest.mark.anyio
async def test_openai_fan_out_stays_unbounded(mocker, podcast_env):
    """Zero-behavior-change guard: OpenAI-only episodes must not be throttled."""
    import anyio

    from sanzaru.tools.podcast import generate_podcast

    in_flight = 0
    peak = 0

    async def create(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await anyio.sleep(0)
        in_flight -= 1
        response = mocker.MagicMock()
        response.content = b"MP3"
        return response

    client = mocker.MagicMock()
    client.audio.speech.create = mocker.AsyncMock(side_effect=create)
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=client)

    script = {
        "title": "unbounded_ep",
        "speakers": [{"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "x"}],
        "segments": [{"speaker": "host", "text": f"Segment {i}."} for i in range(8)],
        "config": {"default_pause_ms": 200, "normalize_loudness": True, "output_format": "mp3"},
    }

    await generate_podcast(script)

    assert peak == 8


@pytest.mark.integration
@pytest.mark.anyio
async def test_segment_order_is_preserved_under_a_limiter(mocker, podcast_env, fake_elevenlabs):
    """Stitching receives segments in script order even when renders finish out of order."""
    from sanzaru.tools.podcast import generate_podcast

    class OrderedTTS:
        def __init__(self):
            self.calls = []

        def convert(self, **kwargs):
            self.calls.append(kwargs)
            text = kwargs["text"]

            async def _stream():
                # Later segments finish first.
                import anyio

                await anyio.sleep(0.001 * (5 - len(text)))
                yield text.encode()

            return _stream()

    client = fake_elevenlabs.Client()
    client.text_to_speech = OrderedTTS()
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)
    stitch = mocker.patch("sanzaru.tools.podcast._stitch_audio", return_value=b"STITCHED")

    script = {
        "title": "order_ep",
        "speakers": [
            {"id": "h", "name": "A", "voice": "v1", "speed": 1.0, "instructions": "", "provider": "elevenlabs"}
        ],
        "segments": [{"speaker": "h", "text": t} for t in ("aaaa", "bbb", "cc", "d")],
        "config": {
            "default_pause_ms": 100,
            "normalize_loudness": True,
            "output_format": "mp3",
            "max_concurrency": 4,
        },
    }

    await generate_podcast(script)

    assert stitch.call_args.kwargs["segment_bytes_list"] == [b"aaaa", b"bbb", b"cc", b"d"]


@pytest.mark.integration
def test_stitch_normalizes_mixed_sample_rates(tmp_path):
    """OpenAI mp3 is 24kHz, ElevenLabs mp3_44100_128 is 44.1kHz.

    pydub resamples on concatenation anyway, but _stitch_audio pins the rate so
    a mixed-provider episode is deterministic regardless of segment order.
    Needs a real encoder, so it is skipped where ffmpeg is unavailable.
    """
    import io
    import shutil

    if not (shutil.which("ffmpeg") or shutil.which("avconv")):
        pytest.skip("ffmpeg not available")

    from pydub import AudioSegment
    from pydub.generators import Sine

    from sanzaru.tools.podcast import _stitch_audio

    def mp3_bytes(freq: int, rate: int, ms: int = 500) -> bytes:
        seg = Sine(freq, sample_rate=rate).to_audio_segment(duration=ms).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="mp3")
        return buf.getvalue()

    out = _stitch_audio(
        segment_bytes_list=[mp3_bytes(440, 24000), mp3_bytes(660, 44100), mp3_bytes(440, 24000)],
        pause_ms_list=[300, 300, 0],
        intro_ms=200,
        outro_ms=200,
        normalize_loudness=True,
        output_format="mp3",
        output_bitrate="192k",
    )

    final = AudioSegment.from_mp3(io.BytesIO(out))
    assert final.frame_rate == 44100
    assert final.channels == 1
    # 200 intro + 3x500 speech + 300 + 300 pauses + 200 outro = 2500ms.
    # A resampling bug would stretch or squash the 24kHz segments.
    assert 2400 <= len(final) <= 2600


# ==================== THE `model` ARGUMENT ====================
# It used to be dropped for anything but OpenAI speakers, so
# `--provider elevenlabs --model eleven_flash_v2_5` silently rendered on
# eleven_v3 — wrong chunk budget, wrong concurrency cap, wrong price.


def _elevenlabs_script(**speaker_extra):
    speaker = {"id": "host", "name": "Alex", "voice": "v1", "speed": 1.0, "instructions": "", "provider": "elevenlabs"}
    speaker.update(speaker_extra)
    return {
        "title": "model_ep",
        "speakers": [speaker],
        "segments": [{"speaker": "host", "text": "Hello there."}],
        "config": {"default_pause_ms": 200, "normalize_loudness": True, "output_format": "mp3"},
    }


@pytest.mark.integration
@pytest.mark.anyio
async def test_elevenlabs_model_argument_reaches_the_api(mocker, podcast_env, fake_elevenlabs):
    client = fake_elevenlabs.Client(chunks=(b"EL",))
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)
    from sanzaru.tools.podcast import generate_podcast

    await generate_podcast(_elevenlabs_script(), model="eleven_flash_v2_5", provider="elevenlabs")

    assert client.text_to_speech.calls[0]["model_id"] == "eleven_flash_v2_5"


@pytest.mark.integration
@pytest.mark.anyio
async def test_speaker_model_still_beats_the_argument(mocker, podcast_env, fake_elevenlabs):
    client = fake_elevenlabs.Client(chunks=(b"EL",))
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)
    from sanzaru.tools.podcast import generate_podcast

    await generate_podcast(
        _elevenlabs_script(model="eleven_multilingual_v2"),
        model="eleven_flash_v2_5",
        provider="elevenlabs",
    )

    assert client.text_to_speech.calls[0]["model_id"] == "eleven_multilingual_v2"


@pytest.mark.integration
@pytest.mark.anyio
async def test_foreign_model_falls_back_per_provider(mocker, podcast_env, fake_elevenlabs):
    """A mixed episode gets one `model`; the provider it does not name must not explode."""
    openai_client = _openai_client(mocker)
    el_client = fake_elevenlabs.Client(chunks=(b"EL",))
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=el_client)
    from sanzaru.tools.podcast import generate_podcast

    script = {
        "title": "mixed_model_ep",
        "speakers": [
            {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "Warm", "provider": "openai"},
            {"id": "guest", "name": "Sam", "voice": "v1", "speed": 1.0, "instructions": "", "provider": "elevenlabs"},
        ],
        "segments": [{"speaker": "host", "text": "Hi."}, {"speaker": "guest", "text": "Hello."}],
        "config": {"default_pause_ms": 200, "normalize_loudness": True, "output_format": "mp3"},
    }

    await generate_podcast(script, model="eleven_flash_v2_5", provider="elevenlabs")

    assert el_client.text_to_speech.calls[0]["model_id"] == "eleven_flash_v2_5"
    # OpenAI's resolve_model accepts unknown names, so a foreign one would
    # otherwise sail through to the API as-is.
    assert openai_client.audio.speech.create.await_args.kwargs["model"] == "gpt-4o-mini-tts"


@pytest.mark.audio
@pytest.mark.unit
def test_validation_checks_the_model_the_render_will_use():
    """Validation ran against a hardcoded "gpt-4o-mini-tts", so an ElevenLabs
    episode was rejected (or accepted) on the wrong model's rules.

    speed=1.1 is legal on eleven_flash_v2_5 and rejected by eleven_v3, so this
    also pins the error message to the model the caller actually chose.
    """
    from sanzaru.tools.podcast import _validate_script

    script = _elevenlabs_script()
    script["speakers"][0]["speed"] = 1.1

    _validate_script(script, default_provider="elevenlabs", default_model="eleven_flash_v2_5")

    with pytest.raises(ValueError, match="eleven_v3 does not support speed adjustment"):
        _validate_script(script, default_provider="elevenlabs", default_model="eleven_v3")


# ==================== DERIVED CONCURRENCY LIMITS ====================
# Every other concurrency test pins config.max_concurrency, which bypasses the
# derivation entirely. These exercise the derived path.


def _mixed_model_script(first: str, second: str):
    """One ElevenLabs episode using two models with different caps (flash 4, v3 2)."""
    return {
        "title": f"caps_{first}_{second}",
        "speakers": [
            {
                "id": name,
                "name": name,
                "voice": f"v_{name}",
                "speed": 1.0,
                "instructions": "",
                "provider": "elevenlabs",
                "model": model,
            }
            for name, model in (("first", first), ("second", second))
        ],
        # Enough segments per speaker that a cap of 4 is actually reachable.
        "segments": [{"speaker": s, "text": f"Segment {i}."} for i in range(4) for s in ("first", "second")],
        "config": {"default_pause_ms": 200, "normalize_loudness": True, "output_format": "mp3"},
    }


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("first", "second"),
    [("eleven_flash_v2_5", "eleven_v3"), ("eleven_v3", "eleven_flash_v2_5")],
)
async def test_provider_limiter_takes_the_smallest_model_cap(mocker, podcast_env, fake_elevenlabs, first, second):
    """One limiter serves both models, so the tighter cap has to win.

    Listing the flash speaker first used to yield a limiter of 4 and run the
    eleven_v3 segments four-wide — the case that returns HTTP 429 on Free tier.
    """
    import anyio

    from sanzaru.tools.podcast import generate_podcast

    in_flight = 0
    peak = 0

    async def track():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await anyio.sleep(0)
        in_flight -= 1

    client = fake_elevenlabs.Client(chunks=(b"EL",), on_call=track)
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

    await generate_podcast(_mixed_model_script(first, second), provider="elevenlabs")

    assert peak == 2
    assert len(client.text_to_speech.calls) == 8


@pytest.mark.integration
@pytest.mark.anyio
async def test_config_max_concurrency_still_overrides_the_derived_cap(mocker, podcast_env, fake_elevenlabs):
    """A paid tier raises the cap past what the model table knows about."""
    import anyio

    from sanzaru.tools.podcast import generate_podcast

    in_flight = 0
    peak = 0

    async def track():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await anyio.sleep(0)
        in_flight -= 1

    client = fake_elevenlabs.Client(chunks=(b"EL",), on_call=track)
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

    script = _mixed_model_script("eleven_flash_v2_5", "eleven_v3")
    script["config"]["max_concurrency"] = 8

    await generate_podcast(script, provider="elevenlabs")

    assert peak == 8


@pytest.mark.audio
@pytest.mark.unit
def test_unbounded_never_wins_a_min_against_a_real_cap():
    """0 means unbounded, so min() would silently pin a provider to zero."""
    from sanzaru.tools.podcast import _derive_concurrency_limits

    class FakeProvider:
        def __init__(self, name, caps):
            self.name = name
            self._caps = caps

        def max_concurrency(self, model):
            return self._caps[model]

    capped = FakeProvider("elevenlabs", {"a": 0, "b": 2})
    unbounded = FakeProvider("openai", {"c": 0, "d": 0})

    limits = _derive_concurrency_limits(
        {"s1": capped, "s2": capped, "s3": unbounded, "s4": unbounded},
        {"s1": "a", "s2": "b", "s3": "c", "s4": "d"},
    )

    assert limits == {"elevenlabs": 2, "openai": 0}
