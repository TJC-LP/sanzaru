"""Tests for the multi-voice podcast generation tool."""

import pathlib

import pytest

from sanzaru.tools.podcast import (
    PodcastResult,
    _build_transcript,
    _estimate_duration,
    _safe_title,
    _validate_script,
)

pytestmark = pytest.mark.audio


@pytest.fixture
def tmp_audio_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary directory for audio files."""
    audio_path = tmp_path / "audio"
    audio_path.mkdir()
    return audio_path


# ==================== FIXTURES ====================


@pytest.fixture
def minimal_script():
    """Minimal valid PodcastScript for testing."""
    return {
        "title": "test_podcast",
        "speakers": [
            {
                "id": "host",
                "name": "Alex",
                "voice": "ash",
                "speed": 1.0,
                "instructions": "Confident host",
            }
        ],
        "segments": [
            {"speaker": "host", "text": "Welcome to the show."},
        ],
        "config": {
            "default_pause_ms": 600,
            "section_pause_ms": 1200,
            "normalize_loudness": True,
            "output_format": "mp3",
        },
    }


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
            "section_pause_ms": 1200,
            "intro_silence_ms": 500,
            "outro_silence_ms": 1000,
            "normalize_loudness": True,
            "output_format": "mp3",
            "output_bitrate": "192k",
        },
    }


# ==================== _validate_script UNIT TESTS ====================


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


# ==================== _estimate_duration UNIT TESTS ====================


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

        duration = _estimate_duration(segments, speakers, config)
        assert duration == pytest.approx(60.0)

    def test_faster_speed_reduces_duration(self):
        """Higher speed multiplier reduces estimated duration proportionally.

        Formula: 150 * 60 / (150 * 2.0) = 30s.
        """
        text = " ".join(["word"] * 150)
        speakers = [{"id": "host", "speed": 2.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {"default_pause_ms": 0}

        duration = _estimate_duration(segments, speakers, config)
        assert duration == pytest.approx(30.0)

    def test_pause_contributes_to_duration(self):
        """Pauses are included in estimated duration."""
        text = "short"
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {"default_pause_ms": 1000}

        duration_with_pause = _estimate_duration(segments, speakers, config)
        config["default_pause_ms"] = 0
        duration_no_pause = _estimate_duration(segments, speakers, config)

        assert duration_with_pause - duration_no_pause == pytest.approx(1.0)

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

        duration = _estimate_duration(segments, speakers, config)
        config_no_silence = {"default_pause_ms": 0}
        duration_no_silence = _estimate_duration(segments, speakers, config_no_silence)

        assert duration - duration_no_silence == pytest.approx(1.5)

    def test_speed_override_on_segment(self):
        """speed_override on a segment overrides speaker default speed.

        Formula: 150 * 60 / (150 * 2.0) = 30s.
        """
        text = " ".join(["word"] * 150)
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text, "speed_override": 2.0}]
        config = {"default_pause_ms": 0}

        duration = _estimate_duration(segments, speakers, config)
        assert duration == pytest.approx(30.0)

    def test_empty_intro_outro_defaults_to_zero(self):
        """Missing intro/outro silence defaults to zero without error."""
        text = "hello"
        speakers = [{"id": "host", "speed": 1.0}]
        segments = [{"speaker": "host", "text": text}]
        config = {"default_pause_ms": 0}  # No intro/outro keys

        duration = _estimate_duration(segments, speakers, config)
        assert duration >= 0


# ==================== _build_transcript UNIT TESTS ====================


class TestBuildTranscript:
    """Unit tests for _build_transcript."""

    def test_single_segment_transcript(self):
        """Single segment produces correct transcript format."""
        segments = [{"speaker": "host", "text": "Hello there."}]
        speaker_map = {"host": {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": ""}}

        result = _build_transcript(segments, speaker_map)
        assert result == "**Alex:** Hello there."

    def test_two_speaker_transcript(self):
        """Two-speaker transcript has blank line between turns."""
        segments = [
            {"speaker": "host", "text": "Welcome."},
            {"speaker": "cohost", "text": "Thanks for having me."},
        ]
        speaker_map = {
            "host": {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": ""},
            "cohost": {"id": "cohost", "name": "Sam", "voice": "nova", "speed": 1.05, "instructions": ""},
        }

        result = _build_transcript(segments, speaker_map)
        lines = result.split("\n\n")
        assert len(lines) == 2
        assert lines[0] == "**Alex:** Welcome."
        assert lines[1] == "**Sam:** Thanks for having me."

    def test_alternating_speakers(self):
        """Multiple alternating speakers are ordered correctly."""
        segments = [
            {"speaker": "a", "text": "First."},
            {"speaker": "b", "text": "Second."},
            {"speaker": "a", "text": "Third."},
        ]
        speaker_map = {
            "a": {"id": "a", "name": "Anna", "voice": "ash", "speed": 1.0, "instructions": ""},
            "b": {"id": "b", "name": "Bob", "voice": "nova", "speed": 1.0, "instructions": ""},
        }

        result = _build_transcript(segments, speaker_map)
        lines = result.split("\n\n")
        assert lines[0] == "**Anna:** First."
        assert lines[1] == "**Bob:** Second."
        assert lines[2] == "**Anna:** Third."


# ==================== _safe_title UNIT TESTS ====================


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
        result = _safe_title("café podcast")
        # é is alphanumeric per str.isalnum(), so it is kept
        assert result == "café_podcast"


# ==================== INTEGRATION TESTS ====================


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
    mocker.patch("sanzaru.tools.podcast.get_client", return_value=mock_client)

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
            "section_pause_ms": 600,
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
    assert "Alex" in result.transcript
    assert "Sam" in result.transcript
    assert result.output_file.startswith("test_ep1_")
    assert result.output_file.endswith(".mp3")
    assert result.estimated_duration_seconds >= 0

    # TTS was called once per segment
    assert mock_client.audio.speech.create.call_count == 3

    # Output file was written to storage
    output_path = tmp_audio_path / result.output_file
    assert output_path.exists()
    assert output_path.read_bytes() == fake_stitched


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
            "section_pause_ms": 1200,
            "normalize_loudness": True,
            "output_format": "mp3",
        },
    }

    with pytest.raises(ValueError, match="unknown speaker id"):
        await generate_podcast(script)
