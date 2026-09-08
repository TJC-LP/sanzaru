# SPDX-License-Identifier: MIT
"""Unit tests for utility functions."""

from unittest.mock import patch

import pytest

from sanzaru.utils import generate_filename, reject_reserved_name, suffix_for_variant, validate_resource_id


@pytest.mark.unit
class TestSuffixForVariant:
    """Test file extension mapping for video variants."""

    def test_video_variant(self):
        assert suffix_for_variant("video") == "mp4"

    def test_thumbnail_variant(self):
        assert suffix_for_variant("thumbnail") == "webp"

    def test_spritesheet_variant(self):
        assert suffix_for_variant("spritesheet") == "jpg"


@pytest.mark.unit
class TestGenerateFilename:
    """Test filename generation logic."""

    def test_basic_filename(self):
        result = generate_filename("abc123", "mp4")
        assert result == "abc123.mp4"

    def test_different_extension(self):
        result = generate_filename("test-id", "png")
        assert result == "test-id.png"

    @patch("sanzaru.utils.time.time", return_value=1234567890.0)
    def test_filename_with_timestamp(self, mock_time):
        result = generate_filename("img", "png", use_timestamp=True)
        assert result == "img_1234567890.png"

    @patch("sanzaru.utils.time.time", return_value=9999999999.0)
    def test_timestamp_precision(self, mock_time):
        """Verify timestamp is converted to int (no decimal)."""
        result = generate_filename("test", "jpg", use_timestamp=True)
        assert result == "test_9999999999.jpg"
        assert "." not in result.split("_")[1].split(".")[0]  # No decimal in timestamp part


@pytest.mark.unit
class TestValidateResourceId:
    """An OpenAI id is a URL path segment, so it decides which endpoint we call."""

    @pytest.mark.parametrize(
        "value",
        [
            "video_68d9f7a1b2c34d56789abcdef0123456",
            "resp_abc123",
            "vid_test123",
            "vid-with-dashes",
            "ft:gpt-41-mini:org:custom",  # ':' is in the alphabet for fine-tune-style ids
            "a",
            "z" * 128,
        ],
    )
    def test_real_ids_pass_through_unchanged(self, value):
        assert validate_resource_id(value, "video_id") == value

    @pytest.mark.parametrize(
        "value",
        [
            "../files/file-XYZ/content?",  # the download-redirect exploit
            "../files?",  # the org file-listing reflection
            "../models/ft:gpt-4.1:org:custom",  # DELETE against a fine-tuned model
            "..",
            "vid/123",
            "vid?limit=100",
            "vid#fragment",
            "vid%2Fabc",  # pre-encoded: httpx would not decode it, but nothing legitimate carries '%'
            "vid 123",
            "vid\n123",
            "vid.123",  # '.' is out precisely so '..' cannot be spelled
            "",
            "   ",
            "z" * 129,
        ],
    )
    def test_anything_that_could_move_the_path_is_rejected(self, value):
        with pytest.raises(ValueError, match="not a valid OpenAI resource id"):
            validate_resource_id(value, "video_id")

    def test_message_names_the_offending_parameter(self):
        with pytest.raises(ValueError, match="previous_video_id="):
            validate_resource_id("../files?", "previous_video_id")

    def test_non_string_is_a_value_error_not_a_type_error(self):
        # These ids arrive as JSON from an MCP client; a TypeError out of `re`
        # would be reported as an internal failure rather than bad usage.
        with pytest.raises(ValueError, match="not a valid OpenAI resource id"):
            validate_resource_id(None, "video_id")  # type: ignore[arg-type]

    def test_long_hostile_values_are_truncated_in_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            validate_resource_id("../" + "A" * 500, "video_id")
        assert len(str(excinfo.value)) < 200


@pytest.mark.unit
class TestReservedRunBookkeepingNames:
    """Every audio tool writes into one flat directory under a caller-chosen
    name, and the storage write is an unconditional truncate. Naming your output
    after another run's manifest or checkpoint destroyed state that run had
    already paid for, with no delete or undo to recover it (CWE-73)."""

    @pytest.mark.parametrize(
        "name",
        [
            "simrun_a1b2c3d4.json",
            "simrun_victim.json",
            "SIMRUN_A1B2C3D4.JSON",
        ],
    )
    def test_run_manifest_names_are_refused(self, name):
        with pytest.raises(ValueError, match="reserved"):
            reject_reserved_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "episode.mp3",
            "my_podcast_2026.mp3",
            "interview_final.wav",
            "simrun.json",
            "notes_about_simrun_stuff.mp3",
            "show_notes.json",
        ],
    )
    def test_ordinary_output_names_still_pass(self, name):
        assert reject_reserved_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "interview_20250826_part1.mp3",
            "standup_20240101_notes.json",
            "ep_deadbeef_v2.mp3",
            "Stitch_Test_a1b2c3d4_act1.mp3",
        ],
    )
    def test_date_stamped_recordings_are_not_mistaken_for_checkpoints(self, name):
        """Eight hex digits is also what a date looks like.

        A checkpoint-shaped name rule rejected `interview_20250826_part1.mp3` —
        an ordinary recording name — so checkpoint integrity moved to the
        signature, which can be exact, instead of the filename, which cannot.
        """
        assert reject_reserved_name(name) == name

    def test_the_parameter_name_reaches_the_message(self):
        with pytest.raises(ValueError, match="output file"):
            reject_reserved_name("simrun_x.json", "output file")
