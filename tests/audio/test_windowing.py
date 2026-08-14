"""Tests for overlapping-window planning and transcript stitching (#39).

The failure this guards against is *silent*: a long file came back truncated
with no error, and on another run with a stretch missing from the middle. So
the property that matters is coverage — every second of the file lands in some
window, and merging never deletes speech it was unsure about.
"""

import pytest

from sanzaru.audio.verification import aligned_words
from sanzaru.audio.windowing import (
    MIN_OVERLAP_WORDS,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
    merge_window_texts,
    plan_windows,
)

pytestmark = pytest.mark.audio


@pytest.mark.unit
class TestPlanWindows:
    def test_a_short_file_is_one_window(self):
        windows = plan_windows(30.0)
        assert len(windows) == 1
        assert (windows[0].start_s, windows[0].end_s) == (0.0, 30.0)

    def test_an_empty_file_has_no_windows(self):
        assert plan_windows(0.0) == []
        assert plan_windows(-1.0) == []

    @pytest.mark.parametrize("duration", [91.0, 200.0, 631.0, 1800.0, 3600.0])
    def test_every_second_of_the_file_is_covered(self, duration):
        """The whole point: no gap anywhere, including the tail.

        A trailing sliver is exactly where truncation hides, so the last window
        must reach the end of the file rather than wherever the stride landed.
        """
        windows = plan_windows(duration)

        assert windows[0].start_s == 0.0
        assert windows[-1].end_s == pytest.approx(duration)
        for earlier, later in zip(windows, windows[1:], strict=False):
            assert later.start_s <= earlier.end_s, "a gap between windows would lose speech"

    @pytest.mark.parametrize("duration", [200.0, 631.0, 1800.0])
    def test_adjacent_windows_overlap(self, duration):
        windows = plan_windows(duration)
        overlaps = [a.end_s - b.start_s for a, b in zip(windows, windows[1:], strict=False)]
        # The final window may overlap more, having been pinned to the end.
        assert all(o >= WINDOW_SECONDS - STRIDE_SECONDS for o in overlaps)

    def test_windows_are_indexed_in_order(self):
        windows = plan_windows(1000.0)
        assert [w.index for w in windows] == list(range(len(windows)))

    def test_no_window_exceeds_the_window_length(self):
        for window in plan_windows(1000.0):
            assert window.duration_s <= WINDOW_SECONDS + 1e-6


@pytest.mark.unit
class TestMergeWindowTexts:
    def test_removes_the_overlap_between_two_windows(self):
        first = "the model is the easy part to demo and the hard part is everything after"
        second = "the hard part is everything after that including evaluation and rollout"

        merged = merge_window_texts([first, second])

        assert merged == (
            "the model is the easy part to demo and the hard part is everything after "
            "that including evaluation and rollout"
        )

    def test_keeps_casing_and_punctuation_from_the_raw_text(self):
        """Matching is normalised; slicing is not, or the transcript degrades."""
        merged = merge_window_texts(
            [
                "We shipped it, and it worked on Tuesday, finally.",
                "and it worked on Tuesday, finally. Nobody noticed.",
            ]
        )
        assert merged == "We shipped it, and it worked on Tuesday, finally. Nobody noticed."

    def test_a_single_window_is_returned_as_is(self):
        assert merge_window_texts(["just the one"]) == "just the one"

    def test_empty_windows_are_skipped(self):
        """A failed window leaves a gap rather than truncating the transcript."""
        assert merge_window_texts(["first part", "", "second part"]) == "first part second part"

    def test_nothing_at_all_is_empty(self):
        assert merge_window_texts(["", ""]) == ""

    def test_unrelated_windows_are_concatenated_rather_than_guessed_at(self):
        """When no seam is confident, duplication beats deletion.

        Deleting speech is the bug this module exists to prevent, so an
        uncertain join must fail toward keeping too much.
        """
        merged = merge_window_texts(["alpha bravo charlie", "delta echo foxtrot"])
        assert merged == "alpha bravo charlie delta echo foxtrot"

    def test_a_short_coincidental_match_is_not_treated_as_overlap(self):
        """Below MIN_OVERLAP_WORDS, keep both: a common phrase is not a seam.

        "the thing" turns up in unrelated passages constantly, and treating one
        as an overlap would delete the speech after it.
        """
        merged = merge_window_texts(["ship the thing", "the thing that matters"])
        assert merged == "ship the thing the thing that matters"

    def test_the_overlap_threshold_is_what_decides(self):
        shared = " ".join(f"word{i}" for i in range(MIN_OVERLAP_WORDS))
        merged = merge_window_texts([f"lead in {shared}", f"{shared} tail out"])
        assert merged == f"lead in {shared} tail out"

    def test_merging_many_windows_preserves_order(self):
        """Overlaps here are MIN_OVERLAP_WORDS long, as a real 15s one would be."""
        parts = [
            "one two three four five six seven eight",
            "five six seven eight nine ten eleven twelve",
            "nine ten eleven twelve thirteen fourteen",
        ]
        assert merge_window_texts(parts) == (
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        )


@pytest.mark.unit
class TestAlignedWords:
    def test_raw_and_comparable_stay_index_aligned(self):
        raw, norm = aligned_words("Hello, world! -- it's Tuesday.")
        assert len(raw) == len(norm)
        assert raw[0] == "Hello,"
        assert norm[0] == "hello"

    def test_a_token_that_normalises_to_nothing_is_kept_in_place(self):
        """`words()` drops these; alignment cannot, or slicing indexes shift."""
        raw, norm = aligned_words("real -- words")
        assert len(raw) == len(norm) == 3
        assert norm[1] == ""
