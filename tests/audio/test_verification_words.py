"""What counts as the same word across a script and a transcript.

`words()` is shared by the podcast verify pass and realtime QC, so its
normalisation decides whether a faithful render is reported as a drop.
"""

import pytest

from sanzaru.audio.verification import similarity, strip_audio_tags, words

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("point seven", "0.7"),
        ("one point two", "1.2"),
        ("two thousand", "2,000"),
        ("twenty-one", "21"),
        ("one hundred and twenty five dollars", "$125"),
        ("three point five million", "3.5 million"),
        ("five percent", "5%"),
        ("point five", ".5"),
    ],
)
def test_spoken_and_written_numbers_compare_equal(spoken, written):
    assert words(spoken) == words(written)


def test_a_digit_sequence_is_not_summed():
    """`one two three` is a cadence, not 6."""
    assert words("one two three") == ["1", "2", "3"]


def test_two_teens_in_a_row_stay_separate():
    assert words("eleven twelve") == ["11", "12"]


def test_point_alone_is_a_word():
    assert words("the point is") == ["the", "point", "is"]


def test_scale_words_need_a_number_before_them():
    assert words("a thousand thanks") == ["a", "thousand", "thanks"]


def test_internal_dashes_split():
    assert words("back-to-back") == words("back to back")


def test_numbers_do_not_lift_unrelated_disagreement():
    """Normalisation fixes spelling, not missing speech."""
    assert similarity("same length at point seven as at one point two", "same length at") < 0.75


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[laughs] [defensive] Oh, we checked.", ["oh", "we", "checked"]),
        ("[strong French accent] Bonjour", ["bonjour"]),
        ("No tags here.", ["no", "tags", "here"]),
    ],
)
def test_strip_audio_tags(text, expected):
    assert words(strip_audio_tags(text)) == expected


def test_an_unclosed_bracket_is_left_alone():
    assert words(strip_audio_tags("[unfinished thought")) == ["unfinished", "thought"]
