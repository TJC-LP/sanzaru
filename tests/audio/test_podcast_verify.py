"""Tests for the podcast verify + retry pass (#35).

TTS drops segment tails, and occasionally whole short segments, at random and
with no error — and `PodcastResult.transcript` is only an echo of the input
script, so it is no evidence the audio contains the words. These tests cover
the two halves: the predicate that decides a segment is missing, and the loop
that re-renders it once.

Unlike the other podcast tests, the end-to-end cases here let the *real*
`_stitch_audio` run. Every existing test stubs it, so nothing would have
noticed a change to how `segment_bytes_list` is assembled — which is exactly
what the retry mutates.
"""

import pytest
from pydub import AudioSegment

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.podcast import (
    VERIFY_SHORT_SEGMENT_WORDS,
    _best_window_similarity,
    _verdict_for,
    generate_podcast,
)

pytestmark = pytest.mark.audio

SPOKEN = "the model is the easy part to demo and the hard part is everything after that"


def _tone(ms: int = 200) -> bytes:
    """A real mp3, so the stitcher has something it can actually decode."""
    from io import BytesIO

    buffer = BytesIO()
    AudioSegment.silent(duration=ms, frame_rate=16000).export(buffer, format="mp3")
    return buffer.getvalue()


@pytest.mark.unit
class TestVerdictFor:
    def test_a_faithful_render_passes(self):
        verdict = _verdict_for(0, "Alex", SPOKEN, SPOKEN)
        assert verdict.ok
        assert verdict.reason == ""
        assert verdict.similarity == 1.0

    def test_normal_asr_disagreement_still_passes(self):
        """Punctuation and casing wobble constantly; that is not a drop."""
        rendered = "The model is the easy part to demo, and the hard part is everything after that."
        assert _verdict_for(0, "Alex", SPOKEN, rendered).ok

    def test_a_dropped_tail_is_caught(self):
        """The documented failure: the last words simply are not there."""
        rendered = "the model is the easy part to demo and the hard part is"
        verdict = _verdict_for(3, "Alex", SPOKEN, rendered)
        assert not verdict.ok
        assert verdict.reason == "tail_missing"
        assert verdict.index == 3

    def test_a_whole_short_segment_missing_is_caught(self):
        """A three-word segment vanished entirely in a production run."""
        verdict = _verdict_for(7, "Sam", "Where does that stand?", "completely unrelated speech here")
        assert not verdict.ok
        assert verdict.reason == "segment_missing"

    def test_a_short_segment_that_is_present_passes(self):
        verdict = _verdict_for(7, "Sam", "Where does that stand?", "Where does that stand?")
        assert verdict.ok

    def test_a_short_segment_inside_a_longer_render_is_found(self):
        """Dialogue units render several turns into one buffer."""
        rendered = "Sure, I hear you. Where does that stand? Well, it depends on the quarter."
        assert _verdict_for(7, "Sam", "Where does that stand?", rendered).ok

    def test_the_short_segment_boundary_is_where_it_says(self):
        short = " ".join(["word"] * VERIFY_SHORT_SEGMENT_WORDS)
        assert _verdict_for(0, "A", short, "nothing like it at all").reason == "segment_missing"

    def test_a_render_that_lost_the_middle_is_caught(self):
        """The tail can survive while the body is gone."""
        verdict = _verdict_for(0, "Alex", SPOKEN, "the model is everything after that")
        assert not verdict.ok

    def test_empty_audio_fails_rather_than_passing_vacuously(self):
        verdict = _verdict_for(0, "Alex", SPOKEN, "")
        assert not verdict.ok
        assert verdict.similarity == 0.0


@pytest.mark.unit
class TestBestWindowSimilarity:
    def test_finds_a_needle_anywhere_in_the_haystack(self):
        assert _best_window_similarity("needle words here", "lots of text needle words here and more") == 1.0

    def test_absent_text_scores_low(self):
        assert _best_window_similarity("needle words here", "nothing remotely similar") < 0.5

    def test_a_haystack_shorter_than_the_needle_still_compares(self):
        assert 0.0 < _best_window_similarity("one two three four", "one two") < 1.0

    def test_an_empty_needle_is_vacuously_present(self):
        assert _best_window_similarity("", "anything") == 1.0

    def test_an_empty_haystack_contains_nothing(self):
        assert _best_window_similarity("something", "") == 0.0


@pytest.fixture
def podcast_env(mocker, tmp_path):
    """Real storage and a real stitcher; only TTS and transcription are faked.

    Crucially, transcription answers based on the **actual bytes** it is handed,
    not on a filename or a call counter. Each render of a segment produces a
    distinguishable buffer, so a retry that renders new audio without putting it
    back into `segment_bytes_list` transcribes the *old* take and still fails —
    which is the whole property the retry has to have.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    mocker.patch(
        "sanzaru.infrastructure.file_system.get_storage",
        return_value=LocalStorageBackend(path_overrides={"audio": audio_dir}),
    )

    tts_calls: list[str] = []

    def install(takes: dict[str, list[str]], failing_transcribe: bool = False):
        """`takes` maps segment text -> what each successive render sounds like."""
        heard: dict[bytes, str] = {}
        attempts: dict[str, int] = {}

        async def fake_speech(**kwargs):
            spoken = kwargs["input"]
            tts_calls.append(spoken)
            attempt = attempts.get(spoken, 0)
            attempts[spoken] = attempt + 1
            # A distinct duration per take makes the bytes distinguishable.
            audio = _tone(200 + 40 * len(tts_calls))
            sounds_like = takes.get(spoken, [spoken])
            heard[audio] = sounds_like[min(attempt, len(sounds_like) - 1)]
            response = mocker.MagicMock()
            response.content = audio
            return response

        mock_client = mocker.MagicMock()
        mock_client.audio.speech.create = mocker.AsyncMock(side_effect=fake_speech)
        mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=mock_client)

        async def fake_transcribe(audio, filename, model=None):
            if failing_transcribe:
                raise RuntimeError("transcription is down")
            return heard.get(audio, "")

        mocker.patch("sanzaru.tools.podcast.transcribe_bytes", side_effect=fake_transcribe)
        return tts_calls

    return audio_dir, install


def _script(*texts: str) -> dict:
    return {
        "title": "Verify",
        "speakers": [{"id": "a", "name": "Alex", "voice": "ash"}],
        "segments": [{"speaker": "a", "text": text} for text in texts],
    }


@pytest.mark.integration
@pytest.mark.anyio
class TestVerifyPass:
    async def test_verify_off_by_default_costs_nothing(self, podcast_env, mocker):
        audio_dir, install = podcast_env
        install({})
        transcribe = mocker.patch("sanzaru.tools.podcast.transcribe_bytes")

        result = await generate_podcast(_script(SPOKEN, "A second segment of some length here."))

        assert result.verified is None
        assert result.segment_verdicts == []
        transcribe.assert_not_called()

    async def test_a_clean_render_verifies(self, podcast_env):
        audio_dir, install = podcast_env
        second = "A second segment of some length here."
        install({})  # every take sounds like its own script text

        result = await generate_podcast(_script(SPOKEN, second), verify=True)

        assert result.verified is True
        assert result.verify_retries == 0
        assert [v.ok for v in result.segment_verdicts] == [True, True]

    async def test_a_dropped_tail_is_re_rendered_once_and_recovers(self, podcast_env):
        audio_dir, install = podcast_env
        second = "A second segment of some length here."
        tts_calls = install(
            # First take of segment 0 lost its tail; the retry is clean.
            {SPOKEN: ["the model is the easy part to demo and the hard", SPOKEN]}
        )

        result = await generate_podcast(_script(SPOKEN, second), verify=True)

        assert result.verified is True
        assert result.verify_retries == 1
        assert [v.retried for v in result.segment_verdicts] == [True, False]
        # Segment 0 rendered twice, segment 1 once.
        assert tts_calls.count(SPOKEN) == 2
        assert tts_calls.count(second) == 1

    async def test_a_segment_that_fails_twice_is_reported_not_hidden(self, podcast_env):
        audio_dir, install = podcast_env
        # Every take drops the tail — a sticky failure, not a roving one.
        install({SPOKEN: ["the model is the easy part to demo and the hard"]})

        result = await generate_podcast(_script(SPOKEN), verify=True)

        assert result.verified is False
        assert result.segment_verdicts[0].reason == "tail_missing"
        assert result.segment_verdicts[0].retried is True
        # The episode is still written — a flawed take beats no take.
        assert (audio_dir / result.output_file).exists()

    async def test_the_retried_audio_actually_reaches_the_file(self, podcast_env):
        """The real stitcher runs here, so a retry that updated nothing shows up."""
        audio_dir, install = podcast_env
        install({})

        result = await generate_podcast(_script(SPOKEN), verify=True)

        written = (audio_dir / result.output_file).read_bytes()
        assert len(written) > 0
        assert AudioSegment.from_mp3(__import__("io").BytesIO(written)).duration_seconds > 0

    async def test_a_transcription_failure_does_not_lose_the_episode(self, podcast_env):
        """Verification degrades to "unverified", never to a lost render."""
        audio_dir, install = podcast_env
        install({}, failing_transcribe=True)

        result = await generate_podcast(_script(SPOKEN), verify=True)

        assert (audio_dir / result.output_file).exists()
        assert result.verified is True, "an unverifiable segment is not a failed one"
        assert result.segment_verdicts[0].reason == "not_transcribed"
        assert result.verify_retries == 0
