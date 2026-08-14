"""Tests for TranscriptionService, including the windowed long-file path (#39).

There were no tests of this service at all before. It is where a 13-minute file
came back silently truncated, so the cases below are built around real audio of
a real length: pydub actually slices it, and the fake client records which
windows were uploaded.
"""

import pytest
from pydub import AudioSegment

from sanzaru.audio.models import TranscriptionResult
from sanzaru.audio.services.transcription_service import TranscriptionService, _warn_if_short
from sanzaru.audio.windowing import CHUNK_THRESHOLD_SECONDS, plan_windows
from sanzaru.storage.local import LocalStorageBackend

pytestmark = pytest.mark.audio


class _FakeTranscriptions:
    """Answers per uploaded filename, recording every call."""

    def __init__(self, answers=None, failures=()):
        self.answers = answers or {}
        self.failures = set(failures)
        self.calls = []

    async def create(self, *, file, model, **kwargs):
        filename, _buffer = file
        self.calls.append({"filename": filename, "model": model, **kwargs})
        if filename in self.failures:
            raise RuntimeError("upstream said no")
        return self.answers.get(filename, "some words from this window")


@pytest.fixture
def fake_client(mocker):
    def build(answers=None, failures=()):
        import types

        client = types.SimpleNamespace(
            audio=types.SimpleNamespace(transcriptions=_FakeTranscriptions(answers, failures))
        )
        mocker.patch("sanzaru.audio.services.transcription_service.get_client", return_value=client)
        return client

    return build


@pytest.fixture
def audio_dir(tmp_path, mocker):
    """A real audio directory wired into both the repo and the service."""
    directory = tmp_path / "audio"
    directory.mkdir()
    backend = LocalStorageBackend(path_overrides={"audio": directory})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=backend)
    mocker.patch("sanzaru.audio.services.transcription_service.get_storage", return_value=backend)
    return directory


def _write_silence(directory, name: str, seconds: float) -> None:
    AudioSegment.silent(duration=int(seconds * 1000), frame_rate=8000).export(str(directory / name), format="mp3")


@pytest.mark.integration
@pytest.mark.anyio
class TestShortFiles:
    async def test_a_short_file_takes_the_single_request_path(self, audio_dir, fake_client):
        _write_silence(audio_dir, "short.mp3", 5.0)
        client = fake_client({"short.mp3": "the whole thing"})

        result = await TranscriptionService().transcribe_audio("short.mp3")

        assert result.text == "the whole thing"
        assert result.chunked is False
        assert result.windows is None
        assert [c["filename"] for c in client.audio.transcriptions.calls] == ["short.mp3"]

    async def test_prompt_and_granularities_still_reach_the_api(self, audio_dir, fake_client):
        _write_silence(audio_dir, "short.mp3", 5.0)
        client = fake_client()

        await TranscriptionService().transcribe_audio("short.mp3", prompt="a hint", timestamp_granularities=["segment"])

        call = client.audio.transcriptions.calls[0]
        assert call["prompt"] == "a hint"
        assert call["timestamp_granularities"] == ["segment"]

    async def test_an_unreadable_file_still_transcribes(self, audio_dir, fake_client):
        """A duration probe failure must not refuse the file (#39 is additive)."""
        (audio_dir / "notaudio.mp3").write_bytes(b"definitely not an mp3")
        fake_client({"notaudio.mp3": "transcribed anyway"})

        result = await TranscriptionService().transcribe_audio("notaudio.mp3")

        assert result.text == "transcribed anyway"
        assert result.chunked is False


@pytest.mark.integration
@pytest.mark.anyio
class TestWindowedFiles:
    async def test_a_long_file_is_windowed_and_stitched(self, audio_dir, fake_client):
        """The case from #39: long enough that one request truncates silently."""
        duration = CHUNK_THRESHOLD_SECONDS + 120.0
        _write_silence(audio_dir, "long.mp3", duration)
        expected = plan_windows(duration)
        client = fake_client(
            {f"long_w{w.index}.mp3": f"window {w.index} text here" for w in expected},
        )

        result = await TranscriptionService().transcribe_audio("long.mp3")

        assert result.chunked is True
        uploaded = [c["filename"] for c in client.audio.transcriptions.calls]
        assert sorted(uploaded) == sorted(f"long_w{w.index}.mp3" for w in expected)
        assert result.windows is not None
        assert len(result.windows) == len(expected)
        # Windows are reported in order regardless of completion order.
        assert [w["index"] for w in result.windows] == list(range(len(expected)))
        for window in result.windows:
            assert f"window {window['index']} text here" in result.text

    async def test_the_windows_cover_the_whole_file(self, audio_dir, fake_client):
        duration = CHUNK_THRESHOLD_SECONDS + 200.0
        _write_silence(audio_dir, "long.mp3", duration)
        fake_client()

        result = await TranscriptionService().transcribe_audio("long.mp3")

        assert result.windows is not None
        assert result.windows[0]["start_s"] == 0.0
        # pydub's duration differs from the requested one by an encoder frame or two.
        assert result.windows[-1]["end_s"] == pytest.approx(result.duration, abs=1.0)

    async def test_one_failed_window_does_not_lose_the_transcript(self, audio_dir, fake_client):
        """A gap beats an exception: the rest of the file is still worth having."""
        duration = CHUNK_THRESHOLD_SECONDS + 120.0
        _write_silence(audio_dir, "long.mp3", duration)
        fake_client(failures={"long_w1.mp3"})

        result = await TranscriptionService().transcribe_audio("long.mp3")

        assert result.chunked is True
        assert result.text, "the surviving windows still produced a transcript"
        failed = [w for w in (result.windows or []) if not w["text"]]
        assert [w["index"] for w in failed] == [1]

    async def test_the_model_choice_is_carried_into_every_window(self, audio_dir, fake_client):
        _write_silence(audio_dir, "long.mp3", CHUNK_THRESHOLD_SECONDS + 60.0)
        client = fake_client()

        await TranscriptionService().transcribe_audio("long.mp3", model="whisper-1")

        assert {c["model"] for c in client.audio.transcriptions.calls} == {"whisper-1"}


@pytest.mark.unit
class TestSilentTruncationWarning:
    """The other half of #39: say so when a response covers less than the file.

    A file under the windowing threshold cannot hit the ~10.5 minute
    truncation, but a response that covers a fraction of the audio should not
    come back looking complete.
    """

    def _warn(self, caplog, result, probed):
        with caplog.at_level("WARNING", logger="sanzaru"):
            _warn_if_short("ep.mp3", result, probed)
        return "may have stopped early" in caplog.text

    def test_a_reported_duration_short_of_the_file_warns(self, caplog):
        assert self._warn(caplog, TranscriptionResult(text="...", duration=630.0), 794.0)

    def test_a_full_length_transcript_says_nothing(self, caplog):
        assert not self._warn(caplog, TranscriptionResult(text="...", duration=790.0), 794.0)

    def test_segment_end_timestamps_are_used_when_duration_is_absent(self, caplog):
        """`verbose_json` gives segments; the last end is the coverage."""
        segments = [{"start": 0.0, "end": 100.0}, {"start": 100.0, "end": 210.0}]
        assert self._warn(caplog, TranscriptionResult(text="...", segments=segments), 800.0)

    def test_plain_text_responses_claim_nothing(self, caplog):
        """No coverage information means no warning, right or wrong.

        This is precisely why long files are windowed rather than checked after
        the fact — with `response_format="text"` there is nothing to compare.
        """
        assert not self._warn(caplog, TranscriptionResult(text="a short answer"), 800.0)

    def test_an_unknown_duration_claims_nothing(self, caplog):
        assert not self._warn(caplog, TranscriptionResult(text="...", duration=10.0), None)


@pytest.mark.integration
@pytest.mark.anyio
class TestWarningInContext:
    async def test_a_normal_short_transcription_is_quiet(self, audio_dir, fake_client, caplog):
        _write_silence(audio_dir, "short.mp3", 10.0)
        fake_client()

        with caplog.at_level("WARNING", logger="sanzaru"):
            await TranscriptionService().transcribe_audio("short.mp3")

        assert "may have stopped early" not in caplog.text
