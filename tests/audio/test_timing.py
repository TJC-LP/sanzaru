# SPDX-License-Identifier: MIT
"""Word timing: forced alignment onto script turns, and Scribe transcription."""

from types import SimpleNamespace

import pytest

from sanzaru.audio import timing
from sanzaru.config import set_elevenlabs_client
from sanzaru.exceptions import TTSAPIError


class _FakeAlignment:
    """Answers forced_alignment.create with one evenly spaced entry per character, 0.1 s each."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def create(self, *, file, text):
        self.texts.append(text)
        characters = [SimpleNamespace(text=ch, start=i * 0.1, end=(i + 1) * 0.1) for i, ch in enumerate(text)]
        words = []
        position = 0
        for token in text.split(" "):
            words.append(SimpleNamespace(text=token, start=position * 0.1, end=(position + len(token)) * 0.1, loss=0.5))
            words.append(SimpleNamespace(text=" ", start=0.0, end=0.0, loss=0.0))
            position += len(token) + 1
        return SimpleNamespace(characters=characters, words=words, loss=0.5)


@pytest.fixture
def fake_client():
    client = SimpleNamespace(forced_alignment=_FakeAlignment(), speech_to_text=None)
    set_elevenlabs_client(client)  # type: ignore[arg-type]
    yield client
    set_elevenlabs_client(None)


@pytest.mark.unit
def test_spoken_text_drops_audio_tags_and_collapses_whitespace():
    assert timing.spoken_text("[thoughtful]  Three places:\n growth [laughs] and debt.") == (
        "Three places: growth and debt."
    )


@pytest.mark.unit
async def test_align_turns_maps_each_turn_by_character_offset(fake_client):
    result, turns = await timing.align_turns(b"mp3", "ep.mp3", ["Where from?", "[calm] Three places."], ["A", "B"])

    assert fake_client.forced_alignment.texts == ["Where from? Three places."]
    assert [(t.index, t.speaker, t.text) for t in turns] == [(0, "A", "Where from?"), (1, "B", "Three places.")]
    assert turns[0].start == 0.0 and turns[0].end == pytest.approx(1.1)
    assert turns[1].start == pytest.approx(1.2) and turns[1].end == pytest.approx(2.5)
    assert [w.text for w in turns[0].words] == ["Where", "from?"]
    assert [w.text for w in turns[1].words] == ["Three", "places."]
    assert [w.text for w in result.words] == ["Where", "from?", "Three", "places."]
    assert result.turns == turns and result.loss == 0.5


@pytest.mark.unit
async def test_align_returns_words_without_turns(fake_client):
    result = await timing.align(b"mp3", "n.mp3", "Growth and debt.")

    assert result.turns is None
    assert [(w.text, w.start) for w in result.words] == [
        ("Growth", 0.0),
        ("and", pytest.approx(0.7)),
        ("debt.", pytest.approx(1.1)),
    ]


@pytest.mark.unit
async def test_a_tag_only_turn_has_no_span(fake_client):
    _, turns = await timing.align_turns(b"mp3", "ep.mp3", ["[laughs]", "Hello.", "[laughs]", "Bye.", "[sighs]"])

    assert fake_client.forced_alignment.texts == ["Hello. Bye."]
    assert [(t.index, t.text, t.start) for t in turns] == [(1, "Hello.", 0.0), (3, "Bye.", pytest.approx(0.7))]


@pytest.mark.unit
async def test_align_refuses_text_that_is_only_tags(fake_client):
    with pytest.raises(ValueError, match="nothing to align"):
        await timing.align(b"mp3", "n.mp3", "[sighs]")


@pytest.mark.unit
async def test_align_refuses_a_character_count_that_does_not_match(fake_client):
    async def short(*, file, text):
        return SimpleNamespace(characters=[], words=[], loss=0.0)

    fake_client.forced_alignment.create = short
    with pytest.raises(TTSAPIError, match="returned 0 characters"):
        await timing.align(b"mp3", "n.mp3", "Hello.")


@pytest.mark.unit
async def test_scribe_keeps_words_and_drops_spacing_and_events():
    calls = []

    async def convert(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text="Hi there.",
            words=[
                SimpleNamespace(text="Hi", start=0.1, end=0.3, type="word"),
                SimpleNamespace(text=" ", start=0.3, end=0.35, type="spacing"),
                SimpleNamespace(text="(laughs)", start=0.35, end=0.6, type="audio_event"),
                SimpleNamespace(text="there.", start=0.6, end=0.9, type="word"),
            ],
        )

    set_elevenlabs_client(SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert)))  # type: ignore[arg-type]
    try:
        result = await timing.transcribe_scribe(b"mp3", "ep.mp3")
    finally:
        set_elevenlabs_client(None)

    assert calls[0]["model_id"] == "scribe_v2" and calls[0]["timestamps_granularity"] == "word"
    assert result.source == "scribe_v2" and result.text == "Hi there."
    assert [(w.text, w.start, w.end) for w in result.words] == [("Hi", 0.1, 0.3), ("there.", 0.6, 0.9)]


@pytest.mark.unit
async def test_scribe_refuses_another_model():
    with pytest.raises(ValueError, match="not a Scribe model"):
        await timing.transcribe_scribe(b"mp3", "ep.mp3", model="whisper-1")


@pytest.mark.unit
async def test_a_server_error_is_retried_then_succeeds(fake_client, monkeypatch):
    monkeypatch.setattr(timing, "_BACKOFF_BASE_SECONDS", 0.0)
    real = fake_client.forced_alignment.create
    attempts = []

    async def flaky(*, file, text):
        attempts.append(1)
        if len(attempts) == 1:
            raise type("ApiError", (Exception,), {"status_code": 503})()
        return await real(file=file, text=text)

    fake_client.forced_alignment.create = flaky
    result = await timing.align(b"mp3", "n.mp3", "Hello.")

    assert len(attempts) == 2 and [w.text for w in result.words] == ["Hello."]


@pytest.mark.unit
async def test_a_client_error_is_not_retried(fake_client):
    attempts = []

    async def refused(*, file, text):
        attempts.append(1)
        raise type("ApiError", (Exception,), {"status_code": 403})()

    fake_client.forced_alignment.create = refused
    with pytest.raises(TTSAPIError, match="HTTP 403"):
        await timing.align(b"mp3", "n.mp3", "Hello.")
    assert len(attempts) == 1


@pytest.mark.unit
def test_spoken_text_drops_long_and_wrapped_tags_like_verify():
    assert timing.spoken_text("[speaking quietly, as if sharing a secret with you] We checked.") == "We checked."
    assert timing.spoken_text("[slowly,\n warmly] Yes.") == "Yes."
