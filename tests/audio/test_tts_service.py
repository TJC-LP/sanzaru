"""Tests for TTSService.create_speech across providers."""

import pytest

from sanzaru.audio.services import TTSService
from sanzaru.storage.local import LocalStorageBackend

pytestmark = [pytest.mark.audio, pytest.mark.integration]


@pytest.fixture
def storage(mocker, tmp_audio_path):
    backend = LocalStorageBackend(path_overrides={"audio": tmp_audio_path})
    mocker.patch("sanzaru.infrastructure.file_system.get_storage", return_value=backend)
    return backend


@pytest.fixture
def openai_client(mocker):
    response = mocker.MagicMock()
    response.content = b"FAKE_MP3"
    client = mocker.MagicMock()
    client.audio.speech.create = mocker.AsyncMock(return_value=response)
    mocker.patch("sanzaru.audio.providers.openai_provider.get_client", return_value=client)
    return client


@pytest.mark.anyio
async def test_instructions_reach_the_openai_api(openai_client, storage):
    """Regression: create_speech accepted `instructions` and silently dropped it.

    It was in the signature, documented, and threaded from both the MCP tool and
    the CLI, but never passed to client.audio.speech.create.
    """
    await TTSService().create_speech(
        text_prompt="Hello there",
        output_filename="out.mp3",
        instructions="Speak slowly and warmly",
    )

    kwargs = openai_client.audio.speech.create.call_args.kwargs
    assert kwargs["instructions"] == "Speak slowly and warmly"
    assert kwargs["response_format"] == "mp3"


@pytest.mark.anyio
async def test_omits_instructions_when_none(openai_client, storage):
    from openai._types import Omit

    await TTSService().create_speech(text_prompt="Hello", output_filename="out.mp3")

    kwargs = openai_client.audio.speech.create.call_args.kwargs
    assert isinstance(kwargs["instructions"], Omit)


@pytest.mark.anyio
async def test_openai_defaults_are_unchanged(openai_client, storage):
    await TTSService().create_speech(text_prompt="Hello", output_filename="out.mp3")

    kwargs = openai_client.audio.speech.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini-tts"
    assert kwargs["voice"] == "alloy"
    assert kwargs["speed"] == 1.0


@pytest.mark.anyio
async def test_writes_the_audio_file(openai_client, storage, tmp_audio_path):
    result = await TTSService().create_speech(text_prompt="Hello", output_filename="greeting.mp3")

    assert result.output_file == "greeting.mp3"
    assert (tmp_audio_path / "greeting.mp3").read_bytes() == b"FAKE_MP3"


@pytest.mark.anyio
async def test_generated_filename_when_unspecified(openai_client, storage):
    result = await TTSService().create_speech(text_prompt="Hello")

    assert result.output_file.startswith("speech_")
    assert result.output_file.endswith(".mp3")


@pytest.mark.anyio
async def test_elevenlabs_path(mocker, storage, tmp_audio_path, fake_elevenlabs):
    client = fake_elevenlabs.Client(chunks=(b"EL", b"AUDIO"))
    mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

    result = await TTSService().create_speech(
        text_prompt="Hello there",
        output_filename="el.mp3",
        provider="elevenlabs",
        voice="voice_abc",
    )

    assert (tmp_audio_path / result.output_file).read_bytes() == b"ELAUDIO"
    call = client.text_to_speech.calls[0]
    assert call["voice_id"] == "voice_abc"
    assert call["model_id"] == "eleven_v3"  # provider default


@pytest.mark.anyio
async def test_elevenlabs_requires_a_voice(mocker, storage, fake_elevenlabs):
    mocker.patch(
        "sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client",
        return_value=fake_elevenlabs.Client(),
    )

    with pytest.raises(ValueError, match="requires an explicit voice id"):
        await TTSService().create_speech(text_prompt="Hello", provider="elevenlabs")


@pytest.mark.anyio
async def test_unknown_provider_raises(storage):
    with pytest.raises(ValueError, match="unknown TTS provider"):
        await TTSService().create_speech(text_prompt="Hello", provider="azure")


# ==================== FEATURE DETECTION ====================


@pytest.mark.unit
def test_elevenlabs_unavailable_without_a_key(monkeypatch):
    from sanzaru.features import check_elevenlabs_available

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert check_elevenlabs_available() is False


@pytest.mark.unit
def test_tts_providers_report(monkeypatch):
    from sanzaru.features import get_tts_providers

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    providers = get_tts_providers()

    assert providers["openai"] is True
    assert providers["elevenlabs"] is False


@pytest.mark.unit
def test_get_available_features_stays_media_only():
    """capabilities maps these onto path types, so a provider key would KeyError."""
    from sanzaru.features import get_available_features

    assert set(get_available_features()) == {"video", "audio", "image"}


@pytest.mark.unit
def test_get_elevenlabs_client_requires_a_key(monkeypatch):
    from sanzaru.config import get_elevenlabs_client, set_elevenlabs_client

    set_elevenlabs_client(None)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    # RuntimeError, not ConfigurationError: the CLI maps RuntimeError to exit 3.
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY is not set"):
        get_elevenlabs_client()


@pytest.mark.unit
def test_set_elevenlabs_client_override(monkeypatch):
    from sanzaru.config import get_elevenlabs_client, set_elevenlabs_client

    sentinel = object()
    set_elevenlabs_client(sentinel)
    try:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        assert get_elevenlabs_client() is sentinel
    finally:
        set_elevenlabs_client(None)
