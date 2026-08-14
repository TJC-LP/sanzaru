"""Tests for the TTS provider layer (registry, validation, chunk orchestration).

The ElevenLabs provider only ever touches `client.text_to_speech.convert(**kwargs)`
and iterates the result, so `FakeElevenLabsClient` needs no SDK import — these
tests run without the [elevenlabs] extra installed.
"""

import pathlib
import re

import pytest

from sanzaru.audio.providers import (
    SpeechRequest,
    get_provider,
    synthesize_speech,
)
from sanzaru.exceptions import TTSAPIError

pytestmark = pytest.mark.audio


def elevenlabs_request(**overrides) -> SpeechRequest:
    defaults = {"text": "hello", "voice": "voice_abc", "model": "eleven_multilingual_v2"}
    return SpeechRequest(**{**defaults, **overrides})


# ---------- registry ----------


@pytest.mark.unit
class TestRegistry:
    def test_default_provider_is_openai(self):
        assert get_provider(None).name == "openai"

    def test_named_providers(self):
        assert get_provider("openai").name == "openai"
        assert get_provider("elevenlabs").name == "elevenlabs"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown TTS provider"):
            get_provider("azure")


# ---------- model / voice resolution ----------


@pytest.mark.unit
class TestResolution:
    def test_openai_defaults(self):
        provider = get_provider("openai")
        assert provider.resolve_model(None) == "gpt-4o-mini-tts"
        assert provider.resolve_voice(None) == "alloy"

    def test_elevenlabs_defaults_to_v3(self):
        assert get_provider("elevenlabs").resolve_model(None) == "eleven_v3"

    def test_elevenlabs_rejects_openai_model(self):
        with pytest.raises(ValueError, match="not an ElevenLabs model"):
            get_provider("elevenlabs").resolve_model("tts-1")

    def test_elevenlabs_requires_a_voice_id(self):
        with pytest.raises(ValueError, match="requires an explicit voice id"):
            get_provider("elevenlabs").resolve_voice(None)

    def test_elevenlabs_strips_voice_id(self):
        assert get_provider("elevenlabs").resolve_voice("  abc  ") == "abc"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("eleven_v3", 3000),
            ("eleven_multilingual_v2", 10000),
            ("eleven_flash_v2_5", 40000),
            ("eleven_turbo_v2_5", 40000),
        ],
    )
    def test_elevenlabs_chunk_limits_are_per_model(self, model, expected):
        assert get_provider("elevenlabs").max_chunk_chars(model) == expected

    def test_openai_chunk_limit(self):
        assert get_provider("openai").max_chunk_chars("gpt-4o-mini-tts") == 4000

    def test_openai_is_unbounded_by_default(self):
        # 0 == unbounded, which is the pre-provider-layer behavior.
        assert get_provider("openai").max_concurrency("gpt-4o-mini-tts") == 0

    def test_elevenlabs_concurrency_default_and_env_override(self, monkeypatch):
        provider = get_provider("elevenlabs")
        # 2 is the Free-tier cap for non-Flash models; 3 returns HTTP 429 there.
        assert provider.max_concurrency("eleven_v3") == 2
        assert provider.max_concurrency("eleven_flash_v2_5") == 4
        monkeypatch.setenv("SANZARU_ELEVENLABS_MAX_CONCURRENCY", "7")
        assert provider.max_concurrency("eleven_v3") == 7

    def test_malformed_concurrency_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("SANZARU_ELEVENLABS_MAX_CONCURRENCY", "lots")
        assert get_provider("elevenlabs").max_concurrency("eleven_v3") == 2

    def test_env_example_does_not_suggest_a_429(self):
        """The docs used to recommend 3, which this PR verified returns HTTP 429."""
        from sanzaru.audio.constants import ELEVENLABS_DEFAULT_CONCURRENCY

        env_example = pathlib.Path(__file__).resolve().parents[2] / ".env.example"
        suggested = re.findall(
            r"^#?\s*SANZARU_ELEVENLABS_MAX_CONCURRENCY=(\d+)", env_example.read_text(), flags=re.MULTILINE
        )

        assert suggested, "expected .env.example to document SANZARU_ELEVENLABS_MAX_CONCURRENCY"
        assert all(int(value) <= min(ELEVENLABS_DEFAULT_CONCURRENCY.values()) for value in suggested)


# ---------- validation ----------


@pytest.mark.unit
class TestOpenAIValidation:
    @pytest.mark.parametrize("speed", [0.25, 1.0, 4.0])
    def test_speed_in_range(self, speed):
        get_provider("openai").validate(SpeechRequest(text="t", voice="alloy", model="tts-1", speed=speed))

    @pytest.mark.parametrize("speed", [0.1, 5.0])
    def test_speed_out_of_range(self, speed):
        with pytest.raises(ValueError, match="between 0.25 and 4.0"):
            get_provider("openai").validate(SpeechRequest(text="t", voice="alloy", model="tts-1", speed=speed))

    def test_voice_settings_rejected(self):
        with pytest.raises(ValueError, match="ElevenLabs-only"):
            get_provider("openai").validate(
                SpeechRequest(text="t", voice="alloy", model="tts-1", voice_settings={"stability": 0.5})
            )

    def test_unknown_voice_warns_but_passes(self, caplog):
        # OpenAI adds voices between our releases; a hard allowlist would break.
        get_provider("openai").validate(SpeechRequest(text="t", voice="brand_new", model="tts-1"))
        assert "not a known OpenAI voice" in caplog.text


@pytest.mark.unit
class TestElevenLabsValidation:
    @pytest.mark.parametrize("speed", [0.7, 1.0, 1.2])
    def test_speed_in_range(self, speed):
        get_provider("elevenlabs").validate(elevenlabs_request(speed=speed))

    @pytest.mark.parametrize("speed", [0.5, 1.5])
    def test_speed_out_of_range(self, speed):
        with pytest.raises(ValueError, match="between 0.7 and 1.2"):
            get_provider("elevenlabs").validate(elevenlabs_request(speed=speed))

    def test_openai_speed_range_does_not_carry_over(self):
        # 2.0 is legal on OpenAI; silently rescaling it here would make the same
        # number mean two different things.
        with pytest.raises(ValueError, match="does not carry over"):
            get_provider("elevenlabs").validate(elevenlabs_request(speed=2.0))

    def test_v3_rejects_any_speed_change(self):
        with pytest.raises(ValueError, match="does not support speed"):
            get_provider("elevenlabs").validate(elevenlabs_request(model="eleven_v3", speed=1.1))

    def test_v3_accepts_neutral_speed(self):
        get_provider("elevenlabs").validate(elevenlabs_request(model="eleven_v3", speed=1.0))

    def test_voice_settings_speed_takes_precedence(self):
        # request.speed is legal, voice_settings["speed"] is not — the latter wins.
        with pytest.raises(ValueError, match="between 0.7 and 1.2"):
            get_provider("elevenlabs").validate(elevenlabs_request(speed=1.0, voice_settings={"speed": 1.9}))

    def test_unknown_voice_settings_key(self):
        with pytest.raises(ValueError, match="unknown key 'warmth'"):
            get_provider("elevenlabs").validate(elevenlabs_request(voice_settings={"warmth": 0.5}))

    @pytest.mark.parametrize("value", ["high", True, None])
    def test_non_numeric_voice_setting_is_a_value_error(self, value):
        # Not the TypeError the bare range comparison used to raise: the CLI maps
        # that to internal/exit 1 instead of usage/exit 2.
        with pytest.raises(ValueError, match="voice_settings\\['stability'\\] must be a number"):
            get_provider("elevenlabs").validate(elevenlabs_request(voice_settings={"stability": value}))

    def test_bool_speed_is_rejected(self):
        # float(True) == 1.0, which sits happily inside 0.7-1.2.
        with pytest.raises(ValueError, match="voice_settings\\['speed'\\] must be a number"):
            get_provider("elevenlabs").validate(elevenlabs_request(voice_settings={"speed": True}))

    def test_non_bool_speaker_boost_is_rejected(self):
        with pytest.raises(ValueError, match="voice_settings\\['use_speaker_boost'\\] must be a boolean"):
            get_provider("elevenlabs").validate(elevenlabs_request(voice_settings={"use_speaker_boost": "yes"}))

    @pytest.mark.parametrize("key", ["stability", "similarity_boost", "style"])
    def test_voice_settings_ranges(self, key):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            get_provider("elevenlabs").validate(elevenlabs_request(voice_settings={key: 1.5}))

    def test_cross_provider_model_rejected(self):
        with pytest.raises(ValueError, match="not an ElevenLabs model"):
            get_provider("elevenlabs").validate(elevenlabs_request(model="gpt-4o-mini-tts"))

    def test_instructions_warns_but_does_not_raise(self, caplog):
        # Speaker.instructions is a required field, so mixed-provider scripts
        # always carry one — this must never be fatal.
        get_provider("elevenlabs").validate(elevenlabs_request(instructions="Sound excited"))
        assert "OpenAI-only" in caplog.text


# ---------- synthesis ----------


@pytest.mark.integration
@pytest.mark.anyio
class TestElevenLabsSynthesis:
    async def test_accumulates_stream_chunks(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client(chunks=(b"AB", b"CD", b"EF"))
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        audio = await get_provider("elevenlabs").synthesize_chunk(elevenlabs_request())

        assert audio == b"ABCDEF"

    async def test_requests_mp3_and_passes_voice_settings(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await get_provider("elevenlabs").synthesize_chunk(
            elevenlabs_request(voice_settings={"stability": 0.4, "similarity_boost": 0.9})
        )

        call = client.text_to_speech.calls[0]
        assert call["voice_id"] == "voice_abc"
        assert call["model_id"] == "eleven_multilingual_v2"
        # mp3 keeps the podcast stitcher's AudioSegment.from_mp3 contract valid.
        assert call["output_format"] == "mp3_44100_128"
        assert call["voice_settings"].stability == 0.4
        assert call["voice_settings"].similarity_boost == 0.9

    async def test_no_voice_settings_when_nothing_to_override(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await get_provider("elevenlabs").synthesize_chunk(elevenlabs_request())

        # The sentinel, not None — None would serialize as an explicit null.
        assert client.text_to_speech.calls[0]["voice_settings"] is ...

    async def test_speed_maps_into_voice_settings(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client()
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        await get_provider("elevenlabs").synthesize_chunk(elevenlabs_request(speed=1.15))

        assert client.text_to_speech.calls[0]["voice_settings"].speed == 1.15

    async def test_empty_response_raises(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client(chunks=())
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        with pytest.raises(TTSAPIError, match="no audio"):
            await get_provider("elevenlabs").synthesize_chunk(elevenlabs_request())

    async def test_rate_limit_retries_then_wraps(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client(error=fake_elevenlabs.ApiError(429))
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)
        sleep = mocker.patch("sanzaru.audio.providers.elevenlabs_provider.anyio.sleep", mocker.AsyncMock())

        with pytest.raises(TTSAPIError, match="429"):
            await get_provider("elevenlabs").synthesize_chunk(elevenlabs_request())

        assert len(client.text_to_speech.calls) == 4  # _MAX_ATTEMPTS
        assert sleep.await_count == 3

    async def test_client_error_is_not_retried(self, mocker, fake_elevenlabs):
        client = fake_elevenlabs.Client(error=fake_elevenlabs.ApiError(401))
        mocker.patch("sanzaru.audio.providers.elevenlabs_provider.get_elevenlabs_client", return_value=client)

        with pytest.raises(TTSAPIError, match="ELEVENLABS_API_KEY"):
            await get_provider("elevenlabs").synthesize_chunk(elevenlabs_request())

        assert len(client.text_to_speech.calls) == 1


# ---------- chunk orchestration ----------


class StubProvider:
    """Minimal TTSProvider that echoes each chunk's text, for ordering tests."""

    name = "openai"

    def __init__(self, chunk_chars: int = 20):
        self.chunk_chars = chunk_chars
        self.seen: list[str] = []

    def resolve_model(self, model):
        return model or "stub"

    def resolve_voice(self, voice):
        return voice or "stub"

    def max_chunk_chars(self, model):
        return self.chunk_chars

    def max_concurrency(self, model):
        return 0

    def validate(self, request):
        return None

    async def synthesize_chunk(self, request):
        self.seen.append(request.text)
        return request.text.encode()


@pytest.mark.integration
@pytest.mark.anyio
class TestSynthesizeSpeech:
    async def test_single_chunk_skips_concatenation(self, mocker):
        concat = mocker.patch("sanzaru.audio.processor.AudioProcessor.concatenate_audio_segments")
        provider = StubProvider(chunk_chars=100)

        rendered = await synthesize_speech(provider, SpeechRequest(text="short text", voice="v", model="m"))

        assert rendered.audio == b"short text"
        concat.assert_not_called()
        assert rendered.usage.characters == len("short text")
        assert rendered.usage.requests == 1
        assert (rendered.usage.provider, rendered.usage.model) == (provider.name, "m")

    async def test_multi_chunk_preserves_order(self, mocker):
        # Concatenate by joining so the assertion reflects chunk order, not
        # completion order (chunks are rendered concurrently).
        async def fake_concat(audio_chunks, format="mp3"):
            return b"|".join(audio_chunks)

        mocker.patch(
            "sanzaru.audio.processor.AudioProcessor.concatenate_audio_segments",
            side_effect=fake_concat,
        )
        provider = StubProvider(chunk_chars=12)
        text = "Alpha one. Beta two. Gamma three. Delta four."

        rendered = await synthesize_speech(provider, SpeechRequest(text=text, voice="v", model="m"))
        audio = rendered.audio

        parts = audio.decode().split("|")
        assert len(parts) > 1
        assert "".join(parts).replace(" ", "") == text.replace(" ", "")

    async def test_validation_runs_before_any_request(self):
        provider = StubProvider()
        provider.validate = lambda request: (_ for _ in ()).throw(ValueError("nope"))

        with pytest.raises(ValueError, match="nope"):
            await synthesize_speech(provider, SpeechRequest(text="hi", voice="v", model="m"))

        assert provider.seen == []
