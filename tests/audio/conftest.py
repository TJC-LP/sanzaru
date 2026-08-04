"""Shared fakes for audio/TTS tests.

The ElevenLabs provider only ever calls `client.text_to_speech.convert(**kwargs)`
and iterates the result, so these fakes need no SDK import — every test using
them runs without the optional [elevenlabs] extra installed.
"""

import pytest


class FakeElevenLabsTTS:
    """Stand-in for client.text_to_speech, recording every convert() call."""

    def __init__(self, chunks=(b"ID3", b"FAKE"), error=None, on_call=None):
        self.calls: list[dict[str, object]] = []
        self.chunks = chunks
        self.error = error
        self.on_call = on_call

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        error = self.error
        on_call = self.on_call
        chunks = self.chunks

        async def _stream():
            if error is not None:
                raise error
            if on_call is not None:
                await on_call()
            for chunk in chunks:
                yield chunk

        return _stream()


class FakeElevenLabsClient:
    def __init__(self, chunks=(b"ID3", b"FAKE"), error=None, on_call=None):
        self.text_to_speech = FakeElevenLabsTTS(chunks=chunks, error=error, on_call=on_call)


class FakeApiError(Exception):
    """Mimics elevenlabs.core.ApiError, which carries a status_code."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeElevenLabs:
    """Namespace of ElevenLabs test doubles.

    Delivered as a fixture rather than imported directly: test directories have
    no __init__.py, so cross-module imports between test files don't resolve.
    """

    Client = FakeElevenLabsClient
    TTS = FakeElevenLabsTTS
    ApiError = FakeApiError


@pytest.fixture
def fake_elevenlabs():
    """ElevenLabs test doubles: `.Client`, `.TTS`, `.ApiError`."""
    return _FakeElevenLabs
