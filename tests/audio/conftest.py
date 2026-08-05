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


# ---------- podcast scripts ----------


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
            "normalize_loudness": True,
            "output_format": "mp3",
        },
    }


# ---------- realtime simulation ----------


class FakeEvent:
    """A Realtime server event. Only `.type` plus whatever the handler reads."""

    def __init__(self, type: str, **fields: object) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


def fake_usage(audio_out: int = 100, text_out: int = 40, audio_in: int = 200, cached: int = 120) -> FakeEvent:
    """Mirrors RealtimeResponseUsage's nesting, which the agent flattens."""
    return FakeEvent(
        "usage",
        input_tokens=audio_in + 50,
        output_tokens=audio_out + text_out,
        input_token_details=FakeEvent(
            "in",
            audio_tokens=audio_in,
            text_tokens=50,
            cached_tokens=cached,
            cached_tokens_details=FakeEvent("cached", audio_tokens=cached, text_tokens=10),
        ),
        output_token_details=FakeEvent("out", audio_tokens=audio_out, text_tokens=text_out),
    )


class _Resource:
    """Records every call as (name, kwargs) on the shared connection log."""

    def __init__(self, log: list[tuple[str, dict[str, object]]], prefix: str) -> None:
        self._log = log
        self._prefix = prefix

    def _record(self, name: str, **kwargs: object) -> None:
        self._log.append((f"{self._prefix}{name}", kwargs))


class _SessionResource(_Resource):
    async def update(self, *, session: dict[str, object]) -> None:
        self._record("update", session=session)


class _ResponseResource(_Resource):
    def __init__(self, log, prefix, on_create) -> None:  # type: ignore[no-untyped-def]
        super().__init__(log, prefix)
        self._on_create = on_create

    async def create(self) -> None:
        self._record("create")
        self._on_create()


class _BufferResource(_Resource):
    async def append(self, *, audio: str) -> None:
        self._record("append", bytes=len(audio))

    async def commit(self) -> None:
        self._record("commit")


class _ItemResource(_Resource):
    async def create(self, *, item: dict[str, object]) -> None:
        self._record("create", item=item)


class _ConversationResource(_Resource):
    def __init__(self, log, prefix) -> None:  # type: ignore[no-untyped-def]
        super().__init__(log, prefix)
        self.item = _ItemResource(log, "conversation.item.")


class FakeRealtimeConnection:
    """Stands in for AsyncRealtimeConnection.

    The agent only touches session.update, response.create, input_audio_buffer
    append/commit, conversation.item.create, and async iteration — so no SDK and
    no websocket are needed to exercise floor control end to end.
    """

    def __init__(
        self,
        *,
        seconds: float = 2.0,
        transcripts: list[str] | None = None,
        marker: bytes = b"\x01\x02",
        error: str | None = None,
        status: str = "completed",
        usage: FakeEvent | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.seconds = seconds
        self.transcripts = list(transcripts or [])
        self.marker = marker
        self.error = error
        self.status = status
        self.usage = usage
        self.turn = 0
        self._pending: list[FakeEvent] = []
        self.session = _SessionResource(self.calls, "session.")
        self.response = _ResponseResource(self.calls, "response.", self._queue_response)
        self.input_audio_buffer = _BufferResource(self.calls, "input_audio_buffer.")
        self.conversation = _ConversationResource(self.calls, "conversation.")

    @property
    def heard_bytes(self) -> int:
        """Total audio this connection was fed by the producer."""
        total = 0
        for name, kwargs in self.calls:
            size = kwargs.get("bytes")
            if name == "input_audio_buffer.append" and isinstance(size, int):
                total += size
        return total

    @property
    def steers(self) -> list[str]:
        """Producer notes this connection received, in order."""
        notes: list[str] = []
        for name, kwargs in self.calls:
            if name == "conversation.item.create":
                item = kwargs["item"]
                notes.append(item["content"][0]["text"])  # type: ignore[index]
        return notes

    def _queue_response(self) -> None:
        if self.error is not None:
            self._pending = [FakeEvent("error", error=self.error)]
            return
        # PCM16/24kHz mono, so bytes = seconds * 24000 * 2.
        frames = int(self.seconds * 24000)
        pcm = (self.marker * frames)[: frames * 2]
        text = self.transcripts[self.turn] if self.turn < len(self.transcripts) else f"turn {self.turn}"
        self.turn += 1
        import base64

        self._pending = [
            FakeEvent("response.output_audio.delta", delta=base64.b64encode(pcm).decode()),
            FakeEvent("response.output_audio_transcript.done", transcript=text),
            FakeEvent(
                "response.done",
                response=FakeEvent(
                    "resp",
                    usage=self.usage or fake_usage(),
                    status=self.status,
                    status_details=FakeEvent("details", reason="max_output_tokens", error="boom"),
                ),
            ),
        ]

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        while self._pending:
            yield self._pending.pop(0)


class _FakeRealtime:
    """Namespace of realtime test doubles (see _FakeElevenLabs for why)."""

    Connection = FakeRealtimeConnection
    Event = FakeEvent
    usage = staticmethod(fake_usage)


@pytest.fixture
def fake_realtime():
    """Realtime test doubles: `.Connection`, `.Event`, `.usage()`."""
    return _FakeRealtime


@pytest.fixture
def connect_factory():
    """Build a `connect` factory over a list of prepared connections.

    Returns (factory, connections): the factory hands out one connection per
    call, in order, so a two-host act gets connections[0] and connections[1].
    """
    import contextlib

    def build(*connections: FakeRealtimeConnection):
        handed: list[FakeRealtimeConnection] = []

        @contextlib.asynccontextmanager
        async def factory(model: str):
            conn = connections[len(handed)]
            handed.append(conn)
            yield conn

        return factory, handed

    return build
