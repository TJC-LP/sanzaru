"""Shared fakes for audio/TTS tests.

The ElevenLabs provider only ever calls `client.text_to_speech.convert(**kwargs)`
and iterates the result, so these fakes need no SDK import — every test using
them runs without the optional [elevenlabs] extra installed.
"""

import os
from collections.abc import Sequence

import pytest


@pytest.fixture(autouse=True)
def clean_realtime_env(monkeypatch):
    """Start every test from an empty SANZARU_REALTIME_* environment.

    These variables (turn timeout, session cap, price overrides) are read
    straight from `os.environ` by the code under test, so a developer who set
    one to debug a stalled session — or who runs the documented
    `dotenv-cli -- pytest` workflow with them in `.env` — would watch the tests
    that pin their defaults fail for reasons that have nothing to do with them.
    """
    for name in [key for key in os.environ if key.startswith("SANZARU_REALTIME_")]:
        monkeypatch.delenv(name, raising=False)


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
        seconds: float | Sequence[float] = 2.0,
        transcripts: list[str] | None = None,
        marker: bytes = b"\x01\x02",
        error: str | None = None,
        status: str = "completed",
        usage: FakeEvent | None = None,
        end_early: bool = False,
        hang: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        # A sequence gives each turn its own length (the last value repeats), so
        # a test can make a closing turn shorter than the running average — the
        # only way to reach stop_reason="complete" with turns of one length.
        self.seconds = seconds
        self.transcripts = list(transcripts or [])
        self.marker = marker
        self.error = error
        self.status = status
        self.usage = usage
        # The SDK's __aiter__ returns cleanly on ConnectionClosedOK, so a
        # graceful mid-turn close looks exactly like a stream that just ran out.
        self.end_early = end_early
        # A session that accepted response.create and then went quiet.
        self.hang = hang
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
        if isinstance(self.seconds, (int, float)):
            turn_seconds = float(self.seconds)
        else:
            durations = list(self.seconds) or [2.0]
            turn_seconds = float(durations[min(self.turn, len(durations) - 1)])
        frames = int(turn_seconds * 24000)
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
        if self.end_early:
            self._pending.pop()

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        if self.hang:
            import anyio

            await anyio.sleep_forever()
        while self._pending:
            yield self._pending.pop(0)


class FakeLiveConnection:
    """Stands in for AsyncLiveConnection.

    The agent only touches `send()` and async iteration. Server events are
    scripted as reactions to what the agent sends: `session.start` is answered
    with `session.started` (or an `error`), the turn cue with one turn's worth
    of audio/transcript deltas and a cumulative usage update, `session.close`
    with `session.closed`. Iteration blocks on a memory stream rather than
    ending when the script runs out, because the real socket does too.
    """

    def __init__(
        self,
        *,
        seconds: float | Sequence[float] = 1.0,
        transcripts: list[str] | None = None,
        usage_seconds: Sequence[float] = (),
        final_usage_seconds: float | None = None,
        marker: bytes = b"\x01\x02",
        delta_seconds: float = 0.5,
        start_error: str | None = None,
        pre_cue_seconds: float = 0.0,
        silent: bool = False,
        close_mid_turn: bool = False,
        end_stream_mid_turn: bool = False,
    ) -> None:
        import anyio

        self.sent: list[dict[str, object]] = []
        self.seconds = seconds
        self.transcripts = list(transcripts or [])
        self.usage_seconds = list(usage_seconds)
        self.final_usage_seconds = final_usage_seconds
        self.marker = marker
        self.delta_seconds = delta_seconds
        self.start_error = start_error
        # Audio the model produces right after starting, before any cue: what a
        # full-duplex model talking out of turn looks like.
        self.pre_cue_seconds = pre_cue_seconds
        self.silent = silent
        self.close_mid_turn = close_mid_turn
        self.end_stream_mid_turn = end_stream_mid_turn
        self.turn = 0
        self.closed = False
        send_stream, receive_stream = anyio.create_memory_object_stream[FakeEvent](max_buffer_size=1_000_000)
        self._events = send_stream
        self._incoming = receive_stream

    # ---- what the agent sent, by type ----

    def sent_of(self, event_type: str) -> list[dict[str, object]]:
        return [event for event in self.sent if event.get("type") == event_type]

    @property
    def heard_bytes(self) -> int:
        import base64

        return sum(len(base64.b64decode(str(e["audio"]))) for e in self.sent_of("session.input_audio.append"))

    # ---- the connection surface ----

    async def send(self, event: dict[str, object]) -> None:
        if self.closed:
            raise RuntimeError("send on a closed fake live connection")
        self.sent.append(event)
        event_type = event.get("type")
        if event_type == "session.start":
            if self.start_error is not None:
                self._emit(FakeEvent("error", error=FakeEvent("err", code="bad", message=self.start_error)))
                return
            self._emit(FakeEvent("session.started", session=FakeEvent("session", id="sess_1")))
            if self.pre_cue_seconds:
                self._emit_audio(self.pre_cue_seconds)
        elif event_type == "session.commentary.append":
            from sanzaru.audio.realtime.live_agent import TURN_NUDGE

            # The nudge is the last frame of a cue, so answering it is the
            # earliest a real server could start the turn.
            if event.get("content") == TURN_NUDGE:
                self._script_turn()
        elif event_type == "session.close":
            usage = self.final_usage_seconds
            if usage is None:
                usage = self.usage_seconds[-1] if self.usage_seconds else 0.0
            self._emit(FakeEvent("session.closed", reason="close_requested", usage=FakeEvent("usage", seconds=usage)))
            self._end_stream()

    async def close(self) -> None:
        self._end_stream()

    def emit_error(self, message: str) -> None:
        """A server-side error arriving between turns."""
        self._emit(FakeEvent("error", error=FakeEvent("err", code="server_error", message=message)))

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        import anyio

        try:
            async for event in self._incoming:
                yield event
        except anyio.ClosedResourceError:
            return

    # ---- scripting ----

    def _emit(self, event: FakeEvent) -> None:
        if not self.closed:
            self._events.send_nowait(event)

    def _end_stream(self) -> None:
        if not self.closed:
            self.closed = True
            self._events.close()

    def _emit_audio(self, seconds: float) -> None:
        import base64

        remaining = seconds
        while remaining > 1e-9:
            chunk = min(self.delta_seconds, remaining)
            frames = int(chunk * 24000)
            pcm = (self.marker * frames)[: frames * 2]
            self._emit(FakeEvent("session.output_audio.delta", delta=base64.b64encode(pcm).decode()))
            remaining -= chunk

    def _script_turn(self) -> None:
        index = self.turn
        self.turn += 1
        if self.silent:
            return
        if isinstance(self.seconds, (int, float)):
            turn_seconds = float(self.seconds)
        else:
            durations = list(self.seconds) or [1.0]
            turn_seconds = float(durations[min(index, len(durations) - 1)])
        if self.close_mid_turn or self.end_stream_mid_turn:
            self._emit_audio(min(self.delta_seconds, turn_seconds))
            if self.close_mid_turn:
                self._emit(FakeEvent("session.closed", reason="expired", usage=FakeEvent("usage", seconds=0.0)))
            self._end_stream()
            return
        self._emit_audio(turn_seconds)
        text = self.transcripts[index] if index < len(self.transcripts) else f"turn {index}"
        for word in text.split(" "):
            self._emit(FakeEvent("session.output_transcript.delta", delta=word + " ", start_ms=0, end_ms=0))
        if index < len(self.usage_seconds):
            self._emit(FakeEvent("session.usage.updated", usage=FakeEvent("usage", seconds=self.usage_seconds[index])))


class _FakeRealtime:
    """Namespace of realtime test doubles (see _FakeElevenLabs for why)."""

    Connection = FakeRealtimeConnection
    Event = FakeEvent
    usage = staticmethod(fake_usage)


@pytest.fixture
def fake_realtime():
    """Realtime test doubles: `.Connection`, `.Event`, `.usage()`."""
    return _FakeRealtime


class _FakeLive:
    """Namespace of Live API test doubles."""

    Connection = FakeLiveConnection
    Event = FakeEvent


@pytest.fixture
def fake_live():
    """Live API test doubles: `.Connection`, `.Event`."""
    return _FakeLive


@pytest.fixture
def connect_factory():
    """Build a `connect` factory over a list of prepared connections.

    Returns (factory, connections): the factory hands out one connection per
    call, in order, so a two-host act gets connections[0] and connections[1].
    """
    import contextlib

    def build(*connections: FakeRealtimeConnection | FakeLiveConnection):
        handed: list[FakeRealtimeConnection | FakeLiveConnection] = []

        @contextlib.asynccontextmanager
        async def factory(model: str):
            conn = connections[len(handed)]
            handed.append(conn)
            yield conn

        return factory, handed

    return build
