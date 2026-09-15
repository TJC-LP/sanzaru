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


LIVE_FRAME_BYTES = 24000 * 2 // 10
"""One 100ms PCM16/24kHz frame, the unit both the agent's clock and the fake speak in."""
LIVE_SILENT_FRAME = b"\x00" * LIVE_FRAME_BYTES
LIVE_LOUD_FRAME = (b"\x40\x1f\xc0\xe0" * (LIVE_FRAME_BYTES // 4))[:LIVE_FRAME_BYTES]
"""A +/-8000 square wave: RMS 8000, far above SPEECH_RMS_THRESHOLD."""


class FakeLiveConnection:
    """Stands in for AsyncLiveConnection, behaving the way the real server was measured to.

    The agent only touches `send()` and async iteration. Server events are
    scripted as reactions to what the agent sends, and — like the real API —
    nothing comes out until input audio comes in: `session.start` is answered
    with `session.started` (or an `error`); every append is acknowledged with
    its `*.appended`; the turn cue is *armed* and only plays once the next
    `session.input_audio.append` arrives; every idle input frame is answered
    with one zero output frame, so the output is a continuous stream that
    includes silence; `session.close` is answered with `session.closed`.
    Iteration blocks on a memory stream rather than ending when the script runs
    out, because the real socket does too.
    """

    def __init__(
        self,
        *,
        seconds: float | Sequence[float] = 1.0,
        transcripts: list[str] | None = None,
        usage_seconds: Sequence[float] = (),
        final_usage_seconds: float | None = None,
        lead_silence_s: float = 0.2,
        trail_silence_s: float = 0.3,
        start_error: str | None = None,
        pre_cue_seconds: float = 0.0,
        silent: bool = False,
        close_mid_turn: bool = False,
        end_stream_mid_turn: bool = False,
        lose_injections_at_close: bool = False,
        ack_appends: bool = True,
        timeline: Sequence[tuple[float, float]] = (),
        reply_seconds: float = 0.0,
        reply_after_silence_frames: int = 3,
        max_replies: int = 1,
    ) -> None:
        import anyio

        self.sent: list[dict[str, object]] = []
        self.sent_at: list[float] = []
        # ---- duplex scripting: what this host says on the input-frame clock ----
        # `timeline` is (start_s, duration_s) pairs of speech; `reply_seconds`
        # makes the host answer for that long once it has heard speech followed
        # by `reply_after_silence_frames` quiet frames — "speaks when the other
        # stops". Either makes the fake ignore cues and drive itself.
        self.timeline = list(timeline)
        self.reply_seconds = reply_seconds
        self.reply_after_silence_frames = reply_after_silence_frames
        self.max_replies = max_replies
        self._scripted = bool(self.timeline) or reply_seconds > 0
        self._speaking = False
        self._utterance_start_tick = 0
        self._utterances = 0
        self._heard_loud = False
        self._quiet_input_frames = 0
        self._reply_until_tick = -1
        self._replies = 0
        self.seconds = seconds
        self.transcripts = list(transcripts or [])
        self.usage_seconds = list(usage_seconds)
        self.final_usage_seconds = final_usage_seconds
        self.lead_silence_s = lead_silence_s
        self.trail_silence_s = trail_silence_s
        self.start_error = start_error
        # Speech the model produces right after the first input frame, before
        # any cue: what a full-duplex model talking out of turn looks like.
        self.pre_cue_seconds = pre_cue_seconds
        self.silent = silent
        self.close_mid_turn = close_mid_turn
        self.end_stream_mid_turn = end_stream_mid_turn
        self.lose_injections_at_close = lose_injections_at_close
        self.ack_appends = ack_appends
        self.turn = 0
        self.closed = False
        self.input_frames = 0
        self.heard_speech_bytes = 0
        """Non-silent input bytes: what this host actually heard someone say."""
        self._cue_armed = False
        self._pre_cue_pending = pre_cue_seconds > 0
        self._unacked: list[str] = []
        send_stream, receive_stream = anyio.create_memory_object_stream[FakeEvent](max_buffer_size=1_000_000)
        self._events = send_stream
        self._incoming = receive_stream

    # ---- what the agent sent, by type ----

    def sent_of(self, event_type: str) -> list[dict[str, object]]:
        return [event for event in self.sent if event.get("type") == event_type]

    @property
    def input_bytes(self) -> int:
        return self.input_frames * LIVE_FRAME_BYTES

    # ---- the connection surface ----

    async def send(self, event: dict[str, object]) -> None:
        if self.closed:
            raise RuntimeError("send on a closed fake live connection")
        import time

        self.sent.append(event)
        self.sent_at.append(time.monotonic())
        event_type = event.get("type")
        if event_type == "session.start":
            if self.start_error is not None:
                self._emit(FakeEvent("error", error=FakeEvent("err", code="bad", message=self.start_error)))
                return
            self._emit(FakeEvent("session.started", session=FakeEvent("session", id="sess_1")))
        elif event_type in ("session.instructions.append", "session.commentary.append", "session.thinking.append"):
            from sanzaru.audio.realtime.live_agent import TURN_NUDGE

            event_id = str(event.get("event_id") or "")
            if self.ack_appends and not self.lose_injections_at_close:
                self._emit(FakeEvent(f"{event_type}ed", client_event_id=event_id, start_ms=0, end_ms=0))
            else:
                self._unacked.append(event_id)
            # The nudge is the last frame of a cue, so answering it is the
            # earliest a real server could start the turn.
            if event.get("content") == TURN_NUDGE:
                self._cue_armed = True
        elif event_type == "session.input_audio.append":
            import base64

            pcm = base64.b64decode(str(event["audio"]))
            self.input_frames += 1
            loud = any(pcm)
            if loud:
                self.heard_speech_bytes += len(pcm)
            if self._scripted:
                self._on_scripted_frame(loud)
            else:
                self._on_input_frame()
        elif event_type == "session.close":
            if self._scripted and self._speaking:
                self._end_utterance(self.input_frames)
            for event_id in self._unacked:
                self._emit(
                    FakeEvent(
                        "error",
                        error=FakeEvent(
                            "err",
                            code="context_injection_incomplete",
                            message="The session closed before the estimated context injection completed.",
                            client_event_id=event_id,
                        ),
                    )
                )
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

    def _emit_frames(self, seconds: float, frame: bytes) -> None:
        import base64

        for _ in range(round(seconds * 10)):
            self._emit(FakeEvent("session.output_audio.delta", delta=base64.b64encode(frame).decode()))

    def _on_input_frame(self) -> None:
        """The timeline moved: play whatever is due, else one silent frame."""
        if self._pre_cue_pending:
            self._pre_cue_pending = False
            self._emit_frames(self.pre_cue_seconds, LIVE_LOUD_FRAME)
            return
        if self._cue_armed:
            self._cue_armed = False
            self._script_turn()
            return
        self._emit_frames(0.1, LIVE_SILENT_FRAME)

    def _on_scripted_frame(self, heard_loud: bool) -> None:
        """Duplex script: one output frame per input frame, speech or silence by the timeline."""
        tick = self.input_frames - 1  # 0-based, 100ms each
        t = tick / 10
        if heard_loud:
            self._heard_loud = True
            self._quiet_input_frames = 0
        else:
            self._quiet_input_frames += 1
        if (
            self.reply_seconds > 0
            and self._heard_loud
            and self._quiet_input_frames >= self.reply_after_silence_frames
            and self._replies < self.max_replies
            and not self._speaking
        ):
            self._reply_until_tick = tick + round(self.reply_seconds * 10)
            self._replies += 1
            self._heard_loud = False
        speaking = (
            any(start <= t < start + duration for start, duration in self.timeline) or tick < self._reply_until_tick
        )
        if speaking and not self._speaking:
            self._utterance_start_tick = tick
        if not speaking and self._speaking:
            self._end_utterance(tick)
        self._speaking = speaking
        self._emit_frames(0.1, LIVE_LOUD_FRAME if speaking else LIVE_SILENT_FRAME)

    def _end_utterance(self, end_tick: int) -> None:
        index = self._utterances
        self._utterances += 1
        text = self.transcripts[index] if index < len(self.transcripts) else f"utterance {index}"
        self._emit(
            FakeEvent(
                "session.output_transcript.delta",
                delta=text,
                start_ms=self._utterance_start_tick * 100,
                end_ms=end_tick * 100,
            )
        )

    def _script_turn(self) -> None:
        index = self.turn
        self.turn += 1
        if self.silent:
            self._emit_frames(0.1, LIVE_SILENT_FRAME)
            return
        if isinstance(self.seconds, (int, float)):
            turn_seconds = float(self.seconds)
        else:
            durations = list(self.seconds) or [1.0]
            turn_seconds = float(durations[min(index, len(durations) - 1)])
        if self.close_mid_turn or self.end_stream_mid_turn:
            self._emit_frames(0.1, LIVE_LOUD_FRAME)
            if self.close_mid_turn:
                self._emit(FakeEvent("session.closed", reason="expired", usage=FakeEvent("usage", seconds=0.0)))
            self._end_stream()
            return
        self._emit_frames(self.lead_silence_s, LIVE_SILENT_FRAME)
        self._emit_frames(turn_seconds, LIVE_LOUD_FRAME)
        text = self.transcripts[index] if index < len(self.transcripts) else f"turn {index}"
        for word in text.split(" "):
            self._emit(FakeEvent("session.output_transcript.delta", delta=word + " ", start_ms=0, end_ms=0))
        if index < len(self.usage_seconds):
            # Before the trailing silence: the agent ends the turn on the first
            # sub-threshold frame, and this is what makes a turn's usage delta
            # deterministic in tests (the real report lags; wall clock covers).
            self._emit(FakeEvent("session.usage.updated", usage=FakeEvent("usage", seconds=self.usage_seconds[index])))
        self._emit_frames(self.trail_silence_s, LIVE_SILENT_FRAME)


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
