# SPDX-License-Identifier: MIT
"""One conversational agent on one Live API connection (`gpt-live-*`).

Same public surface as `RealtimeAgent` — `configure`, `steer`, `hear`, `speak` —
so `producer.run_act` can seat either at the table, but the model underneath is
a different animal. Four facts about it, all measured against the real API on
2026-09-15 (see the docs' "gpt-live-1" section), shape everything here:

- **The session timeline only advances while input audio is streaming.** With
  nothing on `session.input_audio.append`, the model never speaks, appended
  instructions are never applied, and at close the server returns one
  `context_injection_incomplete` error per pending append. So every agent runs
  a *clock task* that streams 100ms PCM16 frames at real-time pace for the whole
  session — silence when there is nothing to hear, otherwise frames from the
  inbox that `hear()` / fan-out fill. An act therefore runs in real time.
- **It is full duplex.** There is no `response.create` for the model's own
  speech and no `response.done` after it. The producer's floor control is
  *advisory*: the session instructions tell the model to speak only when cued,
  `speak()` sends the cue, and speech the model produces while it does *not*
  hold the floor is discarded (counted and logged as off-floor seconds).
- **Output is a continuous frame stream that includes silence**: ~10 deltas a
  second whether or not the model is talking, exact-zero frames between and
  after speech. "No delta for N seconds" never fires. A turn is therefore
  bounded by *loudness*: it starts at the first frame whose RMS clears
  `SPEECH_RMS_THRESHOLD`, ends after `end_of_turn_silence_s` of frames below it
  (or at `2 x turn_seconds` of speech, reported `truncated`), and the returned
  PCM is trimmed to the speech.
- **`usage.seconds` meters streamed session time**, not spoken audio, and it is
  cumulative. Each `speak()` reports the delta since the previous turn as
  `RealtimeUsage.live_seconds`; `finish()` reports what the close settles on
  top. Listening bills like talking, so every host's session costs the act.

Live agents in one act hear each other *as they speak*: an agent forwards its
output frames into its listeners' inboxes (`set_listeners`), where their clocks
mix them per frame and play them out at pace. In the *cued* loop that happens
only while the agent holds the floor; in *duplex* mode (`always=True`, see
`duplex.py`) it happens for the whole act, and the models negotiate turn-taking
themselves — which is what the model is for. Producer notes go out on
`session.thinking.append`: `session.instructions.append` was spoken aloud in a
real recording ("Now Gus has wrapped, so you can land the through-line"), while a
deliberately line-shaped thinking note was followed and never voiced.

Every `openai.types.live` import is TYPE_CHECKING-only: the CLI startup-weight
test forbids pulling `openai` in at `sanzaru.cli` import time.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, cast

import anyio
from anyio.abc import TaskGroup

from ...config import logger
from ...exceptions import RealtimeAPIError
from .agent import SpokenTurn
from .types import HostSpec, LiveMode, RealtimeUsage

if TYPE_CHECKING:
    from openai.resources.live.live import AsyncLiveConnection
    from openai.types.live.client_event_param import ClientEventParam


FRAME_MS = 100
"""Input clock period and frame length. The API's own output deltas arrive at
about this cadence, so listeners' inboxes neither starve nor pile up."""

SPEECH_RMS_THRESHOLD = 300
"""PCM16 RMS above which an output frame counts as speech. Measured on
gpt-live-1: speech frames land at 500-3500, silence frames are exact zeros with
a handful of 1-200 frames on the edges of words. 300 sits in the gap."""

DEFAULT_END_OF_TURN_SILENCE_S = 1.2
"""How much sub-threshold audio, after speech, ends the turn. Below ~1s a
mid-sentence breath ends it; well above it every turn carries that much dead
air before the next host is cued."""

START_WAIT_FACTOR = 3.0
START_WAIT_FLOOR_S = 6.0
"""How long `speak()` waits for the first *speech* frame after cueing:
`START_WAIT_FACTOR` silence gaps, never under `START_WAIT_FLOOR_S`. Measured
first speech was 0.6s after the cue; the floor covers a model that first
finishes hearing the turn it was just played."""

TURN_AUDIO_CAP_FACTOR = 2.0
"""A turn is cut at this many `turn_seconds` of speech. There is no token cap on
a Live turn, so this is the only mechanical bound on a monologue; the prompt's
length rule is what actually shapes turns."""

ACK_WAIT_S = 2.0
"""Bound on waiting for `*.appended` after a cue. Acks were measured ~1.1s
behind the cue — *after* the first speech frame — so this never gates speech
detection, which only reads state the reader already collected."""

SESSION_START_TIMEOUT_S = 30.0
SESSION_CLOSE_TIMEOUT_S = 10.0

STEER_MAX_CHARS = 1500
"""`session.instructions.append` takes at most 500 tokens. A note is a sentence
or two, so this only ever trims a runaway caller `turn_note`."""

TURN_CUE = (
    "PRODUCER: it is your turn now. Respond to what you just heard, make your point, then stop and stay "
    "silent until your next cue."
)
TURN_NUDGE = "Your co-host has finished. Go ahead."
"""The cue goes out as an instruction; the nudge as *commentary*, the channel the
Live prompting guide uses to make the model open its mouth rather than wait to
be spoken to. Commentary is speakable context, so it is kept to something the
model could echo without damage."""

STOP_CUE = "PRODUCER: stop talking now. Say nothing more until your next cue."

CUED_SETTLE_S = 1.0
"""Cued loop: pause between the previous turn's fan-out draining and the next
cue. A host cued the instant its co-host stopped answered the *previous*
exchange — verbatim repeats two turns apart — because it had not absorbed the
one it just heard."""

FALSE_START_MIN_VOICE_S = 0.5
FALSE_START_MAX_TRIM_S = 1.0
"""Cued loop: a full-duplex model starts reacting to what it hears ("but", "So")
and restarts on the cue. Speech at the head of a turn before the first
`FALSE_START_MIN_VOICE_S` of sustained voice is trimmed, never more than
`FALSE_START_MAX_TRIM_S` of it, so a genuine short opener survives."""

SPEAKING_HOLD_S = 0.3
"""How long after its last speech frame a host still counts as speaking, so a
breath between words is not read as a yield."""

INJECTION_INCOMPLETE = "context_injection_incomplete"
"""The error code the server returns, at close, for every append it had not
finished applying. A warning naming the lost note, never a fault: the audio
is already recorded by then."""


def frame_rms(pcm: bytes) -> float:
    """RMS of a PCM16 mono buffer; 0.0 for anything shorter than one sample."""
    if len(pcm) < 2:
        return 0.0
    try:
        import audioop  # audioop-lts on 3.13+, part of the [audio] extra
    except ImportError:  # pragma: no cover - the extra is required for this feature
        from array import array

        samples = array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        return (sum(s * s for s in samples) / len(samples)) ** 0.5
    return float(audioop.rms(pcm[: len(pcm) - len(pcm) % 2], 2))


def mix_pcm16(frames: Sequence[bytes]) -> bytes:
    """Sum PCM16 buffers sample by sample, clipping, padded to the longest."""
    frames = [f for f in frames if f]
    if not frames:
        return b""
    if len(frames) == 1:
        return frames[0]
    length = max(len(f) for f in frames)
    length -= length % 2
    try:
        import audioop

        mixed = frames[0][:length].ljust(length, b"\x00")
        for other in frames[1:]:
            mixed = audioop.add(mixed, other[:length].ljust(length, b"\x00"), 2)
        return mixed
    except ImportError:  # pragma: no cover - the extra is required for this feature
        from array import array

        total = array("h", bytes(length))
        for other in frames:
            samples = array("h")
            samples.frombytes(other[:length].ljust(length, b"\x00"))
            for index, value in enumerate(samples):
                total[index] = max(-32768, min(32767, total[index] + value))
        return total.tobytes()


def trim_false_start(pcm: bytes, bytes_per_second: int, frame_bytes: int) -> bytes:
    """Drop a false start at the head of a turn.

    Scans `frame_bytes` windows from the start for the first run of
    `FALSE_START_MIN_VOICE_S` consecutive speech frames and cuts everything
    before it — but only when that run begins within `FALSE_START_MAX_TRIM_S`.
    A turn that never settles into sustained voice is returned unchanged.
    """
    if frame_bytes <= 0 or len(pcm) < frame_bytes:
        return pcm
    need = max(1, round(FALSE_START_MIN_VOICE_S * bytes_per_second / frame_bytes))
    limit = int(FALSE_START_MAX_TRIM_S * bytes_per_second)
    run = 0
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        if frame_rms(pcm[offset : offset + frame_bytes]) >= SPEECH_RMS_THRESHOLD:
            run += 1
            if run >= need:
                start = offset - (need - 1) * frame_bytes
                return pcm[start:] if 0 < start <= limit else pcm
        else:
            run = 0
        if offset > limit + need * frame_bytes:
            break
    return pcm


def _turn_taking_rules(turn_seconds: float) -> str:
    return "\n".join(
        [
            "",
            "TURN-TAKING (this is a recorded conversation run by a producer you never mention):",
            "  - You will HEAR your co-hosts speak. While they speak, stay completely silent: no reactions, "
            "no back-channel, no interjections.",
            "  - Speak ONLY after the producer tells you it is your turn. Then respond to what you just heard, "
            f"make your point in under {turn_seconds:.0f} seconds, and stop.",
            "  - When you have finished your point, stop talking and wait. Silence is expected. Do not fill it.",
            "  - Never announce that it is your turn, never speak the producer's notes aloud, and never say "
            "that you are waiting.",
        ]
    )


def _describe_error(error: object) -> str:
    """`code: message (param)` from a Live `Error`, or its str for anything else."""
    message = getattr(error, "message", None)
    if not isinstance(message, str):
        return str(error)
    code = getattr(error, "code", None)
    param = getattr(error, "param", None)
    text = f"{code}: {message}" if isinstance(code, str) and code else message
    return f"{text} ({param})" if isinstance(param, str) and param else text


class LiveAgent:
    """A persona bound to a Live API connection.

    Enter it (`async with agent:`) before `configure()`; the reader and clock
    tasks live for exactly that scope. Between `configure()` and `finish()` it
    behaves like `RealtimeAgent`.
    """

    def __init__(
        self,
        spec: HostSpec,
        connection: AsyncLiveConnection,
        *,
        model: str,
        turn_seconds: float,
        sample_rate: int,
        end_of_turn_silence_s: float = DEFAULT_END_OF_TURN_SILENCE_S,
        start_wait_s: float | None = None,
        settle_s: float = CUED_SETTLE_S,
        mode: LiveMode = "cued",
    ) -> None:
        if end_of_turn_silence_s <= 0:
            raise ValueError("end_of_turn_silence_s must be positive")
        self.spec = spec
        self.model = model
        self.mode: LiveMode = mode
        self._conn = connection
        self._turn_seconds = turn_seconds
        self._sample_rate = sample_rate
        self._bytes_per_second = sample_rate * 2  # PCM16 mono
        self._frame_bytes = self._bytes_per_second * FRAME_MS // 1000
        self._silence_s = end_of_turn_silence_s
        self._start_wait_s = (
            start_wait_s
            if start_wait_s is not None
            else max(START_WAIT_FLOOR_S, START_WAIT_FACTOR * end_of_turn_silence_s)
        )

        self._tg: TaskGroup | None = None
        self._changed = anyio.Event()
        self._started = False
        self._started_at: float | None = None
        self._closed = False
        self._close_reason: str | None = None
        self._fault: RealtimeAPIError | None = None
        self._finished = False

        # ---- ears: the input clock and its inboxes ----
        self._clock_running = False
        self._clock_started_at: float | None = None
        self._inboxes: dict[str, bytearray] = {}
        """One queue per source; the clock mixes a frame from each per tick, so
        two hosts talking at once reach a third as overlap, not one after the
        other."""
        self._frames_sent = 0
        self._listeners: list[LiveAgent] = []
        self._forward_always = False

        # ---- mouth: the turn being collected ----
        self._on_floor = False
        self._turn_pcm = bytearray()
        self._turn_text: list[str] = []
        self._speech_start: int | None = None
        """Offset in `_turn_pcm` of the first speech frame this turn."""
        self._speech_end = 0
        """Offset just past the last speech frame this turn."""
        self._frame_events = 0
        self._speech_events = 0
        self._off_floor_bytes = 0
        self._off_floor_reported = 0

        # ---- the whole-act view a duplex act needs ----
        self._ever_spoke = False
        self._last_speech_at: float | None = None
        self._speech_onset_at: float | None = None
        self._recording = False
        self._record_t0 = 0.0
        self._stream = bytearray()
        self.transcript_deltas: list[tuple[int, int, str]] = []
        """(start_ms, end_ms, text) on the session timeline, every fragment the
        model reported, whoever held the floor."""

        # ---- injections awaiting their `*.appended` ----
        self._pending: dict[str, str] = {}
        self._event_counter = 0
        self._settle_s = settle_s
        self._ever_heard_a_turn = False

        # ---- billing ----
        self._reported_seconds = 0.0
        self._billed_seconds = 0.0

    # ---------- identity ----------

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def off_floor_seconds(self) -> float:
        """Speech the model produced while another host held the floor, discarded."""
        return self._off_floor_bytes / self._bytes_per_second

    @property
    def frames_sent(self) -> int:
        """Input frames the clock has streamed so far."""
        return self._frames_sent

    @property
    def inbox_seconds(self) -> float:
        """Audio queued for this agent's ears that the clock has not played yet (longest source)."""
        if not self._inboxes:
            return 0.0
        return max(len(queue) for queue in self._inboxes.values()) / self._bytes_per_second

    @property
    def clock_started_at(self) -> float | None:
        """`time.monotonic()` when the input clock began; the session timeline's zero."""
        return self._clock_started_at

    @property
    def speaking(self) -> bool:
        """Whether a speech frame arrived within the last `SPEAKING_HOLD_S`."""
        return self._last_speech_at is not None and time.monotonic() - self._last_speech_at <= SPEAKING_HOLD_S

    @property
    def speech_onset_at(self) -> float | None:
        """When the current (or last) stretch of speech began, monotonic."""
        return self._speech_onset_at

    @property
    def last_speech_at(self) -> float | None:
        return self._last_speech_at

    @property
    def ever_spoke(self) -> bool:
        return self._ever_spoke

    @property
    def stream_pcm(self) -> bytes:
        """Everything this host's session output since `start_recording()`, on the act clock."""
        return bytes(self._stream)

    def set_listeners(self, listeners: Sequence[LiveAgent], *, always: bool = False) -> None:
        """Who hears this agent live.

        Its output frames are fed to their inboxes as they arrive, so the
        producer must not `hear()` them again. With `always`, forwarding runs
        for the whole act from this host's first speech frame — duplex — rather
        than only while it holds the floor.
        """
        self._listeners = [agent for agent in listeners if agent is not self]
        self._forward_always = always

    def start_recording(self) -> None:
        """Begin keeping this host's output stream, aligned to now.

        Every duplex host starts recording at the same instant, so the streams
        line up on one act clock; a host whose first frame arrives late is
        padded with silence for the gap.
        """
        self._recording = True
        self._record_t0 = time.monotonic()
        self._stream = bytearray()

    def stop_recording(self) -> None:
        self._recording = False

    # ---------- lifecycle ----------

    async def __aenter__(self) -> LiveAgent:
        if self._tg is not None:
            raise RuntimeError(f"{self.name}: LiveAgent entered twice")
        tg = anyio.create_task_group()
        await tg.__aenter__()
        self._tg = tg
        tg.start_soon(self._read_forever)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        tg = self._tg
        if tg is None:
            return
        self._tg = None
        try:
            if exc is None and self._started and not self._finished and not self._closed:
                # Normal exit without an explicit `finish()`: still ask for a
                # graceful close so the server finalizes the session.
                await self.finish()
        finally:
            self._clock_running = False
            tg.cancel_scope.cancel()
            # Never hand the task group the caller's exception: the reader and
            # clock swallow their own, so this only unwinds the cancellation.
            await tg.__aexit__(None, None, None)

    async def _read_forever(self) -> None:
        """Drain the socket for the life of the agent. Never raises."""
        try:
            async for event in self._conn:
                self._handle(event)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exc:  # websocket-level failure
            if self._fault is None:
                self._fault = RealtimeAPIError(f"{self.name}: live connection failed: {exc}")
        finally:
            if not self._closed:
                self._closed = True
                self._close_reason = self._close_reason or "connection closed without session.closed"
            self._clock_running = False
            self._notify()

    async def _run_clock(self) -> None:
        """Stream one input frame every `FRAME_MS` for as long as the session lives.

        The timeline only moves with input audio, so this runs from
        `session.started` to `session.close` regardless of who holds the floor —
        including while this agent itself is speaking. Frames come from the
        inbox when there is something to hear and are silence otherwise. Never
        raises: a failed send is already recorded as the agent's fault.
        """
        silence = b"\x00" * self._frame_bytes
        next_tick = time.monotonic()
        self._clock_started_at = next_tick
        while self._clock_running and not self._closed and self._fault is None:
            frame = self._next_input_frame(silence)
            try:
                await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(frame).decode()})
            except RealtimeAPIError:
                break
            self._frames_sent += 1
            next_tick += FRAME_MS / 1000
            await anyio.sleep(max(0.0, next_tick - time.monotonic()))

    def _next_input_frame(self, silence: bytes) -> bytes:
        """One frame for the ears: the per-source queues mixed, or silence."""
        pieces: list[bytes] = []
        drained = False
        for source, queue in list(self._inboxes.items()):
            if not queue:
                continue
            piece = bytes(queue[: self._frame_bytes])
            del queue[: self._frame_bytes]
            pieces.append(piece.ljust(self._frame_bytes, b"\x00"))
            if not queue:
                del self._inboxes[source]
                drained = True
        if drained and not self._inboxes:
            self._notify()  # `hear()` may be waiting for the drain
        if not pieces:
            return silence
        return pieces[0] if len(pieces) == 1 else mix_pcm16(pieces)

    def _handle(self, event: object) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "session.output_audio.delta":
            self._handle_audio(base64.b64decode(getattr(event, "delta", "") or ""))
        elif event_type == "session.output_transcript.delta":
            text = getattr(event, "delta", "") or ""
            if self._on_floor:
                self._turn_text.append(text)
            start_ms = getattr(event, "start_ms", None)
            end_ms = getattr(event, "end_ms", None)
            if isinstance(start_ms, int) and isinstance(end_ms, int):
                self.transcript_deltas.append((start_ms, end_ms, text))
        elif event_type == "session.usage.updated":
            self._note_usage(getattr(event, "usage", None))
        elif event_type in (
            "session.instructions.appended",
            "session.commentary.appended",
            "session.thinking.appended",
        ):
            self._pending.pop(str(getattr(event, "client_event_id", "") or ""), None)
            self._notify()
        elif event_type == "session.started":
            self._started = True
            self._started_at = time.monotonic()
            self._notify()
        elif event_type == "session.closed":
            self._note_usage(getattr(event, "usage", None))
            self._close_reason = str(getattr(event, "reason", "unknown"))
            self._closed = True
            self._clock_running = False
            self._notify()
        elif event_type == "error":
            self._handle_error(getattr(event, "error", event))
        elif event_type == "info":
            logger.debug("%s: live info: %s", self.name, getattr(event, "message", event))

    def _handle_audio(self, pcm: bytes) -> None:
        loud = frame_rms(pcm) >= SPEECH_RMS_THRESHOLD
        now = time.monotonic()
        if loud:
            if not self.speaking:
                self._speech_onset_at = now
            self._last_speech_at = now
            self._ever_spoke = True
        if self._recording:
            if not self._stream:
                # First frame since recording began: pad for the time the act
                # clock ran before this session produced anything, so every
                # host's stream shares one zero.
                gap = int((now - self._record_t0) * self._bytes_per_second)
                gap -= gap % 2
                self._stream.extend(b"\x00" * max(0, gap))
            self._stream.extend(pcm)
        if self._forward_always:
            # Duplex: everything from the first speech frame on, all act long.
            if self._ever_spoke:
                for listener in self._listeners:
                    listener.feed(pcm, self.id)
            self._notify()
            return
        if not self._on_floor:
            if loud:
                # Counted here, reported once per turn in `speak()`.
                self._off_floor_bytes += len(pcm)
            return
        if loud and self._speech_start is None:
            self._speech_start = len(self._turn_pcm)
        self._turn_pcm.extend(pcm)
        self._frame_events += 1
        if loud:
            self._speech_end = len(self._turn_pcm)
            self._speech_events += 1
        if self._speech_start is not None:
            # From the first speech frame on, pauses included: the listeners
            # should hear the delivery, not the words butted together. Leading
            # silence is dropped, and the floor is released before trailing
            # silence runs long.
            for listener in self._listeners:
                listener.feed(pcm, self.id)
        self._notify()

    def _handle_error(self, error: object) -> None:
        code = getattr(error, "code", None)
        if code == INJECTION_INCOMPLETE:
            lost = self._pending.pop(str(getattr(error, "client_event_id", "") or ""), None)
            logger.warning(
                "%s: the session closed before a producer note was applied - lost: %s",
                self.name,
                lost or "(an unidentified append)",
            )
            self._notify()
            return
        if self._fault is None:
            self._fault = RealtimeAPIError(f"{self.name}: live error: {_describe_error(error)}")
        self._notify()

    def _note_usage(self, usage: object) -> None:
        seconds = getattr(usage, "seconds", None)
        if isinstance(seconds, (int, float)) and seconds >= 0:
            # Cumulative for the session — take the latest, never sum.
            self._reported_seconds = max(self._reported_seconds, float(seconds))

    def _notify(self) -> None:
        # anyio events are one-shot; swap in a fresh one and release whoever
        # was waiting on the old one. Waiters re-check state, never the event.
        previous = self._changed
        self._changed = anyio.Event()
        previous.set()

    def _check_alive(self, *, while_speaking: bool = False) -> None:
        if self._fault is not None:
            raise self._fault
        if self._closed:
            # A graceful close mid-act must never read as a quiet turn: the act
            # would checkpoint as complete with a hole in it. Same rationale as
            # the `response.done` requirement in `RealtimeAgent.speak`.
            what = "session closed before the turn finished" if while_speaking else "session is closed"
            raise RealtimeAPIError(f"{self.name}: live {what} ({self._close_reason})")

    async def _wait_until(self, done: Callable[[], bool], timeout: float, *, while_speaking: bool = False) -> bool:
        """True once `done()` holds, False after `timeout`; raises if the session dies."""
        with anyio.move_on_after(timeout):
            while not done():
                self._check_alive(while_speaking=while_speaking)
                event = self._changed
                await event.wait()
            return True
        self._check_alive(while_speaking=while_speaking)
        return False

    async def _send(self, event: dict[str, object]) -> None:
        self._check_alive()
        try:
            await self._conn.send(cast("ClientEventParam", event))
        except Exception as exc:
            # A closed websocket raises from `send` before the reader has seen
            # the close. Same failure, same exception type for the producer.
            if self._fault is None:
                self._fault = RealtimeAPIError(f"{self.name}: live send failed ({event.get('type')}): {exc}")
            raise self._fault from exc

    async def _append(self, kind: str, content: str, label: str) -> None:
        """Send an `instructions`/`commentary` append and remember it until acked."""
        self._event_counter += 1
        event_id = f"{self.id}-{self._event_counter}"
        self._pending[event_id] = label
        await self._send(
            {"type": f"session.{kind}.append", "event_id": event_id, "content": content, "delegation_id": None}
        )

    # ---------- the agent surface ----------

    async def configure(self, instructions: str, *, start_clock: bool = True) -> None:
        """Start the session with the persona, voice and PCM format for this act.

        By default the input clock starts as soon as `session.started` arrives,
        which is what the cued loop wants. A duplex table passes
        `start_clock=False` and calls `start_clock()` itself once *every* host
        is up and wired: the model's timeline only moves with input audio, so
        holding the clocks is what keeps the opener from speaking into a room
        where the slower host is not yet listening and nobody is recording.
        """
        if self._tg is None:
            raise RuntimeError(f"{self.name}: enter the LiveAgent (async with) before configure()")
        audio: dict[str, object] = {"format": {"type": "audio/pcm", "rate": self._sample_rate}}
        if self.spec.voice:
            audio["output"] = {"voice": self.spec.voice}
        # The cued loop's "speak only when cued" contract; a duplex table gets
        # its own rules from `duplex.build_duplex_rules` instead.
        rules = _turn_taking_rules(self._turn_seconds) if self.mode == "cued" else ""
        session: dict[str, object] = {
            "model": self.model,
            "instructions": instructions + rules,
            "audio": audio,
        }
        await self._send({"type": "session.start", "session": session})
        if not await self._wait_until(lambda: self._started, SESSION_START_TIMEOUT_S):
            raise RealtimeAPIError(f"{self.name}: no session.started within {SESSION_START_TIMEOUT_S:.0f}s")
        if start_clock:
            self.start_clock()

    def start_clock(self) -> None:
        """Begin streaming input frames; the model's timeline starts here."""
        if self._tg is None or not self._started:
            raise RuntimeError(f"{self.name}: start_clock() needs an entered, started session")
        if self._clock_running:
            return
        self._clock_running = True
        self._tg.start_soon(self._run_clock)

    def check_alive(self) -> None:
        """Raise the session's fault, or RealtimeAPIError if it has closed."""
        self._check_alive()

    async def steer(self, note: str) -> None:
        """Inject a producer note the audience never hears — on the silent channel."""
        await self.think(note, label=f"steer: {note[:60]!r}")

    async def think(self, note: str, *, label: str | None = None) -> None:
        """`session.thinking.append`: context the model acts on but does not voice.

        Verified live with a note written as a line ("Now Gus has wrapped, so
        you can land the through-line and sign off"): the model followed it and
        said something else. The same text on `instructions.append` was read
        out verbatim in a recording. 500-token limit, trimmed like a steer.
        """
        if len(note) > STEER_MAX_CHARS:
            logger.warning("%s: producer note truncated from %d to %d chars", self.name, len(note), STEER_MAX_CHARS)
            note = note[:STEER_MAX_CHARS]
        await self._append("thinking", note, label or f"note: {note[:60]!r}")

    def feed(self, pcm: bytes, source: str = "") -> None:
        """Queue audio for this agent's ears without waiting; the clock plays it out.

        `source` keeps different speakers in different queues so the clock can
        mix them frame by frame instead of playing one after the other.
        """
        if pcm:
            self._ever_heard_a_turn = True
            self._inboxes.setdefault(source, bytearray()).extend(pcm)

    async def hear(self, pcm: bytes) -> None:
        """Feed another agent's audio into this one's ears, and wait until it has been heard.

        Awaiting the drain — rather than returning once queued — is what makes
        "heard" mean heard: the producer cues this host right after, and a host
        cued while a turn is still queued in front of it would answer what it
        has not yet been played. Live listeners of a live speaker never come
        through here (they are fed frame by frame while the speaker talks);
        this is the path for a live host hearing a Realtime host in a mixed
        episode, and it costs the turn's length in wall clock, once.
        """
        if not pcm:
            return
        self.feed(pcm, "replay")
        bound = len(pcm) / self._bytes_per_second + 5.0
        if not await self._wait_until(lambda: not self._inboxes, bound):
            logger.warning("%s: %.1fs of audio still unplayed after %.0fs", self.name, self.inbox_seconds, bound)

    async def speak(self) -> SpokenTurn:
        """Take the floor: cue the model, then collect its speech until it stops."""
        self._check_alive()
        if self._off_floor_bytes != self._off_floor_reported:
            logger.debug(
                "%s: discarded %.1fs of speech out of turn since the last cue",
                self.name,
                (self._off_floor_bytes - self._off_floor_reported) / self._bytes_per_second,
            )
            self._off_floor_reported = self._off_floor_bytes
        self._on_floor = True
        self._turn_pcm = bytearray()
        self._turn_text = []
        self._speech_start = None
        self._speech_end = 0
        self._frame_events = 0
        self._speech_events = 0
        cap_bytes = int(TURN_AUDIO_CAP_FACTOR * self._turn_seconds * self._bytes_per_second)
        truncated = False
        try:
            if self._ever_heard_a_turn:
                # Let what it just heard settle before asking for an answer.
                await anyio.sleep(self._settle_s)
            await self._append("instructions", TURN_CUE, "turn cue")
            await self._append("commentary", TURN_NUDGE, "turn nudge")
            # Bounded, and never gating: speech frames that arrive meanwhile
            # are collected by the reader and found by the wait below.
            if not await self._wait_until(lambda: not self._pending, ACK_WAIT_S, while_speaking=True):
                logger.debug("%s: %d append(s) unacknowledged after %.1fs", self.name, len(self._pending), ACK_WAIT_S)

            if not await self._wait_until(lambda: self._speech_events > 0, self._start_wait_s, while_speaking=True):
                # Not raised: the producer's stall timeout owns hangs, and a
                # model that declined to speak is a quiet turn, not a dead one.
                logger.warning(
                    "%s: no speech within %.1fs of the cue - recording an empty turn", self.name, self._start_wait_s
                )
                return SpokenTurn(pcm=b"", text="", usage=self.take_usage(), truncated=False)

            # Set by the reader on the first speech frame, which the wait above
            # saw; fixed for the rest of the turn.
            start = self._speech_start if self._speech_start is not None else 0
            while True:
                if self._speech_end - start >= cap_bytes:
                    truncated = True
                    logger.warning(
                        "%s: turn cut at %.0fs of speech (%.0fx turn_seconds) - the model is not honouring the "
                        "length rule; whatever it says next is discarded",
                        self.name,
                        (self._speech_end - start) / self._bytes_per_second,
                        TURN_AUDIO_CAP_FACTOR,
                    )
                    await self._append("instructions", STOP_CUE, "stop cue")
                    break
                trailing = (len(self._turn_pcm) - self._speech_end) / self._bytes_per_second
                if trailing >= self._silence_s:
                    break
                if not await self._wait_for_frame(self._silence_s):
                    # The stream itself went quiet — should not happen on a
                    # live session, but it is as much an end as silence is.
                    break
        finally:
            self._on_floor = False

        pcm = bytes(self._turn_pcm[start : min(self._speech_end, start + cap_bytes)])
        trimmed = trim_false_start(pcm, self._bytes_per_second, self._frame_bytes)
        if len(trimmed) < len(pcm):
            logger.debug(
                "%s: trimmed a %.1fs false start", self.name, (len(pcm) - len(trimmed)) / self._bytes_per_second
            )
        pcm = trimmed
        text = "".join(self._turn_text).strip()
        # The turn is over when everyone has heard it: a listener cued with
        # this speech still queued in front of it would answer too early.
        await self._wait_for_listeners()
        return SpokenTurn(pcm=pcm, text=text, usage=self.take_usage(), truncated=truncated)

    async def _wait_for_frame(self, timeout: float) -> bool:
        """True when the next on-floor output frame lands, False after `timeout`."""
        seen = self._frame_events
        return await self._wait_until(lambda: self._frame_events != seen, timeout, while_speaking=True)

    async def _wait_for_listeners(self) -> None:
        if not self._listeners:
            return
        backlog = max(listener.inbox_seconds for listener in self._listeners)
        with anyio.move_on_after(backlog + 2.0):
            while any(listener.inbox_seconds > 0 for listener in self._listeners):
                await anyio.sleep(FRAME_MS / 1000)

    # ---------- billing ----------

    def _billable_seconds(self) -> float:
        """Seconds this session has run up so far.

        The larger of what the API has reported and the wall clock since
        `session.started`. The two track each other once input is streaming
        (measured 19.0 reported at a 20s close), but the report lags the turn
        it lands after; billing is per session-minute, so wall clock is the
        honest floor, and erring high keeps the cost ceiling fail-closed.
        """
        wall = 0.0 if self._started_at is None else time.monotonic() - self._started_at
        return max(self._reported_seconds, wall)

    def take_usage(self) -> RealtimeUsage:
        """Billable seconds since the last time this was called.

        The one accounting seam: `speak()` takes it for the speaker's turn, the
        producer takes it for every *listening* Live host after each turn (a
        session bills while it listens, and a ceiling that only saw speakers
        let listeners run past it), and `finish()` takes what is left. Each
        call moves the mark, so no second is charged twice.
        """
        total = self._billable_seconds()
        delta = max(0.0, total - self._billed_seconds)
        self._billed_seconds = total
        return RealtimeUsage(live_seconds=delta)

    async def finish(self) -> RealtimeUsage:
        """Stop the clock, close the session, and report the seconds not yet charged.

        Bounded: a server that never sends `session.closed` still gets the
        session charged at the wall clock, not forgiven.
        """
        if self._finished:
            return RealtimeUsage()
        self._finished = True
        self._clock_running = False
        if not self._closed and self._fault is None:
            try:
                await self._conn.send(cast("ClientEventParam", {"type": "session.close"}))
                await self._wait_until(lambda: self._closed, SESSION_CLOSE_TIMEOUT_S)
            except RealtimeAPIError:
                pass  # closed underneath us; the usage below is still right
            except Exception as exc:
                logger.debug("%s: session.close failed (%s) - charging the wall clock", self.name, exc)
        if self._off_floor_bytes:
            logger.info(
                "%s: discarded %.1fs of speech spoken out of turn over the act",
                self.name,
                self.off_floor_seconds,
            )
        return self.take_usage()
