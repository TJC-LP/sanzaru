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

Live agents in one act hear each other *as they speak*: the agent on the floor
forwards each output frame, from the first speech frame on, into its listeners'
inboxes (`set_listeners`), where their clocks play it out at pace. That is what
keeps an act at ~1x real time instead of speak-then-replay's 2x.

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
from .types import HostSpec, RealtimeUsage

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
    ) -> None:
        if end_of_turn_silence_s <= 0:
            raise ValueError("end_of_turn_silence_s must be positive")
        self.spec = spec
        self.model = model
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

        # ---- ears: the input clock and its inbox ----
        self._clock_running = False
        self._inbox = bytearray()
        self._frames_sent = 0
        self._listeners: list[LiveAgent] = []

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

        # ---- injections awaiting their `*.appended` ----
        self._pending: dict[str, str] = {}
        self._event_counter = 0

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
        """Audio queued for this agent's ears that the clock has not played yet."""
        return len(self._inbox) / self._bytes_per_second

    def set_listeners(self, listeners: Sequence[LiveAgent]) -> None:
        """Who hears this agent live: its output frames are fed to their inboxes
        as they arrive, so the producer must not `hear()` them again."""
        self._listeners = [agent for agent in listeners if agent is not self]

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
        while self._clock_running and not self._closed and self._fault is None:
            if self._inbox:
                frame = bytes(self._inbox[: self._frame_bytes])
                del self._inbox[: self._frame_bytes]
                if len(frame) < self._frame_bytes:
                    frame += silence[len(frame) :]
                if not self._inbox:
                    self._notify()  # `hear()` may be waiting for the drain
            else:
                frame = silence
            try:
                await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(frame).decode()})
            except RealtimeAPIError:
                break
            self._frames_sent += 1
            next_tick += FRAME_MS / 1000
            await anyio.sleep(max(0.0, next_tick - time.monotonic()))

    def _handle(self, event: object) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "session.output_audio.delta":
            self._handle_audio(base64.b64decode(getattr(event, "delta", "") or ""))
        elif event_type == "session.output_transcript.delta":
            if self._on_floor:
                self._turn_text.append(getattr(event, "delta", "") or "")
        elif event_type == "session.usage.updated":
            self._note_usage(getattr(event, "usage", None))
        elif event_type in ("session.instructions.appended", "session.commentary.appended"):
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
                listener.feed(pcm)
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

    async def configure(self, instructions: str) -> None:
        """Start the session with the persona, voice and PCM format for this act, then start the clock."""
        if self._tg is None:
            raise RuntimeError(f"{self.name}: enter the LiveAgent (async with) before configure()")
        audio: dict[str, object] = {"format": {"type": "audio/pcm", "rate": self._sample_rate}}
        if self.spec.voice:
            audio["output"] = {"voice": self.spec.voice}
        session: dict[str, object] = {
            "model": self.model,
            "instructions": instructions + _turn_taking_rules(self._turn_seconds),
            "audio": audio,
        }
        await self._send({"type": "session.start", "session": session})
        if not await self._wait_until(lambda: self._started, SESSION_START_TIMEOUT_S):
            raise RealtimeAPIError(f"{self.name}: no session.started within {SESSION_START_TIMEOUT_S:.0f}s")
        self._clock_running = True
        self._tg.start_soon(self._run_clock)

    async def steer(self, note: str) -> None:
        """Inject a producer note the audience never hears."""
        if len(note) > STEER_MAX_CHARS:
            logger.warning("%s: steering note truncated from %d to %d chars", self.name, len(note), STEER_MAX_CHARS)
            note = note[:STEER_MAX_CHARS]
        await self._append("instructions", note, f"steer: {note[:60]!r}")

    def feed(self, pcm: bytes) -> None:
        """Queue audio for this agent's ears without waiting; the clock plays it out."""
        if pcm:
            self._inbox.extend(pcm)

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
        self._inbox.extend(pcm)
        bound = len(pcm) / self._bytes_per_second + 5.0
        if not await self._wait_until(lambda: not self._inbox, bound):
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
