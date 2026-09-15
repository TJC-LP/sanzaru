# SPDX-License-Identifier: MIT
"""One conversational agent on one Live API connection (`gpt-live-*`).

Same public surface as `RealtimeAgent` — `configure`, `steer`, `hear`, `speak` —
so `producer.run_act` can seat either at the table, but the model underneath is
a different animal and three things about it shape everything here:

- **It is full duplex.** There is no `response.create` for the model's own
  speech and no `response.done` after it: the model decides when to talk, the
  way a person on a call does. The producer's floor control is therefore
  *advisory* — the session instructions tell the model to speak only when cued
  and to stay silent otherwise, and `speak()` sends that cue. Audio the model
  produces while it does *not* hold the floor is discarded (counted and logged
  as off-floor seconds), because a full-duplex model may well talk over the
  host it is hearing.
- **A turn has no end marker.** Output audio simply stops arriving. The end of
  a turn is inferred: no new audio for `end_of_turn_silence_s` after some audio
  has arrived, or `2 x turn_seconds` of audio collected (reported `truncated`).
- **It bills by the session-minute, not the token.** `session.usage.updated`
  carries the session's *cumulative* seconds; each `speak()` reports the delta
  since the previous turn as `RealtimeUsage.live_seconds`, and `finish()` reports
  whatever the close settles on top. Listening time bills like talking time, so
  every host's session costs the whole act.

The socket is drained continuously by a reader task the agent owns, because
events arrive whether or not anyone is in `speak()` — usage updates, the other
side's transcript, the model talking out of turn. Entering the agent as an
async context manager starts the reader; leaving it closes the session.

Every `openai.types.live` import is TYPE_CHECKING-only: the CLI startup-weight
test forbids pulling `openai` in at `sanzaru.cli` import time.
"""

from __future__ import annotations

import base64
import time
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


DEFAULT_END_OF_TURN_SILENCE_S = 1.2
"""How long output audio must be absent, after some has arrived, before the
turn is over. Below ~1s a mid-sentence breath ends the turn; well above it every
turn carries that much dead air before the next host is cued."""

START_WAIT_FACTOR = 3.0
START_WAIT_FLOOR_S = 4.0
"""How long `speak()` waits for the first audio after cueing: `START_WAIT_FACTOR`
silence gaps, never under `START_WAIT_FLOOR_S`. The model has just been fed a
whole turn of audio and has to decide it is done hearing it before it answers."""

TURN_AUDIO_CAP_FACTOR = 2.0
"""A turn is cut at this many `turn_seconds` of audio. There is no token cap on
a Live turn, so this is the only mechanical bound on a monologue; the prompt's
length rule is what actually shapes turns."""

SESSION_START_TIMEOUT_S = 30.0
SESSION_CLOSE_TIMEOUT_S = 10.0

STEER_MAX_CHARS = 1500
"""`session.instructions.append` takes at most 500 tokens. A note is a sentence
or two, so this only ever trims a runaway caller `turn_note`."""

HEAR_CHUNK_BYTES = 32 * 1024
"""Audio per `session.input_audio.append`; a whole 15s turn (720 KiB) in one
WebSocket frame is legal but needlessly large."""

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

    Enter it (`async with agent:`) before `configure()`; the reader task lives
    for exactly that scope. Between `configure()` and `finish()` it behaves like
    `RealtimeAgent`.
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

        self._on_floor = False
        self._turn_pcm = bytearray()
        self._turn_text: list[str] = []
        self._audio_events = 0
        self._off_floor_bytes = 0
        self._off_floor_reported = 0

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
        """Audio the model produced while another host held the floor, discarded."""
        return self._off_floor_bytes / self._bytes_per_second

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
            tg.cancel_scope.cancel()
            # Never hand the task group the caller's exception: the reader
            # swallows its own, so this only unwinds the cancellation.
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
            self._notify()

    def _handle(self, event: object) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "session.output_audio.delta":
            pcm = base64.b64decode(getattr(event, "delta", "") or "")
            if self._on_floor:
                self._turn_pcm.extend(pcm)
                self._audio_events += 1
                self._notify()
            else:
                # Counted here, reported once per turn in `speak()`: deltas
                # arrive many times a second.
                self._off_floor_bytes += len(pcm)
        elif event_type == "session.output_transcript.delta":
            if self._on_floor:
                self._turn_text.append(getattr(event, "delta", "") or "")
        elif event_type == "session.usage.updated":
            self._note_usage(getattr(event, "usage", None))
        elif event_type == "session.started":
            self._started = True
            self._started_at = time.monotonic()
            self._notify()
        elif event_type == "session.closed":
            self._note_usage(getattr(event, "usage", None))
            self._close_reason = str(getattr(event, "reason", "unknown"))
            self._closed = True
            self._notify()
        elif event_type == "error":
            if self._fault is None:
                self._fault = RealtimeAPIError(
                    f"{self.name}: live error: {_describe_error(getattr(event, 'error', event))}"
                )
            self._notify()
        elif event_type == "info":
            logger.debug("%s: live info: %s", self.name, getattr(event, "message", event))

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

    async def _wait_for_change(self) -> None:
        event = self._changed
        await event.wait()

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

    # ---------- the agent surface ----------

    async def configure(self, instructions: str) -> None:
        """Start the session with the persona, voice and PCM format for this act."""
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
        try:
            with anyio.fail_after(SESSION_START_TIMEOUT_S):
                while not self._started:
                    self._check_alive()
                    await self._wait_for_change()
        except TimeoutError as exc:
            raise RealtimeAPIError(f"{self.name}: no session.started within {SESSION_START_TIMEOUT_S:.0f}s") from exc

    async def steer(self, note: str) -> None:
        """Inject a producer note the audience never hears."""
        if len(note) > STEER_MAX_CHARS:
            logger.warning("%s: steering note truncated from %d to %d chars", self.name, len(note), STEER_MAX_CHARS)
            note = note[:STEER_MAX_CHARS]
        await self._send({"type": "session.instructions.append", "content": note, "delegation_id": None})

    async def hear(self, pcm: bytes) -> None:
        """Feed another agent's audio into this one's ears."""
        if not pcm:
            return
        for offset in range(0, len(pcm), HEAR_CHUNK_BYTES):
            chunk = pcm[offset : offset + HEAR_CHUNK_BYTES]
            await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(chunk).decode()})

    async def speak(self) -> SpokenTurn:
        """Take the floor: cue the model, then collect audio until it stops."""
        self._check_alive()
        if self._off_floor_bytes != self._off_floor_reported:
            logger.debug(
                "%s: discarded %.1fs of audio spoken out of turn since the last cue",
                self.name,
                (self._off_floor_bytes - self._off_floor_reported) / self._bytes_per_second,
            )
            self._off_floor_reported = self._off_floor_bytes
        self._on_floor = True
        self._turn_pcm = bytearray()
        self._turn_text = []
        cap_bytes = int(TURN_AUDIO_CAP_FACTOR * self._turn_seconds * self._bytes_per_second)
        truncated = False
        try:
            await self._send({"type": "session.instructions.append", "content": TURN_CUE, "delegation_id": None})
            await self._send({"type": "session.commentary.append", "content": TURN_NUDGE, "delegation_id": None})

            if not await self._wait_for_audio(self._start_wait_s):
                # Not raised: the producer's stall timeout owns hangs, and a
                # model that declined to speak is a quiet turn, not a dead one.
                logger.warning(
                    "%s: no audio within %.1fs of the cue - recording an empty turn", self.name, self._start_wait_s
                )
                return SpokenTurn(pcm=b"", text="", usage=self._take_usage(), truncated=False)

            while True:
                if len(self._turn_pcm) >= cap_bytes:
                    truncated = True
                    logger.warning(
                        "%s: turn cut at %.0fs of audio (%.0fx turn_seconds) - the model is not honouring the "
                        "length rule; whatever it says next is discarded",
                        self.name,
                        len(self._turn_pcm) / self._bytes_per_second,
                        TURN_AUDIO_CAP_FACTOR,
                    )
                    await self._send(
                        {"type": "session.instructions.append", "content": STOP_CUE, "delegation_id": None}
                    )
                    break
                if not await self._wait_for_audio(self._silence_s):
                    break
        finally:
            self._on_floor = False

        pcm = bytes(self._turn_pcm[:cap_bytes])
        text = "".join(self._turn_text).strip()
        return SpokenTurn(pcm=pcm, text=text, usage=self._take_usage(), truncated=truncated)

    async def _wait_for_audio(self, timeout: float) -> bool:
        """True once a new audio delta lands, False after `timeout` without one.

        Raises `RealtimeAPIError` if the session faults or closes while waiting.
        """
        seen = self._audio_events
        with anyio.move_on_after(timeout):
            while self._audio_events == seen:
                self._check_alive(while_speaking=True)
                await self._wait_for_change()
            return True
        self._check_alive(while_speaking=True)
        return False

    # ---------- billing ----------

    def _billable_seconds(self) -> float:
        """Seconds this session has run up so far.

        The larger of what the API has reported and the wall clock since
        `session.started`. The API's figure lags the turn it is reported after,
        and billing is per session-minute, so wall clock is the honest floor;
        erring high here is what keeps the cost ceiling fail-closed.
        """
        wall = 0.0 if self._started_at is None else time.monotonic() - self._started_at
        return max(self._reported_seconds, wall)

    def _take_usage(self) -> RealtimeUsage:
        """Billable seconds since the last time this was called."""
        total = self._billable_seconds()
        delta = max(0.0, total - self._billed_seconds)
        self._billed_seconds = total
        return RealtimeUsage(live_seconds=delta)

    async def finish(self) -> RealtimeUsage:
        """Close the session and report the seconds not yet charged.

        Bounded: a server that never sends `session.closed` still gets the
        session charged at the wall clock, not forgiven.
        """
        if self._finished:
            return RealtimeUsage()
        self._finished = True
        if not self._closed and self._fault is None:
            try:
                await self._conn.send(cast("ClientEventParam", {"type": "session.close"}))
                with anyio.move_on_after(SESSION_CLOSE_TIMEOUT_S):
                    while not self._closed and self._fault is None:
                        await self._wait_for_change()
            except Exception as exc:
                logger.debug("%s: session.close failed (%s) - charging the wall clock", self.name, exc)
        if self._off_floor_bytes:
            logger.info(
                "%s: discarded %.1fs of audio spoken out of turn over the act",
                self.name,
                self.off_floor_seconds,
            )
        return self._take_usage()
