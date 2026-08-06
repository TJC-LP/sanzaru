# SPDX-License-Identifier: MIT
"""Typed values shared by the realtime simulation modules.

A *simulated* podcast is not read from a script: N realtime agents are given
personas and a rundown, and they actually talk to each other. The audio is the
performance, so there is no TTS step and no ground-truth transcript — what the
models said is an output, not an input.

Audio is PCM16 mono at 24kHz in both directions, which is what the Realtime API
speaks natively. Keeping it raw all the way to the stitch step means no
transcoding between an agent's mouth and another agent's ears.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator

logger = logging.getLogger("sanzaru")

# ---------- limits ----------
# Bounds are declared on the models rather than checked in the tool, so they
# reach the MCP JSON schema: an agent reads the rule instead of discovering it
# by spending money on a run that was never going to work.

MAX_ACTS = 24
"""Each act opens one session per host, so this bounds concurrent sessions."""

MAX_ACT_SECONDS = 2400.0
"""A Realtime WebSocket closes at 60 minutes. Well under that, because an act
overruns its target before the producer's closing turn lands."""

MAX_ACT_TURNS = 200
MAX_EPISODE_MINUTES = 240.0
MAX_TURN_TOKENS = 32_000

TURN_EXTENSION_FACTOR = 1.5
"""How far past `max_turns` an act may extend to fill its `target_seconds`.

`max_turns` was a hard stop, which made act length a function of how long the
model's turns happen to run: gpt-realtime's full-size turns measured ~10s
against the mini's ~17.5s under the same 15s rule, so full-model acts stopped a
third short of their target. The turn budget stays what the planner priced —
extension turns only happen inside the time the act was already budgeted for,
and the absolute ceiling (`MAX_ACT_TURNS`) still binds. Lives here rather than
in `producer` so the act's own budget warning can account for it without
importing back the other way."""


def extension_cap(max_turns: int) -> int:
    """Absolute turn ceiling for an act whose planned budget is `max_turns`.

    The one definition of the ceiling: `run_act` stops here, `ActBrief` bounds
    `turn_notes` by it, and the act's budget warning measures against it.
    Recomputing `max_turns * TURN_EXTENSION_FACTOR` inline instead diverges at
    both ends — the single-turn exemption below and the `MAX_ACT_TURNS` clamp.

    `max_turns == 1` is returned unchanged: rounding 1.5 up would silently make
    every one-turn act a two-turn act, and a caller who plans exactly one turn
    is stating a shape, not estimating a duration. (It also keeps the
    `turn_notes[max_turns - 1]` closing-takeover contract intact there, which
    only reads as a takeover while turn 0 is still the final turn.)
    """
    if max_turns <= 1:
        return max_turns
    return min(MAX_ACT_TURNS, math.ceil(max_turns * TURN_EXTENSION_FACTOR))


DEFAULT_TURN_SECONDS = 15.0
"""Target upper bound for one turn, stated in the prompt. Enforced socially, not
mechanically — a hard cut mid-sentence sounds far worse than a long turn. Turns
land at roughly this value in practice (measured ~14-16s against a 15s rule),
which is what makes it usable as a budgeting assumption."""

REALTIME_VOICES: tuple[str, ...] = (
    "marin",
    "cedar",
    "alloy",
    "sage",
    "verse",
    "coral",
    "ash",
    "ballad",
    "echo",
    "shimmer",
)
"""Realtime API voices, in assignment order. marin/cedar lead because they are the
most natural of the set in conversation — that pairing is what the spike used.
An unknown voice warns rather than fails: OpenAI adds them faster than we ship."""

# Ids reach filenames (`{slug}_{run_id}_{act_id}.mp3`). `validate_safe_path`
# sanitizes at the storage layer, but rejecting a separator here turns a
# confusing late failure into an obvious early one.
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$", max_length=64)]
RunId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$", max_length=64)]
Filename = Annotated[str, StringConstraints(pattern=r"^[^/\\]+$", max_length=200)]
Bitrate = Annotated[str, StringConstraints(pattern=r"^\d{1,4}k$")]

# ---------- audio format ----------
# The Realtime API's native PCM format. Both `input_audio_buffer.append` and
# `response.output_audio.delta` use it, so one agent's output frames can be fed
# straight into another's input buffer.
REALTIME_SAMPLE_RATE = 24000
REALTIME_SAMPLE_WIDTH = 2  # PCM16
REALTIME_CHANNELS = 1
_BYTES_PER_SECOND = REALTIME_SAMPLE_RATE * REALTIME_SAMPLE_WIDTH * REALTIME_CHANNELS


def pcm_seconds(pcm: bytes) -> float:
    """Duration of a PCM16/24kHz mono buffer, in seconds."""
    return len(pcm) / _BYTES_PER_SECOND


def pcm_silence(milliseconds: int) -> bytes:
    """A PCM16/24kHz mono silence buffer of the given length."""
    if milliseconds <= 0:
        return b""
    frames = int(REALTIME_SAMPLE_RATE * milliseconds / 1000)
    return b"\x00" * (frames * REALTIME_SAMPLE_WIDTH)


# ---------- rundown ----------


class HostSpec(BaseModel):
    """One participant in the conversation."""

    id: Identifier
    name: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    voice: str = ""
    """A Realtime API voice (marin, cedar, alloy, ash, ballad, coral, echo, sage,
    shimmer, verse) — not a TTS voice and not an ElevenLabs voice id.

    Empty means "you pick": `assign_voices` hands out a distinct voice per host.
    Defaulting to a *named* voice instead made every host who did not state one
    speak as marin, which is only audible after the recording is paid for."""
    persona: str = ""
    model: str | None = None
    """Per-host model override; falls back to the episode model."""

    @model_validator(mode="after")
    def _warn_on_unknown_voice(self) -> HostSpec:
        # A bad voice is only rejected by the API at session.update — after the
        # connections are open and an act is under way. Warning here at least
        # puts the reason in the log before that happens.
        if self.voice and self.voice not in REALTIME_VOICES:
            logger.warning(
                "Host %r uses voice %r, which is not a known realtime voice (%s) - "
                "the session will fail if the API does not recognise it either",
                self.id,
                self.voice,
                ", ".join(REALTIME_VOICES),
            )
        return self


class ActBrief(BaseModel):
    """One recordable chunk of the episode.

    Acts are recorded in *separate* sessions, in parallel, which is what keeps a
    long episode under the Realtime API's 60-minute connection limit and keeps
    per-session context (and therefore cost) flat. Coherence across that seam
    comes from `prior_context` and `handoff`, written during pre-production —
    an act never sees another act's audio.
    """

    id: Identifier
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    topic: Annotated[str, StringConstraints(min_length=1)]
    talking_points: list[str] = Field(default_factory=list, max_length=20)
    target_seconds: float = Field(default=180.0, gt=0, le=MAX_ACT_SECONDS)
    max_turns: int = Field(default=20, ge=1, le=MAX_ACT_TURNS)
    prior_context: str = ""
    """What earlier acts already covered. The single most important field for
    parallel recording: without it, act 3 re-introduces the show."""
    upcoming: str = ""
    """What *later* acts own, so this one doesn't wander into it. The mirror of
    `prior_context`, and just as load-bearing: a QC pass on a three-act episode
    found every act on-brief and every act drifting forward into the next act's
    ground, which then read as repetition. Derived automatically when empty."""
    handoff: str = ""
    """Where this act must leave the conversation for the next one."""

    # ---- the director's chair ----
    # `producer.py` generates sensible defaults for all three, because the
    # measured alternative (no steering at all) drifts badly. But the caller is
    # usually itself an agent, and it is a better producer than a set of
    # f-strings: it knows which point deserves dwelling on and when someone
    # should get two turns in a row. Everything below overrides the default
    # rather than adding to it, and an empty value means "you decide".

    direction: str = ""
    """Free-text direction for this act, added to every host's instructions.
    Where `topic` says what the act is about, this says how to play it —
    'let this one get heated', 'no jokes, the subject is grim'."""

    turn_notes: dict[int, str] = Field(default_factory=dict)
    """Producer notes by zero-based turn index, replacing the generated one for
    that turn. An empty string suppresses the note entirely — except on the last
    act's closing turn, where the hosts are told to wait for a cue and silence
    would end the episode mid-conversation.

    A note on the last *planned* turn (`max_turns - 1`) takes over the closing,
    so it is on you to land the act — and it follows the closing turn if the act
    extends, rather than firing mid-act. Indices up to `extension_cap(max_turns)`
    are accepted, since extension turns are real turns. Setting any note pins the
    act's rotation to the first listed host, so the indices mean what they say."""

    speaking_order: list[str] = Field(default_factory=list, max_length=MAX_ACT_TURNS)
    """Explicit host id per turn, cycled if shorter than the act. Empty means
    round-robin. Use it to choreograph — `["avery", "rory", "rory", "avery"]`
    lets one host follow their own point before handing back."""

    @model_validator(mode="after")
    def _check_turn_notes(self) -> ActBrief:
        """Reject notes keyed to turns this act can never reach.

        The bound is the *extended* ceiling, not `max_turns`: extension turns
        are real turns a caller can legitimately direct, and rejecting them
        with "would never fire" would be false.
        """
        ceiling = extension_cap(self.max_turns)
        out_of_range = sorted(i for i in self.turn_notes if i < 0 or i >= ceiling)
        if out_of_range:
            raise ValueError(
                f"act {self.id!r}: turn_notes {out_of_range} are outside this act's "
                f"turns (0-{ceiling - 1}, including extension) and would never fire"
            )
        return self

    @model_validator(mode="after")
    def _warn_on_turn_budget(self) -> ActBrief:
        """Flag an act that will run out of turns before it runs out of time.

        Such an act stops on the turn cap rather than on its scripted closing
        turn, and its last talking point never gets air — exactly the
        calibration bug the first live run hit. Measured against the *extended*
        ceiling, since that is what actually binds. Turn length is assumed to be
        `DEFAULT_TURN_SECONDS`, so an episode that raises `turn_seconds` can see
        this fire spuriously; it stays a log line for that reason.
        """
        ceiling = extension_cap(self.max_turns)
        if ceiling * DEFAULT_TURN_SECONDS < self.target_seconds:
            logger.warning(
                "act %r: %d turns is unlikely to fill %.0fs even extended to %d turns (turns run "
                "~%.0fs), so it will stop on the turn cap short of its target - raise max_turns or "
                "lower target_seconds",
                self.id,
                self.max_turns,
                self.target_seconds,
                ceiling,
                DEFAULT_TURN_SECONDS,
            )
        return self


class Rundown(BaseModel):
    """A full episode plan: who is talking, and about what, act by act."""

    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    premise: str = ""
    style: str = ""
    """Tone/format notes applied to every act (e.g. 'dry, technical, no fluff')."""
    hosts: list[HostSpec] = Field(min_length=1, max_length=8)
    acts: list[ActBrief] = Field(min_length=1, max_length=MAX_ACTS)

    @model_validator(mode="after")
    def _check_ids_are_unique(self) -> Rundown:
        """Duplicate ids fail silently and expensively, so they fail loudly here.

        Two acts sharing an id overwrite each other's checkpoint — a resumed run
        would quietly lose one. Two hosts sharing an id collapse into one speaker,
        breaking stems and `speaking_order`.
        """
        for label, ids in (("act", [a.id for a in self.acts]), ("host", [h.id for h in self.hosts])):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label} id(s): {', '.join(duplicates)} - ids must be unique")
        return self

    @model_validator(mode="after")
    def _check_speaking_order(self) -> Rundown:
        """Catch a bad host id at parse time, not four acts into a paid run."""
        known = {host.id for host in self.hosts}
        for act in self.acts:
            unknown = sorted(set(act.speaking_order) - known)
            if unknown:
                raise ValueError(
                    f"act {act.id!r}: speaking_order names hosts that are not in the episode "
                    f"({', '.join(unknown)}); known hosts are {', '.join(sorted(known))}"
                )
        return self

    def total_target_seconds(self) -> float:
        return sum(act.target_seconds for act in self.acts)

    def total_max_turns(self) -> int:
        return sum(act.max_turns for act in self.acts)


# ---------- results ----------


class RealtimeUsage(BaseModel):
    """Token usage, split the way realtime bills it.

    Audio and text are priced very differently (32x apart on input for
    gpt-realtime-2.1), and cached input is nearly free, so a single
    `input_tokens` number cannot be turned back into a cost.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    cached_text_tokens: int = 0
    cached_audio_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0

    def __add__(self, other: RealtimeUsage) -> RealtimeUsage:
        return RealtimeUsage(**{k: getattr(self, k) + getattr(other, k) for k in RealtimeUsage.model_fields})

    @property
    def uncached_text_tokens(self) -> int:
        return max(0, self.input_text_tokens - self.cached_text_tokens)

    @property
    def uncached_audio_tokens(self) -> int:
        return max(0, self.input_audio_tokens - self.cached_audio_tokens)


class Turn(BaseModel):
    """One agent's contribution, as the model reported it.

    `text` is the Realtime API's own `output_audio_transcript` — what the model
    *intended* to say. The QC pass compares it against a transcription of the
    rendered audio, which is what listeners actually get.
    """

    act_id: str
    index: int
    speaker_id: str
    speaker_name: str
    text: str
    seconds: float
    truncated: bool = False
    """True when the response hit max_output_tokens and was cut off mid-thought."""


@dataclass(slots=True)
class TurnAudio:
    """A turn plus its PCM. Kept out of the pydantic model so audio never
    accidentally lands in a JSON envelope."""

    turn: Turn
    pcm: bytes


@dataclass(slots=True)
class ActResult:
    """Everything one recorded act produced."""

    act_id: str
    audio: list[TurnAudio] = field(default_factory=list)
    usage: RealtimeUsage = field(default_factory=RealtimeUsage)
    stop_reason: str = "complete"
    """Why the act ended:

    - `target_seconds` — it reached its duration target. The normal ending.
    - `complete` — it delivered its closing turn before reaching the target,
      because one more turn would have overshot it.
    - `max_turns` — it hit the extension ceiling still short of the target. This
      is the undershoot signal: the act is shorter than it was planned to be.

    A cost abort is not one of these — `CostCeilingError` cancels the act rather
    than ending it, so no result is ever built for it."""

    @property
    def turns(self) -> list[Turn]:
        return [ta.turn for ta in self.audio]

    @property
    def seconds(self) -> float:
        return sum(ta.turn.seconds for ta in self.audio)

    def join_pcm(self, gap_ms: int = 0) -> bytes:
        """Concatenate the act's turns into one buffer."""
        if gap_ms <= 0:
            return b"".join(ta.pcm for ta in self.audio)
        gap = pcm_silence(gap_ms)
        return gap.join(ta.pcm for ta in self.audio)


class ActSummary(BaseModel):
    """Envelope-facing summary of one act (no audio)."""

    act_id: str
    title: str
    turns: int
    seconds: float
    stop_reason: str
    truncated_turns: int = 0
    usage: RealtimeUsage = Field(default_factory=RealtimeUsage)
    reused: bool = False
    """True when the act was read back from a checkpoint instead of re-recorded."""
