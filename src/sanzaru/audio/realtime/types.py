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

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

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

    id: str
    name: str
    voice: str = "marin"
    """A Realtime API voice (marin, cedar, alloy, ash, ballad, coral, echo, sage,
    shimmer, verse) — not a TTS voice and not an ElevenLabs voice id."""
    persona: str = ""
    model: str | None = None
    """Per-host model override; falls back to the episode model."""


class ActBrief(BaseModel):
    """One recordable chunk of the episode.

    Acts are recorded in *separate* sessions, in parallel, which is what keeps a
    long episode under the Realtime API's 60-minute connection limit and keeps
    per-session context (and therefore cost) flat. Coherence across that seam
    comes from `prior_context` and `handoff`, written during pre-production —
    an act never sees another act's audio.
    """

    id: str
    title: str
    topic: str
    talking_points: list[str] = Field(default_factory=list)
    target_seconds: float = 180.0
    max_turns: int = 20
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
    that turn. An empty string suppresses the note entirely. Setting a note on
    the last turn takes over the closing, so it is on you to land the act."""

    speaking_order: list[str] = Field(default_factory=list)
    """Explicit host id per turn, cycled if shorter than the act. Empty means
    round-robin. Use it to choreograph — `["avery", "rory", "rory", "avery"]`
    lets one host follow their own point before handing back."""


class Rundown(BaseModel):
    """A full episode plan: who is talking, and about what, act by act."""

    title: str
    premise: str = ""
    style: str = ""
    """Tone/format notes applied to every act (e.g. 'dry, technical, no fluff')."""
    hosts: list[HostSpec]
    acts: list[ActBrief]

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
    """Why the act ended: complete | max_turns | target_seconds | budget."""

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
