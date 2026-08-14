# SPDX-License-Identifier: MIT
"""Quality control for a simulated episode.

With a scripted podcast the script is ground truth, so verification means
"did the renderer say what I wrote". With a simulated one there is no script —
the conversation *is* the output — which inverts the problem. Two transcripts
make it tractable:

- **Intended** — the Realtime API hands back `output_audio_transcript` per turn,
  free, describing what the model meant to say.
- **Rendered** — `gpt-transcribe` on the finished audio, describing what a
  listener actually hears.

Comparing them catches the class of failure that motivated all of this (#35: a
provider silently dropping the tail of a sentence) with a deterministic
similarity score and no model judgement at all. A judge model then reads the
rendered transcript against the rundown for the failures a diff cannot see:
ground that was skipped, an act that repeats an earlier one, drift off brief.

Transcription runs per act, not per episode. Three reasons: acts are already on
disk as checkpoints, a 30-minute episode would blow past the 25MB upload limit,
and per-act verdicts are what make `--qc-retry` able to re-record just the acts
that failed.
"""

from __future__ import annotations

from collections.abc import Sequence

import anyio
from aioresult import ResultCapture  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from ...config import logger
from ..verification import (
    DEFAULT_TRANSCRIBE_MODEL,
    SIMILARITY_WARN_THRESHOLD,
    TRANSCRIBE_MAX_BYTES,
    similarity,
    transcribe_bytes,
)
from .types import Rundown, Turn

# Re-exported: this module was their only home before #35 and #39 needed them too.
__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_TRANSCRIBE_MODEL",
    "SIMILARITY_WARN_THRESHOLD",
    "TRANSCRIBE_MAX_BYTES",
    "ActVerdict",
    "QCReport",
    "intended_text",
    "run_qc",
    "similarity",
    "transcribe_bytes",
]

DEFAULT_JUDGE_MODEL = "gpt-5.5"


class ActVerdict(BaseModel):
    """One act's QC result."""

    act_id: str
    similarity: float = 1.0
    """Intended-vs-rendered word overlap, 0-1. A dropped sentence tail shows up
    here as a sharp drop."""
    rendered_words: int = 0
    intended_words: int = 0
    covered_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)
    repeats_earlier: bool = False
    off_brief: bool = False
    truncated_turns: int = 0
    """Turns in this act that hit the token cap, from the recording itself."""
    tail_truncated: bool = False
    """True when the act's *final* turn was cut off. A mid-act truncation can
    pass as a barge-in; an act that ends mid-sentence is an audible defect and
    was the one truncation QC consistently failed to flag."""
    capped_short: bool = False
    """True when the act spent every extension turn and still landed short of
    `target_seconds` (`stop_reason == "max_turns"`). Derived from the recording,
    like `tail_truncated`: #46 restored the stop reason, but nothing consumed
    it, so such an act was exactly as unflagged as before."""
    notes: str = ""
    transcription_error: str = ""

    @property
    def flagged(self) -> bool:
        return bool(
            self.missed_points
            or self.repeats_earlier
            or self.off_brief
            or self.tail_truncated
            or self.capped_short
            or self.similarity < SIMILARITY_WARN_THRESHOLD
        )

    @property
    def retry_may_help(self) -> bool:
        """Whether re-recording this act at the SAME settings could fix it.

        Every content judgement is a real second chance for a fresh take. The
        two recording-derived flags are not:

        - a tail truncation is mechanical — the final turn hit
          `max_output_tokens`, and re-recording against that same cap mostly
          buys the defect twice. It wants a human raising `turn_tokens`.
        - a capped-short act already spent every turn `max_turns` allows, so a
          fresh take at the same budget caps again. It wants `max_turns` raised
          or `target_seconds` lowered.

        Both stay in `flagged`, so a person still sees them.
        """
        return bool(
            self.missed_points or self.repeats_earlier or self.off_brief or self.similarity < SIMILARITY_WARN_THRESHOLD
        )


class QCReport(BaseModel):
    """Episode-level QC outcome."""

    verdict: str = "pass"
    """pass | warn — never a hard fail: the audio exists either way, and the
    caller decides whether to re-record."""
    summary: str = ""
    acts: list[ActVerdict] = Field(default_factory=list)
    flagged_acts: list[str] = Field(default_factory=list)
    """Acts worth a human's attention."""

    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    transcribed_minutes: float = 0.0

    @property
    def retryable_acts(self) -> list[str]:
        """The subset `qc_retry` should re-record: flags a fresh take can fix.

        Not the same as `flagged_acts`, which is what a person should look at.
        """
        return [v.act_id for v in self.acts if v.retry_may_help]


class _JudgedAct(BaseModel):
    act_id: str
    covered_points: list[str]
    missed_points: list[str]
    repeats_earlier: bool
    off_brief: bool
    notes: str


class _Judgement(BaseModel):
    summary: str
    acts: list[_JudgedAct]


def intended_text(turns: Sequence[Turn]) -> str:
    """What the models meant to say, in speaking order."""
    return " ".join(turn.text for turn in turns if turn.text)


async def _judge(rundown: Rundown, rendered: dict[str, str], model: str) -> _Judgement | None:
    """Read the rendered transcripts against the rundown."""
    from ...config import get_client

    sections: list[str] = [
        f"EPISODE: {rundown.title}",
        f"PREMISE: {rundown.premise}" if rundown.premise else "",
        "",
        "You are reviewing a podcast whose acts were recorded separately and in "
        "parallel by voice models that could not hear each other. For each act, "
        "compare the transcript against the brief it was recorded from.",
        "",
        "Report, per act:",
        "  - covered_points / missed_points: which of the act's talking points "
        "actually got discussed, quoting the point text verbatim.",
        "  - repeats_earlier: true if this act re-covers ground from an earlier "
        "act, re-introduces the show, or re-introduces the hosts.",
        "  - off_brief: true if the conversation wandered away from the act's topic.",
        "  - notes: one sentence, only if something is worth a human's attention.",
        "",
        "Calibrate repeats_earlier for how these acts are made: they record in "
        "parallel and only know each other as one line of text, so brief thematic "
        "overlap, a one-line recap at an act boundary, and the episode's core "
        "tension resurfacing are all STRUCTURAL and must not be flagged — a "
        "re-record cannot remove them. Flag repeats_earlier only when an act "
        "re-tells the same story or example an earlier act already told, spends "
        "multiple exchanges on ground an earlier act already settled, or "
        "re-introduces the show or the hosts.",
        "",
        "Calibrate off_brief the same way: hosts paraphrase talking points in "
        "their own words, and a fluent paraphrase that keeps the substance is "
        "on-brief. Flag off_brief only when the substance changed or the "
        "conversation left the act's topic.",
    ]
    for act in rundown.acts:
        transcript = rendered.get(act.id, "")
        if not transcript:
            continue
        points = "\n".join(f"      - {p}" for p in act.talking_points)
        sections.extend(
            [
                "",
                f"--- {act.id}: {act.title} ---",
                f"    TOPIC: {act.topic}",
                f"    TALKING POINTS:\n{points}" if points else "",
                f"    TRANSCRIPT: {transcript}",
            ]
        )

    client = get_client()
    response = await client.responses.parse(
        model=model,
        input="\n".join(s for s in sections if s),
        text_format=_Judgement,
    )
    parsed = response.output_parsed
    if parsed is None:
        logger.warning("QC judge (%s) returned no structured verdict", model)
    return parsed


async def run_qc(
    rundown: Rundown,
    act_audio: dict[str, bytes],
    act_turns: dict[str, list[Turn]],
    act_stop_reasons: dict[str, str] | None = None,
    *,
    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    limiter: anyio.CapacityLimiter | None = None,
) -> QCReport:
    """Transcribe every act and judge the episode against its rundown.

    Args:
        rundown: The plan the episode was recorded from.
        act_audio: act_id → rendered audio bytes (mp3).
        act_turns: act_id → the turns the models reported speaking.
        act_stop_reasons: act_id → why recording stopped. Only `ActResult` knows
            this and `run_qc` never saw one, which is why an act capped short of
            its target was visible in the result and invisible to QC (#48).
        transcribe_model: Batch transcription model for the rendered audio.
        judge_model: Text model that reads transcripts against the briefs.
        limiter: Bounds concurrent transcription requests.

    Returns:
        A report. Never raises for QC-level problems: a transcription failure
        becomes an unverified act, not a lost episode.
    """
    act_ids = [act.id for act in rundown.acts if act.id in act_audio]
    if not act_ids:
        return QCReport(verdict="pass", summary="nothing to verify", transcribe_model=transcribe_model)

    async def _one(act_id: str) -> tuple[str, str, str]:
        """Returns (act_id, rendered_text, error)."""
        audio = act_audio[act_id]
        if len(audio) > TRANSCRIBE_MAX_BYTES:
            return (
                act_id,
                "",
                f"act audio is {len(audio) / 1e6:.1f}MB, over the {TRANSCRIBE_MAX_BYTES / 1e6:.0f}MB limit",
            )
        try:
            if limiter is None:
                text = await transcribe_bytes(audio, f"{act_id}.mp3", transcribe_model)
            else:
                async with limiter:
                    text = await transcribe_bytes(audio, f"{act_id}.mp3", transcribe_model)
        except Exception as exc:  # noqa: BLE001 — QC must degrade, never lose the episode
            logger.warning("QC transcription failed for %s: %s", act_id, exc)
            return act_id, "", f"{type(exc).__name__}: {exc}"
        return act_id, text, ""

    async with anyio.create_task_group() as tg:
        captures = [ResultCapture.start_soon(tg, _one, act_id) for act_id in act_ids]
    # Read by index — the limiter only delays task entry, it does not reorder.
    transcribed = [c.result() for c in captures]

    rendered: dict[str, str] = {}
    verdicts: list[ActVerdict] = []
    total_seconds = 0.0
    for act_id, text, error in transcribed:
        turns = act_turns.get(act_id, [])
        intended = intended_text(turns)
        total_seconds += sum(t.seconds for t in turns)
        if text:
            rendered[act_id] = text
        verdicts.append(
            ActVerdict(
                act_id=act_id,
                similarity=round(similarity(intended, text), 3) if text else 0.0,
                rendered_words=len(text.split()),
                intended_words=len(intended.split()),
                truncated_turns=sum(1 for t in turns if t.truncated),
                tail_truncated=bool(turns) and turns[-1].truncated,
                capped_short=(act_stop_reasons or {}).get(act_id) == "max_turns",
                transcription_error=error,
            )
        )

    judgement = None
    if rendered:
        try:
            judgement = await _judge(rundown, rendered, judge_model)
        except Exception as exc:  # noqa: BLE001 — same reasoning as above
            logger.warning("QC judge failed: %s", exc)

    by_id = {v.act_id: v for v in verdicts}
    if judgement is not None:
        for judged in judgement.acts:
            verdict = by_id.get(judged.act_id)
            if verdict is None:
                continue
            verdict.covered_points = judged.covered_points
            verdict.missed_points = judged.missed_points
            verdict.repeats_earlier = judged.repeats_earlier
            verdict.off_brief = judged.off_brief
            verdict.notes = judged.notes

    flagged = [v.act_id for v in verdicts if v.flagged or v.transcription_error]
    return QCReport(
        verdict="warn" if flagged else "pass",
        summary=(judgement.summary if judgement else "") or ("no issues found" if not flagged else ""),
        acts=verdicts,
        flagged_acts=flagged,
        transcribe_model=transcribe_model,
        judge_model=judge_model,
        transcribed_minutes=round(total_seconds / 60.0, 2),
    )
