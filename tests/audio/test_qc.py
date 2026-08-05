"""Tests for the QC pass over a simulated episode.

QC is on by default and runs against audio that has already been paid for, so
the property that matters most is the one it advertises: it never loses an
episode. A transcription that fails, an act too big to upload, a judge that
falls over — each has to come back as an unverified act inside a report, not as
an exception out of `simulate_podcast`.

Everything here mocks at the API layer (`sanzaru.config.get_client`), so no
audio is transcribed and no judge is called.
"""

from types import SimpleNamespace

import pytest

from sanzaru.audio.realtime.qc import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TRANSCRIBE_MODEL,
    SIMILARITY_WARN_THRESHOLD,
    TRANSCRIBE_MAX_BYTES,
    ActVerdict,
    _JudgedAct,
    _Judgement,
    intended_text,
    run_qc,
    similarity,
    transcribe_bytes,
)
from sanzaru.audio.realtime.types import ActBrief, HostSpec, Rundown, Turn

pytestmark = pytest.mark.audio


def _turn(act_id: str, index: int, text: str, seconds: float = 30.0) -> Turn:
    return Turn(
        act_id=act_id,
        index=index,
        speaker_id="avery",
        speaker_name="Avery",
        text=text,
        seconds=seconds,
    )


def _rundown(act_ids=("act1", "act2")) -> Rundown:
    return Rundown(
        title="Stitch",
        premise="the plumbing",
        hosts=[HostSpec(id="avery", name="Avery", voice="marin")],
        acts=[
            ActBrief(id=act_id, title=act_id.title(), topic=f"topic {act_id}", talking_points=["a point"])
            for act_id in act_ids
        ],
    )


class _FakeTranscriptions:
    """`client.audio.transcriptions`, answering per uploaded filename."""

    def __init__(self, results: dict[str, object]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def create(self, *, file, model, response_format):  # type: ignore[no-untyped-def]
        filename, _buffer = file
        self.calls.append({"filename": filename, "model": model, "response_format": response_format})
        answer = self.results.get(filename, "")
        if isinstance(answer, Exception):
            raise answer
        return answer


class _FakeResponses:
    """`client.responses`, standing in for the judge."""

    def __init__(self, parsed: object = None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def parse(self, *, model, input, text_format):  # type: ignore[no-untyped-def] # noqa: A002
        self.calls.append({"model": model, "input": input, "text_format": text_format})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.parsed)


@pytest.fixture
def qc_client(mocker):
    """Install a fake OpenAI client; returns it so calls can be inspected."""

    def build(transcripts=None, judgement=None, judge_error=None):
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=_FakeTranscriptions(transcripts or {})),
            responses=_FakeResponses(parsed=judgement, error=judge_error),
        )
        mocker.patch("sanzaru.config.get_client", return_value=client)
        return client

    return build


# ---------- scoring ----------


@pytest.mark.unit
class TestSimilarity:
    def test_identical_text_scores_one(self):
        assert similarity("the plumbing under it", "the plumbing under it") == 1.0

    def test_case_does_not_count_as_divergence(self):
        assert similarity("The Plumbing", "the plumbing") == 1.0

    def test_a_dropped_tail_falls_below_the_warn_threshold(self):
        """The failure this whole pass exists for (#35): a provider silently
        cutting the end of a sentence."""
        intended = "one two three four five six seven eight nine ten"
        rendered = "one two three four five six"
        assert similarity(intended, rendered) < SIMILARITY_WARN_THRESHOLD

    def test_ordinary_transcription_disagreement_stays_above_it(self):
        # Punctuation and a stray filler are what two models normally differ on;
        # scoring those as failures would make the threshold useless.
        words = [f"word{i}" for i in range(50)]
        rendered = list(words)
        rendered[3] = "word3,"
        rendered[20] = "word20."
        assert similarity(" ".join(words), " ".join(rendered)) > SIMILARITY_WARN_THRESHOLD

    def test_two_silences_agree(self):
        # An act with no intended text and no rendered text is not a mismatch.
        assert similarity("", "") == 1.0
        assert similarity("   ", "") == 1.0

    def test_one_empty_side_scores_zero(self):
        assert similarity("something was said", "") == 0.0
        assert similarity("", "something was heard") == 0.0


@pytest.mark.unit
class TestIntendedText:
    def test_joins_turns_in_speaking_order(self):
        turns = [_turn("act1", 0, "first thing"), _turn("act1", 1, "second thing")]
        assert intended_text(turns) == "first thing second thing"

    def test_skips_turns_that_reported_no_transcript(self):
        turns = [_turn("act1", 0, "kept"), _turn("act1", 1, ""), _turn("act1", 2, "also kept")]
        assert intended_text(turns) == "kept also kept"

    def test_no_turns_is_empty(self):
        assert intended_text([]) == ""


@pytest.mark.unit
class TestTranscribeBytes:
    async def test_uploads_the_act_under_its_own_name(self, qc_client):
        client = qc_client(transcripts={"act1.mp3": "what a listener hears"})
        text = await transcribe_bytes(b"audio", "act1.mp3", "gpt-transcribe")
        assert text == "what a listener hears"
        assert client.audio.transcriptions.calls == [
            {"filename": "act1.mp3", "model": "gpt-transcribe", "response_format": "text"}
        ]

    async def test_accepts_a_response_object_as_well_as_a_string(self, qc_client):
        qc_client(transcripts={"act1.mp3": SimpleNamespace(text="from an object")})
        assert await transcribe_bytes(b"audio", "act1.mp3") == "from an object"

    async def test_an_unrecognisable_response_is_empty_rather_than_a_crash(self, qc_client):
        qc_client(transcripts={"act1.mp3": SimpleNamespace(nothing_useful=True)})
        assert await transcribe_bytes(b"audio", "act1.mp3") == ""


# ---------- the pass as a whole ----------


@pytest.mark.integration
class TestRunQc:
    async def test_a_clean_episode_passes(self, qc_client):
        client = qc_client(
            transcripts={"act1.mp3": "the plumbing under it", "act2.mp3": "and what it costs"},
            judgement=_Judgement(
                summary="clean",
                acts=[
                    _JudgedAct(
                        act_id=act_id,
                        covered_points=["a point"],
                        missed_points=[],
                        repeats_earlier=False,
                        off_brief=False,
                        notes="",
                    )
                    for act_id in ("act1", "act2")
                ],
            ),
        )
        report = await run_qc(
            _rundown(),
            {"act1": b"one", "act2": b"two"},
            {
                "act1": [_turn("act1", 0, "the plumbing under it")],
                "act2": [_turn("act2", 0, "and what it costs")],
            },
        )
        assert report.verdict == "pass"
        assert report.flagged_acts == []
        assert report.summary == "clean"
        assert [v.similarity for v in report.acts] == [1.0, 1.0]
        # Two 30s turns.
        assert report.transcribed_minutes == pytest.approx(1.0)
        assert len(client.responses.calls) == 1

    async def test_nothing_to_verify_is_not_a_failure(self, qc_client):
        client = qc_client()
        report = await run_qc(_rundown(), {}, {})
        assert report.verdict == "pass"
        assert report.summary == "nothing to verify"
        assert client.audio.transcriptions.calls == []

    async def test_an_act_over_the_upload_limit_is_reported_not_uploaded(self, qc_client):
        """25MB is the API's limit, and a long act really can cross it.

        Reporting it as unverified keeps the episode; raising would throw away a
        finished recording over a QC step that is only advisory.
        """
        client = qc_client(transcripts={"act2.mp3": "and what it costs"})
        report = await run_qc(
            _rundown(),
            {"act1": bytes(TRANSCRIBE_MAX_BYTES + 1), "act2": b"two"},
            {"act1": [_turn("act1", 0, "too big to check")], "act2": [_turn("act2", 0, "and what it costs")]},
        )
        oversized = next(v for v in report.acts if v.act_id == "act1")
        assert "over the" in oversized.transcription_error
        assert f"{(TRANSCRIBE_MAX_BYTES + 1) / 1e6:.1f}MB" in oversized.transcription_error
        assert oversized.similarity == 0.0
        assert [call["filename"] for call in client.audio.transcriptions.calls] == ["act2.mp3"]
        assert report.flagged_acts == ["act1"]

    async def test_a_failed_transcription_costs_one_act_not_the_episode(self, qc_client):
        qc_client(
            transcripts={"act1.mp3": RuntimeError("upstream exploded"), "act2.mp3": "and what it costs"},
            judgement=_Judgement(summary="only act2 could be read", acts=[]),
        )
        report = await run_qc(
            _rundown(),
            {"act1": b"one", "act2": b"two"},
            {"act1": [_turn("act1", 0, "never checked")], "act2": [_turn("act2", 0, "and what it costs")]},
        )
        failed = next(v for v in report.acts if v.act_id == "act1")
        assert failed.transcription_error == "RuntimeError: upstream exploded"
        assert failed.rendered_words == 0
        # The other act was still verified, and the report still exists.
        verified = next(v for v in report.acts if v.act_id == "act2")
        assert verified.similarity == 1.0
        assert report.verdict == "warn"
        assert report.flagged_acts == ["act1"]

    async def test_every_transcription_failing_still_returns_a_report(self, qc_client):
        client = qc_client(transcripts={"act1.mp3": RuntimeError("down"), "act2.mp3": RuntimeError("down")})
        report = await run_qc(
            _rundown(),
            {"act1": b"one", "act2": b"two"},
            {"act1": [_turn("act1", 0, "a")], "act2": [_turn("act2", 0, "b")]},
        )
        assert report.flagged_acts == ["act1", "act2"]
        # Nothing rendered, so there is nothing for the judge to read.
        assert client.responses.calls == []

    async def test_a_judge_that_falls_over_leaves_the_diff_verdicts_intact(self, qc_client):
        qc_client(
            transcripts={"act1.mp3": "the plumbing under it", "act2.mp3": "and what it costs"},
            judge_error=RuntimeError("judge exploded"),
        )
        report = await run_qc(
            _rundown(),
            {"act1": b"one", "act2": b"two"},
            {
                "act1": [_turn("act1", 0, "the plumbing under it")],
                "act2": [_turn("act2", 0, "and what it costs")],
            },
        )
        # The deterministic half of QC needs no model, so it still stands.
        assert [v.similarity for v in report.acts] == [1.0, 1.0]
        assert report.verdict == "pass"
        assert report.summary == "no issues found"

    async def test_flagged_acts_collect_every_kind_of_failure_in_rundown_order(self, qc_client):
        qc_client(
            transcripts={
                "act1.mp3": "the plumbing under it",
                "act2.mp3": "and what it costs",
                # A dropped tail: nothing the judge says is needed to flag it.
                "act3.mp3": "one two three",
            },
            judgement=_Judgement(
                summary="two problems",
                acts=[
                    _JudgedAct(
                        act_id="act1",
                        covered_points=[],
                        missed_points=["a point"],
                        repeats_earlier=False,
                        off_brief=False,
                        notes="never got to it",
                    ),
                    _JudgedAct(
                        act_id="act2",
                        covered_points=["a point"],
                        missed_points=[],
                        repeats_earlier=True,
                        off_brief=False,
                        notes="re-introduced the show",
                    ),
                ],
            ),
        )
        report = await run_qc(
            _rundown(("act1", "act2", "act3")),
            {"act1": b"one", "act2": b"two", "act3": b"three"},
            {
                "act1": [_turn("act1", 0, "the plumbing under it")],
                "act2": [_turn("act2", 0, "and what it costs")],
                "act3": [_turn("act3", 0, "one two three four five six seven eight nine ten")],
            },
        )
        assert report.verdict == "warn"
        assert report.flagged_acts == ["act1", "act2", "act3"]
        assert report.summary == "two problems"

    async def test_a_judgement_about_an_act_that_is_not_in_the_run_is_ignored(self, qc_client):
        qc_client(
            transcripts={"act1.mp3": "the plumbing under it", "act2.mp3": "and what it costs"},
            judgement=_Judgement(
                summary="hallucinated an act",
                acts=[
                    _JudgedAct(
                        act_id="act99",
                        covered_points=[],
                        missed_points=["invented"],
                        repeats_earlier=True,
                        off_brief=True,
                        notes="",
                    )
                ],
            ),
        )
        report = await run_qc(
            _rundown(),
            {"act1": b"one", "act2": b"two"},
            {
                "act1": [_turn("act1", 0, "the plumbing under it")],
                "act2": [_turn("act2", 0, "and what it costs")],
            },
        )
        assert report.flagged_acts == []
        assert {v.act_id for v in report.acts} == {"act1", "act2"}

    async def test_only_acts_with_audio_are_verified(self, qc_client):
        client = qc_client(transcripts={"act1.mp3": "the plumbing under it"})
        report = await run_qc(
            _rundown(),
            {"act1": b"one"},
            {"act1": [_turn("act1", 0, "the plumbing under it")]},
        )
        assert [v.act_id for v in report.acts] == ["act1"]
        assert [call["filename"] for call in client.audio.transcriptions.calls] == ["act1.mp3"]

    async def test_the_limiter_does_not_reorder_verdicts(self, qc_client):
        import anyio

        qc_client(transcripts={f"act{i}.mp3": f"text {i}" for i in range(1, 5)})
        report = await run_qc(
            _rundown(("act1", "act2", "act3", "act4")),
            {f"act{i}": b"x" for i in range(1, 5)},
            {f"act{i}": [_turn(f"act{i}", 0, f"text {i}")] for i in range(1, 5)},
            limiter=anyio.CapacityLimiter(2),
        )
        assert [v.act_id for v in report.acts] == ["act1", "act2", "act3", "act4"]
        assert all(v.similarity == 1.0 for v in report.acts)

    async def test_the_models_used_are_recorded_on_the_report(self, qc_client):
        client = qc_client(transcripts={"act1.mp3": "the plumbing under it"})
        report = await run_qc(
            _rundown(("act1",)),
            {"act1": b"one"},
            {"act1": [_turn("act1", 0, "the plumbing under it")]},
            transcribe_model="whisper-next",
            judge_model="judge-next",
        )
        assert report.transcribe_model == "whisper-next"
        assert report.judge_model == "judge-next"
        assert client.audio.transcriptions.calls[0]["model"] == "whisper-next"
        assert client.responses.calls[0]["model"] == "judge-next"

    async def test_the_judge_reads_the_brief_beside_the_transcript(self, qc_client):
        client = qc_client(transcripts={"act1.mp3": "what a listener hears"})
        await run_qc(
            _rundown(("act1",)),
            {"act1": b"one"},
            {"act1": [_turn("act1", 0, "what the model meant")]},
        )
        prompt = client.responses.calls[0]["input"]
        assert "a point" in prompt
        assert "what a listener hears" in prompt
        # The judge grades the rendered audio, never the model's own claim.
        assert "what the model meant" not in prompt


@pytest.mark.unit
class TestActVerdict:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"missed_points": ["a point"]},
            {"repeats_earlier": True},
            {"off_brief": True},
            {"similarity": SIMILARITY_WARN_THRESHOLD - 0.01},
        ],
    )
    def test_each_kind_of_problem_flags_the_act(self, kwargs):
        assert ActVerdict(act_id="act1", **kwargs).flagged

    def test_a_clean_act_is_not_flagged(self):
        assert not ActVerdict(act_id="act1", similarity=1.0, covered_points=["a point"]).flagged

    def test_the_defaults_are_the_models_the_tool_advertises(self):
        assert DEFAULT_TRANSCRIBE_MODEL == "gpt-transcribe"
        assert DEFAULT_JUDGE_MODEL.startswith("gpt-")
