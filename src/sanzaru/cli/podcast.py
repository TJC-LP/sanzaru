# SPDX-License-Identifier: MIT
"""`sanzaru podcast` — three ways to make an episode.

    rundown   plan only: a premise becomes an editable act-by-act JSON
    simulate  record realtime agents actually talking to each other
    generate  speak a script you already wrote (multi-voice TTS)

`rundown` is split out from `simulate` on purpose: planning is cheap and
recording is not, so the plan should be inspectable and hand-editable before
anyone spends realtime money on it.
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import click

from ._io import PathSession, finalize_output, install_overrides, plan_output, read_content_arg
from ._output import EXIT_CONFIG, EXIT_PARTIAL, EXIT_USAGE, emit, note, success_envelope
from ._runtime import CLIError, _classify, find_in_group, get_state, run_async
from .audio import _ELEVENLABS_MODELS, _PROVIDERS, _TTS_MODELS, resolve_tts_model

if TYPE_CHECKING:
    # Runtime import would pull openai/pydantic in at `sanzaru.cli` import time.
    from ..tools.simulate_podcast import SimulatedPodcastResult

_AUDIO_DEP_MESSAGE = "podcast generation requires optional dependencies — install with: uv pip install 'sanzaru[audio]'"

# Literal lists rather than imports: importing sanzaru.cli must not pull in
# openai/pydantic (tests/cli/test_root.py guards the startup weight).
_REALTIME_MODELS = ["gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime-2", "gpt-realtime"]
_REALTIME_VOICES = ["marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]


def _parse_host(value: str) -> dict[str, str]:
    """Parse a `--host name[:voice[:persona]]` spec.

    Split at most twice so a persona can contain colons, which it usually does
    the moment anyone writes a real one.
    """
    parts = value.split(":", 2)
    name = parts[0].strip()
    if not name:
        raise CLIError("usage", f"--host {value!r}: name is required (name[:voice[:persona]])", exit_code=EXIT_USAGE)
    voice = parts[1].strip() if len(parts) > 1 else ""
    if voice and voice not in _REALTIME_VOICES:
        raise CLIError(
            "usage",
            f"--host {value!r}: {voice!r} is not a realtime voice; choose one of: {', '.join(_REALTIME_VOICES)}",
            exit_code=EXIT_USAGE,
        )
    persona = parts[2].strip() if len(parts) > 2 else ""
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") or f"host{abs(hash(name)) % 1000}"
    return {"id": slug, "name": name, "voice": voice, "persona": persona}


def _progress_printer(quiet: bool) -> Callable[[str], None]:
    """Greppable, line-based stderr progress, matching the rest of the CLI.

    Every line carries elapsed wall clock: during a blocking multi-minute
    recording that is the only signal the run is alive.
    """
    start = time.monotonic()

    def printer(message: str) -> None:
        if not quiet:
            note(f"{message} t={int(time.monotonic() - start)}s")

    return printer


def _load_json_arg(value: str, arg_name: str) -> dict[str, object]:
    text = read_content_arg(value, arg_name)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLIError("usage", f"{arg_name} is not valid JSON: {exc}", exit_code=EXIT_USAGE) from exc
    if not isinstance(parsed, dict):
        raise CLIError("usage", f"{arg_name} must be a JSON object", exit_code=EXIT_USAGE)
    return cast("dict[str, object]", parsed)


@click.group()
def podcast() -> None:
    """Podcast generation: plan a rundown, simulate a real conversation, or speak a script."""


@podcast.command("generate")
@click.argument("script")
@click.option("--provider", type=click.Choice(_PROVIDERS), default="openai", show_default=True)
@click.option(
    "--model",
    default=None,
    help=f"openai: {', '.join(_TTS_MODELS)} [default: gpt-4o-mini-tts]  |  "
    f"elevenlabs: {', '.join(_ELEVENLABS_MODELS)} [default: eleven_v3]",
)
@click.option(
    "--render-mode",
    type=click.Choice(["segments", "dialogue"]),
    default=None,
    help="segments (default): one request per turn, joined with silence gaps. "
    "dialogue: consecutive ElevenLabs eleven_v3 turns go out together so the model paces them. "
    "Overrides config.render_mode.",
)
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("podcast.generate")
async def podcast_generate(
    ctx: click.Context,
    script: str,
    provider: str,
    model: str | None,
    render_mode: str | None,
    output: str | None,
) -> int:
    """Render a podcast from a PodcastScript JSON. SCRIPT is inline JSON, @file, or - (stdin).

    \b
    Script shape:
      {"title": str,
       "speakers": [{id, name, voice, speed, instructions,     (1-4 speakers)
                     provider?, model?, voice_settings?}],
       "segments": [{speaker, text, pause_after?, speed_override?,
                     instruction_override?}],
       "config": {"default_pause_ms": int, "normalize_loudness": bool,
                  "output_format": "mp3"|"wav",                 (all three REQUIRED)
                  "intro_silence_ms"?, "outro_silence_ms"?, "output_bitrate"?,
                  "provider"?, "max_concurrency"?,
                  "render_mode"?: "segments"|"dialogue", "dialogue_stability"?}}
    \b
    render_mode="dialogue" batches consecutive eleven_v3 turns into one request so
    the model paces the exchange itself — noticeably more natural than fixed gaps.
    Turns that cannot join a run (OpenAI speakers, other models, lone turns,
    stretches in one voice, turns over the 2000-char request budget) still render per segment, so
    mixed episodes keep working. Inside a dialogue run, pause_after and
    per-speaker voice_settings do not apply.
    \b
    The trade: a run is ONE request, so it is all-or-nothing on retry — fixing a
    single bad line re-spends every character in the batch, where segments mode
    retries one turn. ElevenLabs draws quota per character submitted and inline
    audio tags count too, so the expressive mode is the costly one to redo. The
    budget is per request, not per turn: a turn left single-voice after a run
    splits quietly renders as a plain segment, even well under the ceiling. See
    docs/cli.md (Render modes) for the full rules.
    \b
    Provider precedence: speaker.provider > config.provider > --provider. Speakers
    may differ, so one episode can mix OpenAI and ElevenLabs voices. ElevenLabs
    speakers need a voice id, cap speed at 0.7-1.2 (eleven_v3: unsupported), and
    ignore `instructions` — use inline audio tags like [whispers] in the text.
    \b
    Segments TTS in parallel internally, bounded per provider; the transcript is
    included in the result envelope.
    Example: sanzaru podcast generate - < episode.json -o ep.mp3
    """
    try:
        from ..tools import podcast as podcast_tools
    except ImportError as exc:
        raise CLIError("config", f"{_AUDIO_DEP_MESSAGE} ({exc})", exit_code=EXIT_CONFIG) from exc
    from openai.types.audio.speech_model import SpeechModel

    from ..audio.constants import TTSProviderName
    from ..tools.podcast import PodcastScript

    resolved_model = resolve_tts_model(provider, model)

    state = get_state(ctx)
    script_text = read_content_arg(script, "SCRIPT")
    try:
        parsed = json.loads(script_text)
    except json.JSONDecodeError as exc:
        raise CLIError("usage", f"SCRIPT is not valid JSON: {exc}", exit_code=EXIT_USAGE) from exc
    if not isinstance(parsed, dict):
        raise CLIError("usage", "SCRIPT must be a JSON object (PodcastScript)", exit_code=EXIT_USAGE)

    # The flag is an override, so it wins over config.render_mode. Only merged
    # into a config that is already there: fabricating one would mask
    # _validate_script's "missing required field: 'config'" with whatever
    # unrelated error it hits next.
    if render_mode is not None and "config" in parsed:
        if not isinstance(parsed["config"], dict):
            raise CLIError("usage", "SCRIPT 'config' must be a JSON object", exit_code=EXIT_USAGE)
        parsed["config"]["render_mode"] = render_mode

    session = PathSession()
    plan = plan_output(session, output, "audio", quiet=state.quiet)
    install_overrides(session)

    started = time.monotonic()
    result = await podcast_tools.generate_podcast(
        cast(PodcastScript, parsed),
        model=cast(SpeechModel, resolved_model),
        provider=cast(TTSProviderName, provider),
    )
    # generate_podcast has no output-filename parameter — it auto-names inside
    # the (possibly overridden) audio dir. Honor `-o file.mp3` by renaming after.
    final_path = await finalize_output(session, plan, result.output_file)
    if plan.filename is not None and plan.filename != result.output_file:
        import pathlib
        import shutil

        import anyio

        target = pathlib.Path(final_path).with_name(plan.filename)
        await anyio.to_thread.run_sync(shutil.move, final_path, str(target))
        final_path = str(target)
    payload: dict[str, object] = result.model_dump(mode="json")
    payload["file"] = {"path": final_path}
    emit(success_envelope("podcast.generate", payload, elapsed_s=time.monotonic() - started))
    return 0


@podcast.command("rundown")
@click.argument("premise")
@click.option("--acts", type=int, default=4, show_default=True, help="How many acts to split the episode into.")
@click.option(
    "-m",
    "--target-minutes",
    type=float,
    default=12.0,
    show_default=True,
    help="Total episode length; divided evenly across acts.",
)
@click.option("--title", default=None, help="Episode title (default: the planner picks one).")
@click.option("--style", default=None, help="One line of tone/format direction applied to every act.")
@click.option(
    "--host",
    "hosts",
    multiple=True,
    metavar="NAME[:VOICE[:PERSONA]]",
    help="Fix a host instead of letting the planner invent one. Repeatable.",
)
@click.option(
    "--turn-seconds",
    type=float,
    default=15.0,
    show_default=True,
    help="Target upper bound per turn; sets each act's turn budget.",
)
@click.option("--model", default=None, help="Planner text model [default: gpt-5.5]. Not a realtime model.")
@click.option("-o", "--output", default=None, help="Write the rundown JSON here (also printed in the envelope).")
@click.pass_context
@run_async("podcast.rundown")
async def podcast_rundown(
    ctx: click.Context,
    premise: str,
    acts: int,
    target_minutes: float,
    title: str | None,
    style: str | None,
    hosts: tuple[str, ...],
    turn_seconds: float,
    model: str | None,
    output: str | None,
) -> int:
    """Plan an episode. PREMISE is inline text, @file, or - (stdin).

    \b
    Pre-production for `podcast simulate`. Acts record in parallel, in separate
    sessions that cannot hear each other, so every act carries `prior_context`
    (what earlier acts already covered) and `handoff` (where to leave off).
    That wiring is the whole point of planning first.
    \b
    Costs one text-model call and no audio. Edit the JSON freely — rewrite
    talking points, retune target_seconds/max_turns, swap voices — then feed it
    straight to `podcast simulate`.
    \b
    Example:
      sanzaru podcast rundown "why TTS providers drop sentence tails" \\
        --acts 3 -m 6 --host "Avery::You host and translate jargon." \\
        --host "Rory:cedar:You chased the bug. Dry, specific." -o rundown.json
      sanzaru podcast simulate @rundown.json --dry-run
    """
    try:
        from ..audio.realtime.rundown import RundownRequest, plan_rundown
    except ImportError as exc:
        raise CLIError("config", f"{_AUDIO_DEP_MESSAGE} ({exc})", exit_code=EXIT_CONFIG) from exc
    from ..audio.realtime.types import HostSpec

    state = get_state(ctx)
    premise_text = read_content_arg(premise, "PREMISE")
    host_specs = [HostSpec.model_validate(_parse_host(h)) for h in hosts]

    request = RundownRequest(
        premise=premise_text,
        acts=acts,
        target_minutes=target_minutes,
        title=title,
        style=style,
        hosts=host_specs,
        turn_seconds=turn_seconds,
        **({"model": model} if model else {}),
    )

    started = time.monotonic()
    if not state.quiet:
        note(f"planning {acts} acts / {target_minutes:.0f} min with {request.model}")
    rundown = await plan_rundown(request)

    payload: dict[str, object] = rundown.model_dump(mode="json")
    if output is not None:
        # The file is the pure rundown so it can be piped straight back into
        # `simulate` (and hand-edited) — the envelope alone carries `file`.
        target = pathlib.Path(output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload = dict(payload)
        payload["file"] = {"path": str(target.resolve())}

    if not state.quiet:
        # Local import: `sanzaru.cli` must stay free of pydantic at import time
        # (tests/cli/test_root.py guards the startup weight).
        from ..audio.realtime.types import extension_cap

        for act in rundown.acts:
            note(
                f"  {act.id}: {act.title} — {act.target_seconds:.0f}s, "
                f"~{act.max_turns} planned turns (up to {extension_cap(act.max_turns)})"
            )
    emit(success_envelope("podcast.rundown", payload, elapsed_s=time.monotonic() - started))
    return 0


@podcast.command("simulate")
@click.argument("brief", required=False)
@click.option("-p", "--premise", default=None, help="Plan a rundown from this premise (inline text, @file, or -).")
@click.option("--acts", type=int, default=None, help="Act count when planning from a premise [default: 4].")
@click.option("-m", "--target-minutes", type=float, default=None, help="Episode length in minutes [default: 12].")
@click.option("--title", default=None, help="Episode title.")
@click.option("--style", default=None, help="Tone/format direction applied to every act.")
@click.option(
    "--host",
    "hosts",
    multiple=True,
    metavar="NAME[:VOICE[:PERSONA]]",
    help=f"A participant. Voices: {', '.join(_REALTIME_VOICES)}. Repeatable.",
)
@click.option(
    "--model",
    type=click.Choice(_REALTIME_MODELS),
    default=None,
    help="Realtime model [default: gpt-realtime-2.1]. -mini is ~3x cheaper and noticeably faster.",
)
@click.option("--planner-model", default=None, help="Text model for pre-production [default: gpt-5.5].")
@click.option("--turn-seconds", type=float, default=None, help="Target upper bound per turn [default: 15].")
@click.option(
    "--turn-tokens",
    type=int,
    default=None,
    help="Hard per-turn output cap [default: derived from --turn-seconds]. Raise if turns get cut short.",
)
@click.option(
    "--max-cost",
    type=float,
    default=None,
    metavar="USD",
    help="Abort once spend crosses this. Finished acts stay on disk.",
)
@click.option(
    "--max-sessions", type=int, default=None, help="Concurrent realtime sessions across all acts [default: 6]."
)
@click.option(
    "--resume",
    "resume_id",
    default=None,
    metavar="RUN_ID",
    help="Re-use checkpointed acts from an earlier run and record only what is missing.",
)
@click.option(
    "--run-id",
    "run_id_opt",
    default=None,
    metavar="RUN_ID",
    help="Record under this run id instead of a minted one, so --resume is predictable. "
    'A top-level "run_id" in a rundown BRIEF does the same thing.',
)
@click.option(
    "--stems/--no-stems", default=False, show_default=True, help="Also write one time-aligned track per host."
)
@click.option(
    "--qc/--no-qc",
    default=True,
    show_default=True,
    help="Transcribe the rendered audio and judge it against the rundown.",
)
@click.option("--qc-retry", is_flag=True, help="Re-record acts QC flags, once. Only the flagged acts are re-run.")
@click.option("--dry-run", is_flag=True, help="Plan and project turns, duration, tokens and cost. Records nothing.")
@click.option("--act-gap", type=int, default=None, metavar="MS", help="Silence between acts [default: 700].")
# Like every other option here, these default to None rather than to their
# value: on `--resume` the tool restores from the run manifest whatever the
# caller did not pass, and click filling in "mp3" would be indistinguishable
# from the user asking for mp3.
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["mp3", "wav"]),
    default=None,
    help="Container for the episode [default: mp3].",
)
@click.option("--bitrate", default=None, help="MP3 bitrate [default: 192k]. Ignored for WAV.")
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("podcast.simulate")
async def podcast_simulate(
    ctx: click.Context,
    brief: str | None,
    premise: str | None,
    acts: int | None,
    target_minutes: float | None,
    title: str | None,
    style: str | None,
    hosts: tuple[str, ...],
    model: str | None,
    planner_model: str | None,
    turn_seconds: float | None,
    turn_tokens: int | None,
    max_cost: float | None,
    max_sessions: int | None,
    resume_id: str | None,
    run_id_opt: str | None,
    stems: bool,
    qc: bool,
    qc_retry: bool,
    dry_run: bool,
    act_gap: int | None,
    output_format: str | None,
    bitrate: str | None,
    output: str | None,
) -> int:
    """Record a podcast by having realtime agents actually talk to each other.

    \b
    BRIEF is a rundown (from `podcast rundown`) or a full SimulationBrief, as
    inline JSON, @file, or - (stdin). Or skip it and pass --premise to plan and
    record in one go. Flags override whatever the BRIEF says.
    \b
    Unlike `podcast generate`, nothing here is scripted: each host is a
    gpt-realtime session with a persona, and one host's audio is played into the
    others' ears, so they respond to delivery and timing. The conversation is an
    output, not an input — the transcript comes back in the envelope.
    \b
    Acts record in parallel (a 30-minute episode lands in ~1-2 minutes of wall
    clock) and each is checkpointed to the audio dir the moment it finishes, so
    an interrupt, a timeout, or a --max-cost abort never throws away audio you
    already paid for. The run id is printed on stderr before recording starts:
      sanzaru podcast simulate --resume RUN_ID
    picks up exactly where it stopped and records only the missing acts.
    \b
    Cost is real — this is the most expensive thing sanzaru does. Always
    --dry-run first; it plans, projects tokens and dollars, and spends nothing.
    \b
    Exit codes: 0 ok · 2 usage · 3 config · 6 cost ceiling hit (partial run,
    resumable) · 1 other runtime failure.
    \b
    Examples:
      sanzaru podcast simulate -p "the plumbing under generative media" --dry-run
      sanzaru podcast simulate @rundown.json --model gpt-realtime-2.1-mini \\
        --max-cost 2.00 --stems -o ./out/ep1.mp3
      sanzaru podcast simulate --resume 6f1a9c02
    """
    try:
        from ..tools import simulate_podcast as sim
    except ImportError as exc:
        raise CLIError("config", f"{_AUDIO_DEP_MESSAGE} ({exc})", exit_code=EXIT_CONFIG) from exc
    from ..exceptions import CostCeilingError

    state = get_state(ctx)

    payload: dict[str, object] = {}
    if brief is not None:
        parsed = _load_json_arg(brief, "BRIEF")
        # A bare rundown and a full brief are both reasonable things to pipe in,
        # and they overlap: both carry `premise`. `acts` is what separates them —
        # a list of act briefs in a rundown, an integer count in a brief.
        is_rundown = "rundown" not in parsed and isinstance(parsed.get("acts"), list)
        if is_rundown:
            # A top-level "run_id" in a rundown file is caller intent, and losing
            # it strands paid audio under a minted id that only ever appeared on
            # stderr. Lift it out before the rundown is validated (Rundown has no
            # such field) and let it seed the brief below.
            lifted_run_id = parsed.pop("run_id", None)
            payload = {"rundown": parsed}
            # Anything present is passed through, including the wrong type and
            # the empty string: `RunId` rejects those as a usage error, which is
            # the whole point — dropping a malformed run id silently is the
            # defect being fixed, and "" is malformed rather than absent.
            if lifted_run_id is not None:
                payload["run_id"] = lifted_run_id
        else:
            payload = parsed
    if premise is not None:
        payload["premise"] = read_content_arg(premise, "--premise")
    if hosts:
        payload["hosts"] = [_parse_host(h) for h in hosts]

    for key, value in (
        ("acts", acts),
        ("target_minutes", target_minutes),
        ("title", title),
        ("style", style),
        ("model", model),
        ("planner_model", planner_model),
        ("turn_seconds", turn_seconds),
        ("turn_tokens", turn_tokens),
        ("max_cost_usd", max_cost),
        ("max_concurrent_sessions", max_sessions),
        ("act_gap_ms", act_gap),
        ("output_format", output_format),
        ("output_bitrate", bitrate),
    ):
        if value is not None:
            payload[key] = value

    # Flags, unlike the options above, always carry a value the user can see in
    # --help, so passing them through unconditionally is honest intent.
    payload["stems"] = stems
    payload["qc"] = qc
    payload["qc_retry"] = qc_retry
    payload["dry_run"] = dry_run
    # `--run-id` beating a BRIEF's own id is a documented override. Disagreeing
    # with the run being RESUMED is not an override, it is two answers to "which
    # run is this?", and silently picking one is how the stranded-audio defect
    # happened. Checked against the resolved payload, so a BRIEF carrying
    # `resume: true` counts the same as the flag.
    resume_target = resume_id or (payload.get("run_id") if payload.get("resume") else None)
    if resume_target:
        conflicting = sorted(
            {i for i in (run_id_opt, payload.get("run_id")) if isinstance(i, str) and i and i != resume_target}
        )
        if conflicting:
            raise CLIError(
                "usage",
                f"{', '.join(repr(i) for i in conflicting)} names a different run than the one being "
                f"resumed ({resume_target!r}); a resume already carries the id of the run to continue",
                exit_code=EXIT_USAGE,
            )
    if run_id_opt:
        # Explicit flag beats a run_id lifted from the BRIEF.
        payload["run_id"] = run_id_opt
    if resume_id:
        payload["resume"] = True
        payload["run_id"] = resume_id

    try:
        simulation = sim.SimulationBrief.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — pydantic validation is a usage error here
        raise CLIError("usage", f"BRIEF is not a valid simulation request: {exc}", exit_code=EXIT_USAGE) from exc

    if not simulation.premise.strip() and simulation.rundown is None and not simulation.resume:
        raise CLIError(
            "usage",
            "nothing to record: pass a rundown/brief, --premise, or --resume RUN_ID",
            exit_code=EXIT_USAGE,
        )

    started = time.monotonic()
    progress = _progress_printer(state.quiet)

    if dry_run:
        # No output plan and no storage override: a dry run must not create
        # directories or touch the media dir at all.
        result = await sim.simulate_podcast(simulation, on_progress=progress)
        _note_dry_run(result, state.quiet)
        emit(success_envelope("podcast.simulate", result.model_dump(mode="json"), elapsed_s=time.monotonic() - started))
        return 0

    session = PathSession()
    plan = plan_output(session, output, "audio", quiet=state.quiet)
    install_overrides(session)
    if plan.filename is not None:
        simulation = simulation.model_copy(update={"filename": plan.filename})

    # Print the run id before anything is recorded: if this is interrupted, that
    # id is the difference between resuming and paying twice.
    run_id = simulation.run_id or sim.new_run_id()
    simulation = simulation.model_copy(update={"run_id": run_id})
    if not state.quiet:
        note(f"run {run_id} — resume with: sanzaru podcast simulate --resume {run_id}")

    try:
        result = await sim.simulate_podcast(simulation, on_progress=progress)
    except Exception as raised:  # noqa: BLE001 — reclassified below, never swallowed
        # Acts record inside an anyio task group, so the ceiling arrives wrapped
        # in an ExceptionGroup — and if several acts trip it in the same
        # scheduling window, one error per act.
        ceiling = find_in_group(raised, CostCeilingError)
        if ceiling is None:
            # Any other failure — a stalled turn, a dropped session — still
            # leaves the finished acts checkpointed. The run id is the only way
            # back to them, so it goes in the envelope rather than in a stderr
            # line the caller has already scrolled past.
            other = _classify(raised)
            other.resume = f"sanzaru podcast simulate --resume {run_id}"
            other.extra = {**(other.extra or {}), "run_id": run_id}
            raise other from raised
        # A resumed run restores this ceiling from the manifest and replays the
        # spend already on disk against it, so a bare `--resume` would abort at
        # the same total having paid to re-record the same acts. The command we
        # print has to be one that can actually finish.
        resume_command = f"sanzaru podcast simulate --resume {run_id}"
        if ceiling.suggested_limit_usd is not None:
            resume_command += f" --max-cost {ceiling.suggested_limit_usd:g}"
        raise CLIError(
            "cost_limit",
            f"{ceiling} — {len(ceiling.completed_acts)} act(s) checkpointed and safe. The run's ceiling is "
            f"restored on resume, so raise it to finish: {resume_command}",
            exit_code=EXIT_PARTIAL,
            resume=resume_command,
            extra={
                "spent_usd": round(ceiling.spent_usd, 4),
                "limit_usd": ceiling.limit_usd,
                "suggested_limit_usd": ceiling.suggested_limit_usd,
                "completed_acts": ceiling.completed_acts,
                "run_id": run_id,
            },
        ) from ceiling

    final_path = await finalize_output(session, plan, result.output_file)
    envelope_payload: dict[str, object] = result.model_dump(mode="json")
    envelope_payload["file"] = {"path": final_path}
    _note_result(result, state.quiet)
    emit(success_envelope("podcast.simulate", envelope_payload, elapsed_s=time.monotonic() - started))
    return 0


def _note_dry_run(result: SimulatedPodcastResult, quiet: bool) -> None:
    """Human-readable projection on stderr; the envelope carries the numbers."""
    if quiet:
        return
    # "up to N turns" was true when max_turns was a hard stop; an act now
    # extends past it to reach its target, and the dry run is where a caller
    # calibrates --max-cost.
    from ..audio.realtime.types import extension_cap

    note(f"dry run — {result.title!r}: {len(result.acts)} acts, ~{result.turn_count} planned turns")
    for act in result.acts:
        note(
            f"  {act.act_id}: {act.title} — {act.seconds:.0f}s, "
            f"~{act.turns} planned turns (up to {extension_cap(act.turns)})"
        )
    usage = result.cost.usage
    note(
        f"projected ~{result.duration_seconds / 60:.0f} min audio, "
        f"{usage.input_tokens:,} input / {usage.output_tokens:,} output tokens"
    )
    if result.cost.usd is not None:
        # Projected from rates measured in a live spike, not from a price quote.
        note(f"projected cost ~${result.cost.usd:.2f} (estimate, not a quote)")
    else:
        unpriced = ", ".join(result.cost.unpriced_models)
        note(f"no price known for {unpriced or 'this model'} — set SANZARU_REALTIME_PRICE_* to estimate cost")
    note("nothing was recorded; drop --dry-run to record")


def _note_result(result: SimulatedPodcastResult, quiet: bool) -> None:
    """One line per fact on stderr; the envelope carries the detail."""
    if quiet:
        return
    note(
        f"episode {result.run_id}: {len(result.acts)} acts, {result.turn_count} turns, "
        f"{result.duration_seconds / 60:.1f} min"
    )
    truncated = sum(act.truncated_turns for act in result.acts)
    if truncated:
        note(f"{truncated} turn(s) hit the token cap and were cut short — raise --turn-tokens")
    if result.cost.usd is not None:
        note(f"spend ${result.cost.usd:.2f}")
    if result.stems:
        note(f"stems: {', '.join(result.stems.values())}")
    if result.qc is not None:
        if result.qc.flagged_acts:
            note(
                f"qc {result.qc.verdict}: {', '.join(result.qc.flagged_acts)} — "
                "see result.qc for why (--qc-retry re-records just those)"
            )
        else:
            note(f"qc {result.qc.verdict}")
