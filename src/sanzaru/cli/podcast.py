# SPDX-License-Identifier: MIT
"""`sanzaru podcast` — multi-voice podcast rendering from a structured script."""

from __future__ import annotations

import json
import time
from typing import cast

import click

from ._io import PathSession, finalize_output, install_overrides, plan_output, read_content_arg
from ._output import EXIT_CONFIG, EXIT_USAGE, emit, success_envelope
from ._runtime import CLIError, get_state, run_async
from .audio import _ELEVENLABS_MODELS, _PROVIDERS, _TTS_MODELS, resolve_tts_model

_AUDIO_DEP_MESSAGE = "podcast generation requires optional dependencies — install with: uv pip install 'sanzaru[audio]'"


@click.group()
def podcast() -> None:
    """Podcast generation (multi-voice TTS, stitched with pauses)."""


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
