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

_AUDIO_DEP_MESSAGE = "podcast generation requires optional dependencies — install with: uv pip install 'sanzaru[audio]'"


@click.group()
def podcast() -> None:
    """Podcast generation (multi-voice TTS, stitched with pauses)."""


@podcast.command("generate")
@click.argument("script")
@click.option(
    "--model", type=click.Choice(["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]), default="gpt-4o-mini-tts", show_default=True
)
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("podcast.generate")
async def podcast_generate(ctx: click.Context, script: str, model: str, output: str | None) -> int:
    """Render a podcast from a PodcastScript JSON. SCRIPT is inline JSON, @file, or - (stdin).

    \b
    Script shape:
      {"title": str,
       "speakers": [{id, name, voice, speed, instructions}],   (1-4 speakers)
       "segments": [{speaker, text, pause_after?, speed_override?}],
       "config": {"default_pause_ms": int, "normalize_loudness": bool,
                  "output_format": "mp3"|"wav",                 (all three REQUIRED)
                  "intro_silence_ms"?, "outro_silence_ms"?, "output_bitrate"?}}
    Segments TTS in parallel internally; the transcript is included in the
    result envelope. Example: sanzaru podcast generate - < episode.json -o ep.mp3
    """
    try:
        from ..tools import podcast as podcast_tools
    except ImportError as exc:
        raise CLIError("config", f"{_AUDIO_DEP_MESSAGE} ({exc})", exit_code=EXIT_CONFIG) from exc
    from openai.types.audio.speech_model import SpeechModel

    from ..tools.podcast import PodcastScript

    state = get_state(ctx)
    script_text = read_content_arg(script, "SCRIPT")
    try:
        parsed = json.loads(script_text)
    except json.JSONDecodeError as exc:
        raise CLIError("usage", f"SCRIPT is not valid JSON: {exc}", exit_code=EXIT_USAGE) from exc
    if not isinstance(parsed, dict):
        raise CLIError("usage", "SCRIPT must be a JSON object (PodcastScript)", exit_code=EXIT_USAGE)

    session = PathSession()
    plan = plan_output(session, output, "audio", quiet=state.quiet)
    install_overrides(session)

    started = time.monotonic()
    result = await podcast_tools.generate_podcast(cast(PodcastScript, parsed), model=cast(SpeechModel, model))
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
