# SPDX-License-Identifier: MIT
"""`sanzaru audio` — transcription, TTS, chat, and audio processing.

All audio operations are synchronous (single API call or local processing);
there is no job id to poll. TTS is `speak` — the `create` verb is reserved
for async jobs. Requires the [audio] extra (pydub, ffmpeg).
"""

from __future__ import annotations

import time
from typing import Literal, cast

import anyio
import click

from ._io import PathSession, finalize_output, install_overrides, plan_output, read_content_arg, resolve_input
from ._output import EXIT_CONFIG, EXIT_USAGE, aggregate_exit_code, emit, emit_line, error_envelope, success_envelope
from ._runtime import CLIError, _classify, get_state, run_async

_AUDIO_DEP_MESSAGE = "audio commands require optional dependencies — install with: uv pip install 'sanzaru[audio]'"

_VOICES = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]
_TTS_MODELS = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
_TRANSCRIBE_MODELS = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]
_ENHANCEMENTS = ["detailed", "storytelling", "professional", "analytical"]


def _audio_dep_error(exc: ImportError) -> CLIError:
    return CLIError("config", f"{_AUDIO_DEP_MESSAGE} ({exc})", exit_code=EXIT_CONFIG)


@click.group()
def audio() -> None:
    """Audio: transcribe, chat, speak (TTS), convert, compress, files.

    All synchronous — results return in one call, no polling.
    """


@audio.command("transcribe")
@click.argument("files", nargs=-1, required=True)
@click.option("--model", type=click.Choice(_TRANSCRIBE_MODELS), default="gpt-4o-mini-transcribe", show_default=True)
@click.option(
    "--format",
    "response_format",
    type=click.Choice(["text", "json", "verbose_json", "srt", "vtt"]),
    default="text",
    show_default=True,
    help="gpt-4o models support text/json only; use whisper-1 for srt/vtt/verbose_json.",
)
@click.option("--prompt", default=None, help="Guidance prompt (mutually exclusive with --enhance).")
@click.option("--enhance", type=click.Choice(_ENHANCEMENTS), default=None, help="Templated enhancement style.")
@click.option(
    "--timestamps",
    type=click.Choice(["word", "segment"]),
    multiple=True,
    help="whisper-1 + verbose_json only; repeatable.",
)
@click.option("--concurrency", type=click.IntRange(1, 16), default=4, show_default=True)
@run_async("audio.transcribe")
async def audio_transcribe(
    files: tuple[str, ...],
    model: str,
    response_format: str,
    prompt: str | None,
    enhance: str | None,
    timestamps: tuple[str, ...],
    concurrency: int,
) -> int:
    """Transcribe audio file(s). Multiple files fan out concurrently (JSONL output)."""
    try:
        from ..tools import audio as audio_tools
    except ImportError as exc:
        raise _audio_dep_error(exc) from exc
    from openai.types import AudioModel, AudioResponseFormat

    from ..audio.constants import EnhancementType

    if prompt is not None and enhance is not None:
        raise CLIError("usage", "--prompt and --enhance are mutually exclusive", exit_code=EXIT_USAGE)

    session = PathSession()
    names = [resolve_input(session, f, "audio", "FILE") for f in files]
    install_overrides(session)

    granularities = cast("list[Literal['word', 'segment']] | None", list(timestamps) or None)
    single = len(names) == 1
    limiter = anyio.CapacityLimiter(concurrency)
    codes: list[int] = []

    async def worker(index: int, name: str) -> None:
        envelope: dict[str, object]
        async with limiter:
            started = time.monotonic()
            try:
                if enhance is not None:
                    result = await audio_tools.transcribe_with_enhancement(
                        input_file_name=name,
                        enhancement_type=cast(EnhancementType, enhance),
                        model=cast(AudioModel, model),
                        response_format=cast(AudioResponseFormat, response_format),
                        timestamp_granularities=granularities,
                    )
                else:
                    result = await audio_tools.transcribe_audio(
                        input_file_name=name,
                        model=cast(AudioModel, model),
                        response_format=cast(AudioResponseFormat, response_format),
                        prompt=prompt,
                        timestamp_granularities=granularities,
                    )
                code, envelope = (
                    0,
                    success_envelope(
                        "audio.transcribe",
                        result.model_dump(mode="json", exclude_none=True),
                        elapsed_s=time.monotonic() - started,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — batch siblings must keep going
                error = _classify(exc)
                code, envelope = (
                    error.exit_code,
                    error_envelope("audio.transcribe", error.error_type, str(error), extra=error.extra),
                )
        envelope["input"] = {"index": index, "file": name}
        codes.append(code)
        if single:
            emit(envelope)
        else:
            emit_line(envelope)

    async with anyio.create_task_group() as tg:
        for index, name in enumerate(names):
            tg.start_soon(worker, index, name)

    return aggregate_exit_code(codes)


@audio.command("chat")
@click.argument("file")
@click.option("--model", default="gpt-4o-audio-preview", show_default=True, help="Audio chat model.")
@click.option("--system", "system_prompt", default=None, help="System prompt for context.")
@click.option("--prompt", "user_prompt", default=None, help="Question/instructions about the audio.")
@run_async("audio.chat")
async def audio_chat(file: str, model: str, system_prompt: str | None, user_prompt: str | None) -> int:
    """Ask questions about audio content (GPT-4o audio models)."""
    try:
        from ..tools import audio as audio_tools
    except ImportError as exc:
        raise _audio_dep_error(exc) from exc
    from ..audio.constants import AudioChatModel

    session = PathSession()
    name = resolve_input(session, file, "audio", "FILE")
    install_overrides(session)

    result = await audio_tools.chat_with_audio(
        input_file_name=name,
        model=cast(AudioChatModel, model),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    emit(success_envelope("audio.chat", result.model_dump(mode="json")))
    return 0


@audio.command("speak")
@click.argument("text")
@click.option("--model", type=click.Choice(_TTS_MODELS), default="gpt-4o-mini-tts", show_default=True)
@click.option("--voice", type=click.Choice(_VOICES), default="alloy", show_default=True)
@click.option("--instructions", default=None, help="Speech style: tonality, accent, pacing, ...")
@click.option("--speed", type=click.FloatRange(0.25, 4.0), default=1.0, show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (default: media dir).")
@click.pass_context
@run_async("audio.speak")
async def audio_speak(
    ctx: click.Context,
    text: str,
    model: str,
    voice: str,
    instructions: str | None,
    speed: float,
    output: str | None,
) -> int:
    """Text-to-speech. TEXT is inline, @file, or - (stdin); long text auto-chunks."""
    try:
        from ..tools import audio as audio_tools
    except ImportError as exc:
        raise _audio_dep_error(exc) from exc
    from openai.types.audio.speech_model import SpeechModel

    from ..audio.constants import TTSVoice

    state = get_state(ctx)
    text_content = read_content_arg(text, "TEXT")
    session = PathSession()
    plan = plan_output(session, output, "audio", quiet=state.quiet)
    install_overrides(session)

    started = time.monotonic()
    result = await audio_tools.create_audio(
        text_prompt=text_content,
        model=cast(SpeechModel, model),
        voice=cast(TTSVoice, voice),
        instructions=instructions,
        speed=speed,
        output_file_name=plan.filename,
    )
    final_path = await finalize_output(session, plan, result.output_file)
    payload: dict[str, object] = result.model_dump(mode="json")
    payload["file"] = {"path": final_path}
    emit(success_envelope("audio.speak", payload, elapsed_s=time.monotonic() - started))
    return 0


@audio.command("convert")
@click.argument("file")
@click.option("--to", "target_format", type=click.Choice(["mp3", "wav"]), default="mp3", show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (default: alongside input).")
@click.pass_context
@run_async("audio.convert")
async def audio_convert(ctx: click.Context, file: str, target_format: str, output: str | None) -> int:
    """Convert audio to mp3/wav (GPT-4o-compatible formats)."""
    try:
        from ..tools import audio as audio_tools
    except ImportError as exc:
        raise _audio_dep_error(exc) from exc

    state = get_state(ctx)
    session = PathSession()
    name = resolve_input(session, file, "audio", "FILE")
    plan = plan_output(session, output, "audio", quiet=state.quiet)
    install_overrides(session)

    result = await audio_tools.convert_audio(
        input_file_name=name,
        target_format=cast(Literal["mp3", "wav"], target_format),
        output_file_name=plan.filename,
    )
    final_path = await finalize_output(session, plan, result.output_file)
    payload: dict[str, object] = result.model_dump(mode="json")
    payload["file"] = {"path": final_path}
    emit(success_envelope("audio.convert", payload))
    return 0


@audio.command("compress")
@click.argument("file")
@click.option("--max-mb", type=click.IntRange(1, 1000), default=25, show_default=True)
@click.option("-o", "--output", default=None, help="Output file or directory (default: alongside input).")
@click.pass_context
@run_async("audio.compress")
async def audio_compress(ctx: click.Context, file: str, max_mb: int, output: str | None) -> int:
    """Compress audio under a size budget (e.g. the 25MB transcription limit)."""
    try:
        from ..tools import audio as audio_tools
    except ImportError as exc:
        raise _audio_dep_error(exc) from exc

    state = get_state(ctx)
    session = PathSession()
    name = resolve_input(session, file, "audio", "FILE")
    plan = plan_output(session, output, "audio", quiet=state.quiet)
    install_overrides(session)

    result = await audio_tools.compress_audio(input_file_name=name, max_mb=max_mb, output_file_name=plan.filename)
    final_path = await finalize_output(session, plan, result.output_file)
    payload: dict[str, object] = result.model_dump(mode="json")
    payload["file"] = {"path": final_path}
    emit(success_envelope("audio.compress", payload))
    return 0


@audio.command("files")
@click.option("--pattern", default=None, help="Regex filter on file names.")
@click.option("--format", "format_filter", default=None, help='Filter by format, e.g. "mp3".')
@click.option("--min-size", type=int, default=None, help="Minimum size in bytes.")
@click.option("--max-size", type=int, default=None, help="Maximum size in bytes.")
@click.option("--min-duration", type=float, default=None, help="Minimum duration in seconds.")
@click.option("--max-duration", type=float, default=None, help="Maximum duration in seconds.")
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["name", "size", "duration", "modified_time", "format"]),
    default="name",
    show_default=True,
)
@click.option("--reverse", is_flag=True, help="Descending order.")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--latest", is_flag=True, help="Print only the most recently modified file.")
@run_async("audio.files")
async def audio_files(
    pattern: str | None,
    format_filter: str | None,
    min_size: int | None,
    max_size: int | None,
    min_duration: float | None,
    max_duration: float | None,
    sort_by: str,
    reverse: bool,
    limit: int,
    latest: bool,
) -> int:
    """List audio files in the media dir with metadata + model support."""
    try:
        from ..tools import audio as audio_tools
    except ImportError as exc:
        raise _audio_dep_error(exc) from exc
    from ..audio.constants import SortBy

    if latest:
        newest = await audio_tools.get_latest_audio()
        emit(success_envelope("audio.files", newest.model_dump(mode="json", exclude_none=True)))
        return 0

    results = await audio_tools.list_audio_files(
        pattern=pattern,
        min_size_bytes=min_size,
        max_size_bytes=max_size,
        min_duration_seconds=min_duration,
        max_duration_seconds=max_duration,
        format=format_filter,
        sort_by=SortBy(sort_by),
        reverse=reverse,
    )
    payload = {
        "data": [item.model_dump(mode="json", exclude_none=True) for item in results[:limit]],
        "total": len(results),
    }
    emit(success_envelope("audio.files", payload))
    return 0
