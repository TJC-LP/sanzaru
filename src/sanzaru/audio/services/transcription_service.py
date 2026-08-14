"""Transcription service - orchestrates transcription operations.

Migrated from mcp-server-whisper v1.1.0 by Richie Caputo (MIT license).
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import anyio
from aioresult import ResultCapture
from openai._types import omit
from openai.types import AudioModel, AudioResponseFormat

from ...config import get_client, logger
from ...infrastructure import FileSystemRepository
from ...storage import get_storage
from .. import AudioProcessor
from ..constants import ENHANCEMENT_PROMPTS, AudioChatModel, EnhancementType
from ..models import ChatResult, TranscriptionResult
from ..providers.base import env_concurrency
from ..windowing import CHUNK_THRESHOLD_SECONDS, Window, merge_window_texts, plan_windows

TRANSCRIBE_CONCURRENCY_ENV = "SANZARU_TRANSCRIBE_MAX_CONCURRENCY"
DEFAULT_TRANSCRIBE_CONCURRENCY = 4
"""Windows in flight at once. Matches the CLI's own `--concurrency` default for
batch transcription, so one long file and four short ones cost the same."""

SHORT_TRANSCRIPT_RATIO = 0.9
"""Below this fraction of the probed duration, a transcript is suspiciously
short and worth a warning."""


def _warn_if_short(filename: str, result: TranscriptionResult, probed_s: float | None) -> None:
    """Flag the silent-truncation signature on a single-request transcription.

    Only fires when the response actually carries coverage information — a
    `duration`, or segments with end timestamps (`verbose_json`). With
    `response_format="text"` there is nothing to compare, which is the other
    half of why long files are windowed rather than checked after the fact.
    """
    if probed_s is None or probed_s <= 0:
        return
    covered = result.duration
    if covered is None and result.segments:
        ends = [end for seg in result.segments if isinstance(end := seg.get("end"), int | float)]
        covered = max(ends) if ends else None
    if covered is None or covered >= probed_s * SHORT_TRANSCRIPT_RATIO:
        return
    logger.warning(
        "%s: transcript covers %.0fs of a %.0fs file - transcription may have stopped early. "
        "Files over %.0fs are windowed automatically; this one was not",
        filename,
        covered,
        probed_s,
        CHUNK_THRESHOLD_SECONDS,
    )


class TranscriptionService:
    """Service for audio transcription operations."""

    def __init__(self):
        """Initialize the transcription service."""
        self.file_repo = FileSystemRepository()

    async def transcribe_audio(
        self,
        filename: str,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using OpenAI's transcription API.

        Args:
        ----
            filename: Name of the audio file.
            model: Transcription model to use.
            response_format: Format of the response.
            prompt: Optional prompt to guide transcription.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            TranscriptionResult: Transcription result with typed fields. `chunked`
            is True when the file was long enough to be transcribed as
            overlapping windows and stitched.

        """
        duration_s = await self._probe_duration(filename)
        if duration_s is not None and duration_s > CHUNK_THRESHOLD_SECONDS:
            return await self._transcribe_windowed(
                filename,
                duration_s,
                model=model,
                prompt=prompt,
            )

        client = get_client()

        # Read audio file via storage backend
        audio_bytes = await self.file_repo.read_audio_file(filename)

        # Transcribe using OpenAI
        transcription = await client.audio.transcriptions.create(
            file=(filename, BytesIO(audio_bytes)),
            model=model,
            response_format=response_format,  # type: ignore[arg-type]  # SDK stubs incomplete for AudioResponseFormat
            prompt=prompt if prompt is not None else omit,
            timestamp_granularities=timestamp_granularities if timestamp_granularities is not None else omit,
        )

        # Convert to TranscriptionResult
        if isinstance(transcription, str):
            result = TranscriptionResult(text=transcription)
        else:
            result = TranscriptionResult(**transcription.model_dump())
        _warn_if_short(filename, result, duration_s)
        return result

    async def _probe_duration(self, filename: str) -> float | None:
        """True duration in seconds, or None if it cannot be determined.

        Never fatal: a probe failure means "transcribe it the way we always
        did", not "refuse the file". Uses pydub through `local_path`, which is
        free on the local backend and a download on Databricks — the same
        trade `list_audio_files` already makes.
        """
        try:
            storage = get_storage()
            async with storage.local_path("audio", filename) as local:
                audio = await AudioProcessor.load_audio_from_path(local)
                return len(audio) / 1000.0
        except Exception as exc:  # noqa: BLE001 - probing is best-effort
            logger.debug("could not probe duration of %s: %s", filename, exc)
            return None

    async def _transcribe_windowed(
        self,
        filename: str,
        duration_s: float,
        *,
        model: AudioModel,
        prompt: str | None,
    ) -> TranscriptionResult:
        """Transcribe a long file as overlapping windows, then stitch (#39).

        Windows are transcribed concurrently under a limiter and read back by
        index — the limiter delays task entry, it does not reorder. A window
        that fails leaves a gap rather than losing the whole transcript, and
        says so in `windows`.
        """
        windows = plan_windows(duration_s)
        logger.info(
            "%s is %.0fs, over the %.0fs single-request threshold - transcribing as %d overlapping windows",
            filename,
            duration_s,
            CHUNK_THRESHOLD_SECONDS,
            len(windows),
        )

        storage = get_storage()
        async with storage.local_path("audio", filename) as local:
            audio = await AudioProcessor.load_audio_from_path(local)
            slices = [await AudioProcessor.export_slice(audio, w.start_s, w.end_s) for w in windows]

        limiter = anyio.CapacityLimiter(env_concurrency(TRANSCRIBE_CONCURRENCY_ENV, DEFAULT_TRANSCRIBE_CONCURRENCY))
        client = get_client()

        async def _one(window: Window, data: bytes) -> str:
            async with limiter:
                try:
                    answer = await client.audio.transcriptions.create(
                        file=(f"{Path(filename).stem}_w{window.index}.mp3", BytesIO(data)),
                        model=model,
                        response_format="text",
                        prompt=prompt if prompt is not None else omit,
                    )
                except Exception as exc:  # noqa: BLE001 - one window must not lose the rest
                    logger.warning("window %d of %s failed to transcribe: %s", window.index, filename, exc)
                    return ""
            return answer if isinstance(answer, str) else getattr(answer, "text", "")

        async with anyio.create_task_group() as tg:
            captures = [ResultCapture.start_soon(tg, _one, w, data) for w, data in zip(windows, slices, strict=True)]
        texts = [c.result() for c in captures]

        failed = [w.index for w, text in zip(windows, texts, strict=True) if not text]
        if failed:
            logger.warning("%s: %d of %d windows returned nothing (%s)", filename, len(failed), len(windows), failed)

        return TranscriptionResult(
            text=merge_window_texts(texts),
            duration=duration_s,
            chunked=True,
            windows=[
                {"index": w.index, "start_s": round(w.start_s, 2), "end_s": round(w.end_s, 2), "text": text}
                for w, text in zip(windows, texts, strict=True)
            ],
        )

    async def chat_with_audio(
        self,
        filename: str,
        model: AudioChatModel = "gpt-4o-audio-preview-2025-06-03",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> ChatResult:
        """Chat with audio using GPT-4o audio models.

        Args:
        ----
            filename: Name of the audio file.
            model: Audio chat model to use.
            system_prompt: Optional system prompt.
            user_prompt: Optional user text prompt.

        Returns:
        -------
            ChatResult: Chat response with typed text field.

        """
        client = get_client()

        # Validate format from filename extension
        ext = Path(filename).suffix.lower().replace(".", "")
        if ext not in ["mp3", "wav"]:
            raise ValueError(f"Expected mp3 or wav extension, but got {ext}")

        # Read audio file via storage backend
        audio_bytes = await self.file_repo.read_audio_file(filename)

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Build messages
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add audio input with optional text prompt
        user_content: list[dict[str, Any]] = [
            {"type": "input_audio", "input_audio": {"data": audio_base64, "format": ext}}
        ]
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})

        messages.append({"role": "user", "content": user_content})

        # Chat with audio using OpenAI
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
        )

        # Extract text from response
        text = response.choices[0].message.content or ""

        return ChatResult(text=text)

    async def transcribe_enhanced(
        self,
        filename: str,
        enhancement_type: EnhancementType,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio with enhancement prompts for different styles.

        Args:
        ----
            filename: Name of the audio file.
            enhancement_type: Type of enhancement (detailed, storytelling, professional, analytical).
            model: Transcription model to use.
            response_format: Format of the response.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            TranscriptionResult: Enhanced transcription result.

        """
        # Use enhancement prompts from constants
        prompt = ENHANCEMENT_PROMPTS.get(enhancement_type, ENHANCEMENT_PROMPTS["detailed"])

        return await self.transcribe_audio(
            filename=filename,
            model=model,
            response_format=response_format,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )
