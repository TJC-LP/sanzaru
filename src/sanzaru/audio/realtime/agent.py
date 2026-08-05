# SPDX-License-Identifier: MIT
"""One conversational agent on one Realtime API connection.

The agent is deliberately dumb: it speaks when told to, hears what it is given,
and accepts steering notes. All judgement about *who* speaks next and *what*
they should be nudged toward lives in `producer.py`.

Two mechanics are worth knowing:

- **No VAD.** `turn_detection` is null, so the server never decides to answer on
  its own. Without that, every agent would talk over every other one the moment
  it heard audio.
- **Agents literally hear each other.** The producer feeds one agent's output
  frames into the others' input buffers. Realtime is PCM16/24kHz on both sides,
  so this is a straight copy — the models react to delivery and timing, not just
  to a transcript.

Every `openai.realtime` import is function-local or TYPE_CHECKING-only: the CLI
startup-weight test forbids pulling `openai` in at `sanzaru.cli` import time.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ...config import logger
from ...exceptions import RealtimeAPIError
from .types import HostSpec, RealtimeUsage, pcm_seconds

if TYPE_CHECKING:
    from openai.resources.realtime.realtime import AsyncRealtimeConnection
    from openai.types.realtime.conversation_item_param import ConversationItemParam
    from openai.types.realtime.session_update_event_param import Session


@dataclass(slots=True)
class SpokenTurn:
    """What one `response.create` produced."""

    pcm: bytes
    text: str
    usage: RealtimeUsage
    truncated: bool

    @property
    def seconds(self) -> float:
        return pcm_seconds(self.pcm)


def _usage_from_event(usage: object) -> RealtimeUsage:
    """Flatten the API's nested usage into the shape pricing needs.

    Written defensively with getattr: the fields are all Optional in the SDK, and
    a missing usage block must degrade to zeros rather than kill a recording that
    already cost money.
    """
    if usage is None:
        return RealtimeUsage()
    input_details = getattr(usage, "input_token_details", None)
    output_details = getattr(usage, "output_token_details", None)
    cached_details = getattr(input_details, "cached_tokens_details", None)

    def _int(obj: object, name: str) -> int:
        value = getattr(obj, name, None)
        return int(value) if isinstance(value, int) else 0

    return RealtimeUsage(
        input_tokens=_int(usage, "input_tokens"),
        output_tokens=_int(usage, "output_tokens"),
        input_text_tokens=_int(input_details, "text_tokens"),
        input_audio_tokens=_int(input_details, "audio_tokens"),
        cached_text_tokens=_int(cached_details, "text_tokens"),
        cached_audio_tokens=_int(cached_details, "audio_tokens"),
        output_text_tokens=_int(output_details, "text_tokens"),
        output_audio_tokens=_int(output_details, "audio_tokens"),
    )


class RealtimeAgent:
    """A persona bound to a live Realtime connection."""

    def __init__(
        self,
        spec: HostSpec,
        connection: AsyncRealtimeConnection,
        *,
        model: str,
        max_turn_tokens: int,
        sample_rate: int,
    ) -> None:
        self.spec = spec
        self.model = model
        self._conn = connection
        self._max_turn_tokens = max_turn_tokens
        self._sample_rate = sample_rate

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.name

    async def configure(self, instructions: str) -> None:
        """Install the persona, voice, and output format for this act."""
        session = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "max_output_tokens": self._max_turn_tokens,
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self._sample_rate},
                    # The producer owns the floor; see the module docstring.
                    "turn_detection": None,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": self._sample_rate},
                    "voice": self.spec.voice,
                },
            },
        }
        await self._conn.session.update(session=cast("Session", session))

    async def steer(self, note: str) -> None:
        """Inject a producer note the audience never hears.

        Sent as a system message rather than folded into `instructions` so it
        applies to the next turn only — which is what makes "start landing this"
        actually land instead of colouring the rest of the act.
        """
        item = {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": note}],
        }
        await self._conn.conversation.item.create(item=cast("ConversationItemParam", item))

    async def hear(self, pcm: bytes) -> None:
        """Feed another agent's audio into this one's ears."""
        if not pcm:
            return
        payload = base64.b64encode(pcm).decode()
        await self._conn.input_audio_buffer.append(audio=payload)
        await self._conn.input_audio_buffer.commit()

    async def speak(self) -> SpokenTurn:
        """Take the floor: request a response and collect it to completion."""
        await self._conn.response.create()

        pcm = bytearray()
        text = ""
        usage = RealtimeUsage()
        truncated = False
        completed = False

        async for event in self._conn:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_audio.delta":
                pcm.extend(base64.b64decode(getattr(event, "delta", "")))
            elif event_type == "response.output_audio_transcript.done":
                text = getattr(event, "transcript", "") or ""
            elif event_type == "error":
                raise RealtimeAPIError(f"{self.name}: realtime error: {getattr(event, 'error', event)}")
            elif event_type == "response.done":
                response = getattr(event, "response", None)
                usage = _usage_from_event(getattr(response, "usage", None))
                status = getattr(response, "status", "completed")
                if status == "failed":
                    details = getattr(response, "status_details", None)
                    raise RealtimeAPIError(f"{self.name}: response failed: {getattr(details, 'error', details)}")
                if status == "incomplete":
                    details = getattr(response, "status_details", None)
                    reason = getattr(details, "reason", "unknown")
                    truncated = True
                    logger.warning(
                        "%s: turn cut short (%s) — raise --turn-tokens if this recurs",
                        self.name,
                        reason,
                    )
                completed = True
                break

        if not completed:
            # `AsyncRealtimeConnection.__aiter__` *returns* on ConnectionClosedOK
            # rather than raising, so a graceful mid-act close — including the
            # Realtime API's 60-minute session ceiling — would otherwise look
            # like a successful turn with no audio, no text and zero usage. The
            # act would then be checkpointed as complete, which is both a silent
            # hole in the episode and exactly the zero-audio checkpoint that
            # used to poison every later `--resume`.
            raise RealtimeAPIError(f"{self.name}: realtime stream ended before response.done (session closed?)")

        return SpokenTurn(pcm=bytes(pcm), text=text.strip(), usage=usage, truncated=truncated)
