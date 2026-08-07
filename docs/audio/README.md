# Sanzaru Audio Feature

Audio processing capabilities for sanzaru via OpenAI's Whisper and GPT-4o Audio APIs.

## Installation

```bash
# Install sanzaru with audio support
uv add "sanzaru[audio]"

# Or install all features
uv add "sanzaru[all]"
```

## Configuration

Set the audio files directory:

```bash
export AUDIO_PATH=/path/to/your/audio/files
export OPENAI_API_KEY=sk-...
```

### ElevenLabs (optional second TTS provider)

`create_audio` and `generate_podcast` accept `provider="elevenlabs"` alongside the default
`"openai"`. Speakers in a podcast choose independently, so an episode can mix both.

```bash
uv add "sanzaru[elevenlabs]"
export ELEVENLABS_API_KEY=...
# Optional: defaults are the Free-tier caps (2, or 4 on flash/turbo). Raise it on
# a paid tier; lower it if renders still hit HTTP 429.
export SANZARU_ELEVENLABS_MAX_CONCURRENCY=2
```

Differences that matter when switching:

- **Voice** is an opaque voice id from your library, not a name like `alloy`, and it is required.
- **`instructions` is ignored.** ElevenLabs has no equivalent parameter — put inline audio tags
  (`[whispers]`, `[excited]`) directly in the text with the default `eleven_v3` model.
- **Speed** is 0.7–1.2, and `eleven_v3` rejects any change at all; use `eleven_multilingual_v2`
  when you need speed control. Out-of-range values raise rather than being rescaled from OpenAI's
  0.25–4.0 range.
- **`voice_settings`** (`stability`, `similarity_boost`, `style`, `use_speaker_boost`, `speed`) is
  accepted here and rejected by the OpenAI provider.

Implementation lives in `src/sanzaru/audio/providers/`. A provider synthesizes one chunk and
returns mp3 bytes; `providers/base.py` owns text splitting, the bounded parallel fan-out, and
concatenation.

### Simulated podcasts (realtime)

`simulate_podcast` is a third mode, and a different thing entirely: the conversation is
**generated, not read**. N `gpt-realtime` sessions get personas and a rundown, one is given the
floor at a time, and its audio is played into the others' ears — so they react to delivery, not
to a transcript. No script, no TTS step.

```bash
sanzaru podcast rundown "your topic" --acts 3 -m 6 -o rundown.json
sanzaru podcast simulate @rundown.json --dry-run            # plan + cost, spends nothing
sanzaru podcast simulate @rundown.json --max-cost 2.00 -o ep.mp3
sanzaru podcast simulate --resume <run_id>                  # only the missing acts
```

Needs only `OPENAI_API_KEY` and the `[audio]` extra. It is the most expensive thing sanzaru does
(~$0.21 for a measured 7-minute episode on `gpt-realtime-2.1-mini`), so `--dry-run` and
`--max-cost` are habits, not options. Acts record in parallel and are checkpointed as they land;
QC transcribes the rendered audio and judges it against the plan.

A producer inside the tool handles floor control, talking-point coverage and landing each act —
but those are defaults. Each act takes `direction`, `turn_notes` and `speaking_order` so the
caller can direct it turn by turn.

Full rationale and measured numbers: [simulated-podcasts.md](simulated-podcasts.md).

### Podcast render modes

`generate_podcast` accepts `config.render_mode`:

- **`segments`** (default) — one request per turn, joined with your silence gaps. Exact control,
  per-speaker `voice_settings`/`speed`, and independent per-segment retry.
- **`dialogue`** — consecutive `eleven_v3` turns are sent as one request and the model paces the
  exchange itself. Distinctly more natural conversation.

Grouping is per-run: turns that can't join one (OpenAI speakers, other models, lone turns,
stretches in one voice, turns that alone fill the 2000-character request budget) still render
per segment, so mixed-provider episodes work in either mode. Inside a run, `pause_after` and
per-speaker `voice_settings` don't apply — use `config.dialogue_stability` (0–1) instead. Runs
split at turn boundaries to stay under 2000 characters, the ceiling ElevenLabs documents for a
reliable dialogue request.

**The trade is partial retry.** A run is a single request, so one bad line cannot be re-rendered
alone — fixing it re-spends every character in the batch. Since ElevenLabs draws quota down per
character submitted, and per-turn direction inside a run has to live in inline audio tags
(`[whispers]`) that count too, the expressive mode is also the one where a retry is most expensive.

The budget is per request, not per turn, so where a run splits decides what gets batched: a turn
left single-voice after a split silently renders as an ordinary segment (regaining its `pause_after`
and per-speaker tuning, losing the model-paced turn-taking). Nothing fails and nothing costs extra —
you just don't get dialogue for that turn. Full rules and worked examples in
[`docs/cli.md`](../cli.md#render-modes).

## Available Tools

### File Management
- `list_audio_files`: List and filter audio files
- `get_latest_audio`: Get most recent audio file

### Audio Processing
- `convert_audio`: Convert between formats (mp3, wav)
- `compress_audio`: Compress oversized files

### Transcription
- `transcribe_audio`: Standard Whisper transcription
- `chat_with_audio`: Interactive audio analysis with GPT-4o
- `transcribe_with_enhancement`: Enhanced transcription with templates

### Text-to-Speech
- `create_audio`: Generate TTS audio

### Podcasts
- `generate_podcast`: Multi-voice podcast from a script (segments or dialogue render mode)
- `simulate_podcast`: Realtime agents conversing from a rundown — no script

## Supported Formats

**Transcription:** flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
**Audio Chat:** mp3, wav
**TTS Output:** mp3, opus, aac, flac, wav, pcm

## Example Usage

```python
# With Claude Code
claude

# Then in Claude:
"List my audio files and transcribe the latest one with detailed enhancement"
```

## Documentation

- [Architecture](architecture.md) - Technical architecture details
- [MCP Overview](mcp-overview.md) - Model Context Protocol integration
- [MCP README](mcp-readme.md) - MCP server configuration
- [OpenAI Audio APIs](openai-audio.md) - API reference and capabilities
- [OpenAI Realtime](openai-realtime.md) - Realtime audio features
- [Simulated Podcasts](simulated-podcasts.md) - Realtime agents in conversation: the producer model, act chunking, cost, QC

## Attribution

This feature incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) v1.1.0 by Richie Caputo (MIT license).
