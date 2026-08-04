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
# Optional: lower if renders hit HTTP 429 (their cap is per subscription tier)
export SANZARU_ELEVENLABS_MAX_CONCURRENCY=3
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

## Attribution

This feature incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) v1.1.0 by Richie Caputo (MIT license).
