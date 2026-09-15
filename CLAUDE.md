# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A stateless MCP server (`mcp.server.mcpserver.MCPServer`, mcp SDK 2.x, protocol 2026-07-28) wrapping OpenAI's Sora Video API and Responses API (image generation). Supports both stdio (for MCP clients) and HTTP streaming (for web clients) transports. Exposes MCP tools for async video/image generation with polling-based workflows.

**Key Architecture Principles:**
- **Stateless**: No database, no in-memory job tracking. All state lives in OpenAI's cloud.
- **Async polling pattern**: Create → Poll → Download workflow for both videos and images
- **Security sandbox**: Reference images restricted to configured media paths with path traversal protection
- **Type-safe**: Extensive use of TypedDict and Literal types from OpenAI SDK
- **Dual transport**: stdio (default) for Claude Desktop, HTTP for web clients and remote access

## Development Commands

```bash
# Install dependencies
uv sync

# Run the MCP server (stdio mode - default)
uv run sanzaru
uv run sanzaru serve            # explicit alias

# Run the MCP server (HTTP mode - stateless). Loopback runs open; any other bind
# refuses to start without a bearer token (or the proxy escape hatch — see Transport Modes)
uv run sanzaru --transport http
uv run sanzaru --transport http --port 3000
SANZARU_HTTP_TOKEN="$(openssl rand -hex 32)" uv run sanzaru --transport http --host 0.0.0.0 --port 8080

# Agent CLI (same tools as shell commands — see docs/cli.md)
uv run sanzaru capabilities
uv run sanzaru video create "a cat stretches" --seconds 4 -o ./out/cat.mp4
uv run sanzaru image generate "an icon" --quality high -o ./art/icon.png
uv run sanzaru podcast generate @script.json --verify -o ep.mp3   # check the audio says it
uv run sanzaru podcast rundown "why TTS drops sentence tails" --acts 3 -m 6 -o rundown.json
uv run sanzaru podcast simulate @rundown.json --dry-run          # plan + cost, spends nothing
uv run sanzaru podcast simulate @rundown.json --max-cost 2.00 -o ep.mp3

# Lint and format code
ruff check .
ruff format .

# Test the server locally (requires Claude Code or MCP client)
claude  # in this directory with .mcp.json configured
```

## Core Architecture

**Async Architecture:**
- Fully non-blocking I/O with `aiofiles` and `anyio`
- CPU-bound operations (PIL, base64) run in thread pools
- Streaming downloads with async iteration
- See [`docs/async-optimizations.md`](docs/async-optimizations.md) for details

### Modular Server Design
The server is organized into focused modules for maintainability and code reuse:

```
src/sanzaru/
├── server.py           # MCPServer initialization & tool registration (run_server + argparse main shim)
├── polling.py          # wait_for_video/wait_for_image (neutral async wait loops, used by the CLI)
├── cli/                # Agent CLI (click): root group runs the server when no subcommand given
│   ├── __init__.py     # `sanzaru` entry point (console script → sanzaru.cli:main)
│   ├── _runtime.py     # run_async bridge, CLIError → envelope/exit-code mapping, client lifecycle
│   ├── _output.py      # JSON envelope contract (stdout) + exit-code constants
│   ├── _io.py          # -o/input path resolution onto storage path_overrides; @file/- content args
│   ├── video.py        # video create/remix/status/wait/download/list/delete/files
│   ├── image.py        # image generate/edit (sync) + create/status/wait/download (async) + prepare/files
│   ├── audio.py        # audio transcribe/chat/speak/convert/compress/files
│   ├── podcast.py      # podcast rundown (plan) / simulate (realtime) / generate (scripted TTS)
│   ├── misc.py         # top-level `wait` (mixed job types) + `capabilities`
│   └── serve.py        # explicit `sanzaru serve`
├── types.py            # TypedDict definitions
├── config.py           # OpenAI client + path configuration (get_client/set_client, get_path)
├── security.py         # File security utilities (validate_safe_path, check_not_symlink, open_nofollow)
├── utils.py            # Shared helpers (+ validate_resource_id, reject_reserved_name)
├── features.py         # Feature detection (optional deps + env vars)
├── descriptions.py     # LLM-facing tool descriptions
├── dotenv_loader.py    # ./.env autoload, minus the variables that redirect credentials
├── user_context.py     # Per-request user context (multi-tenant support)
├── storage/            # Pluggable file I/O
│   ├── protocol.py     # StorageBackend protocol + FileInfo
│   ├── factory.py      # get_storage() factory (+ set_storage_backend override used by the CLI)
│   ├── local.py        # Local filesystem backend
│   └── databricks.py   # Databricks Unity Catalog Volumes backend
├── infrastructure/     # Shared infrastructure
│   ├── cache.py        # Audio file support caching
│   ├── file_system.py  # FileSystemRepository (storage-backed file I/O)
│   ├── path_resolver.py # SecurePathResolver
│   └── text_utils.py   # split_text_for_tts and text chunking
├── audio/              # Audio domain logic and services
│   ├── processor.py    # AudioProcessor (format conversion, concatenation, slicing)
│   ├── verification.py # words/similarity/transcribe_bytes — shared by qc, #35, #39
│   ├── windowing.py    # overlapping windows for long-file transcription
│   ├── providers/      # Pluggable TTS backends
│   │   ├── base.py               # SpeechRequest, TTSProvider protocol, synthesize_speech
│   │   ├── openai_provider.py    # client.audio.speech (default)
│   │   ├── elevenlabs_provider.py # client.text_to_speech.convert (optional extra)
│   │   └── __init__.py           # get_provider() registry (lazy imports)
│   ├── realtime/       # Simulated podcasts: agents that actually converse
│   │   ├── types.py    # HostSpec/ActBrief/Rundown/Turn/RealtimeUsage + PCM16 helpers
│   │   ├── agent.py    # one persona on one connection: configure/speak/hear/steer
│   │   ├── live_agent.py # same surface on the full-duplex Live API (gpt-live-1, per-minute billing)
│   │   ├── duplex.py   # gpt-live default: chain-fed hosts, silent notes on the act clock, mixed act
│   │   ├── producer.py # floor control, coverage steering, act budgets, prompts
│   │   ├── rundown.py  # pre-production: premise → parallel-recordable acts
│   │   ├── budget.py   # shared cost ceiling, charged every turn
│   │   ├── pricing.py  # token→dollars + the measured rates a dry run projects from
│   │   ├── mixdown.py  # PCM→AudioSegment, time-aligned stems, checkpoint decode
│   │   └── qc.py       # transcribe rendered audio, judge it against the rundown
│   └── services/       # TTSService, FileService, AudioService, TranscriptionService
├── tools/              # Tool implementations
│   ├── video.py        # 7 video tools
│   ├── reference.py    # 2 reference image tools
│   ├── image.py        # 3 image generation tools (Responses API)
│   ├── images_api.py   # 2 image tools (Images API, gpt-image-2.5 by default)
│   ├── audio.py        # 9 audio tools (list, transcribe, TTS, chat)
│   ├── podcast.py      # 1 podcast generation tool (scripted TTS)
│   ├── simulate_podcast.py # 1 simulated podcast tool (realtime agents, parallel acts)
│   ├── wait.py         # 1 job tool: wait_for (server-side wait on video_*/resp_* ids, progress per poll)
│   └── media_viewer.py # 2 media viewer tools (MCP App)
└── app/                # Frontend assets (built, committed)
    └── media-viewer/   # React MCP App for media playback
```

**server.py** registers all tools with MCPServer decorators and delegates to tool implementations; the media viewer goes through the SDK's `Apps` extension so the server advertises `io.modelcontextprotocol/ui`
**types.py** defines all return types (DownloadResult, VideoSummary, etc.)
**config.py** provides `get_client()` and `get_path()` with validation
**security.py** provides reusable functions: `validate_safe_path()`, `check_not_symlink()`, `safe_open_file()`
**utils.py** provides helpers: `suffix_for_variant()`, `generate_filename()`
**tools/*.py** contain the actual tool implementations as plain async functions

### Agent CLI

`sanzaru <group> <verb>` wraps the same `tools/*.py` functions for shell/agent use (thin
wrappers, exactly like `server.py` does for MCP). Bare `sanzaru` still starts the MCP server —
back-compat for every existing config is guarded by tests. Key design points:

- **Contract**: stdout = one JSON envelope per input (JSONL for fan-out, completion order);
  stderr = progress/hints. Exit codes: 0 ok · 1 runtime · 2 usage · 3 config · 4 timeout
  (job still running — envelopes carry a `resume` command) · 5 job failed · 6 partial batch ·
  130 interrupted. See `docs/cli.md`.
- **One-shots**: `-o` ⇒ `--download` ⇒ `--wait`; `polling.py` owns the wait loops (adaptive
  backoff, transient-error retry, `WaitTimeoutError` carries last-seen state).
- **Override seams** (CLI-only; the MCP server never touches them, reset in `finally`):
  `config.set_client()` shares one `AsyncOpenAI` per invocation across poll loops;
  `storage.set_storage_backend()` maps `-o`/path inputs onto
  `LocalStorageBackend(path_overrides=..., file_overrides=...)` while `validate_safe_path` still
  sanitizes basenames. `path_overrides` is one directory per path type and anchors the *output*
  side; `file_overrides` keys `(path_type, basename)` so an input batch can span directories,
  each file validated under its own parent. Per-file has to live inside one backend instance:
  the backend is a process global installed once per invocation, and the fan-out reads
  concurrently, so there is no point at which it could be swapped. Two inputs of one type
  sharing a basename stay a usage error — the tool layer only ever sees bare names.
- **Lazy imports**: command bodies import `sanzaru.tools.*` at call time so `sanzaru --help`
  never pays the openai/MCPServer import cost (enforced by a startup-weight test); missing
  optional extras surface as `config` envelopes (exit 3) with the install command.
- CLI tests live in `tests/cli/` (CliRunner; mock at the `sanzaru.tools.*` layer).

### TTS Providers

Speech generation goes through `audio/providers/`, never straight to an SDK. Both TTS entry points
(`TTSService.create_speech` → `create_audio`, and `generate_podcast`) build a `SpeechRequest` and
call `synthesize_speech(provider, request, limiter=...)`.

Division of labour: **a provider synthesizes one chunk and returns mp3 bytes**; `base.py` owns
splitting long text, the bounded parallel fan-out, and concatenation. mp3 is a contract, not a
detail — `podcast._stitch_audio` decodes every segment with `AudioSegment.from_mp3`.

| | openai (default) | elevenlabs |
|---|---|---|
| Default model | `gpt-4o-mini-tts` | `eleven_v3` |
| Voice | named (`alloy`, `onyx`, …) | opaque voice id, required |
| `instructions` | supported | **ignored** — use inline audio tags (`[whispers]`) with eleven_v3 |
| `voice_settings` | rejected | stability / similarity_boost / style / use_speaker_boost / speed |
| Speed | 0.25–4.0 | 0.7–1.2; **eleven_v3 rejects any change** |
| Chunk limit | 4000 chars | 3000 (v3) / 10000 (multilingual) / 40000 (flash, turbo) |
| Concurrency | unbounded | 2, or 4 on flash/turbo (Free-tier caps), per subscription tier |

Every synthesis reports what it submitted. `synthesize_speech` returns a `SpeechResult`
(audio + `SpeechUsage`) rather than bare bytes, counting characters from the chunks actually
sent — the text is already in hand at that layer, so nothing has to come back from the API and
the count survives a failed request. The `TTSProvider` protocol is untouched: making every
backend count would have been the same number, computed in more places. ElevenLabs bills per
character against a monthly quota, so this is the number that matters there; OpenAI is not
metered that way and reports it for information. `sanzaru capabilities --quota` reads the
allowance itself, and is the only part of that command that touches the network.

Speed ranges are **not** rescaled across providers — an out-of-range value is a `ValueError`, so
`speed=2.0` never silently means two different things.

Provider selection in podcasts resolves `speaker.provider` → `config.provider` → the tool's
`provider` argument, so one episode can mix providers. Concurrency uses one
`anyio.CapacityLimiter` per provider, built per invocation (a limiter binds to its event loop —
never cache one at module scope) and shared between segment- and chunk-level fan-out, because that
is what ElevenLabs' cap actually counts.

### Podcast Render Modes

`config.render_mode` picks how turns become TTS requests. Default `"segments"` — unchanged
behavior, one request per segment joined with configured silence.

`"dialogue"` batches maximal runs of consecutive turns sharing a dialogue-capable provider **and**
model into a single `/v1/text-to-dialogue` request, so the model paces the conversation. Grouping
happens in `_plan_render_units`, which returns `RenderUnit`s — each either one segment or a run.
Everything downstream (limiters, stitching, pause list) works in units, and a unit's pause comes
from its *last* segment.

Turns that cannot join a run fall back to segment units, which is what keeps dialogue mode
compatible with mixed-provider episodes. A run is only batched when it carries
≥`MIN_DIALOGUE_SPEAKERS` distinct *resolved voices* (not speaker ids — two speaker entries can
share one voice id) — a monologue has no turn-taking to pace, so batching it would cost a dialogue
request and swallow its `pause_after`s for nothing; at 2 that rule also excludes a lone turn.
Excluded outright: OpenAI speakers and non-`eleven_v3` models. Runs are split at turn boundaries to
stay within `ELEVENLABS_DIALOGUE_MAX_CHARS` (2000, the ceiling `/v1/text-to-dialogue` documents for
reliable generation — past it the stream can terminate early, which reads as a short but successful
take, so `synthesize_dialogue` refuses over-budget requests outright). That budget is the only
length rule, and it sits below every `max_chunk_chars`: a turn too long to share a dialogue request
opens a run of its own, which closes as a single-voice run — a segment unit, chunked normally.

The same mechanism strands ordinary turns, which is the less obvious half: the budget is per
*request*, so where a run splits decides what batches. Three 900-char turns (`a, b, a`) close a run
after turn 2 and leave turn 3 single-voice — a segment unit, with nothing near the ceiling. Callers
see it in the `Dialogue mode: N/M segments batched` line (`N < M`), and
[`docs/cli.md`](docs/cli.md#render-modes) documents it as the likelier way to lose dialogue than an
over-long turn.

Dialogue is a **separate** `DialogueProvider` Protocol (`runtime_checkable`, narrowed by
`as_dialogue_provider`) rather than a method on `TTSProvider`, so OpenAI isn't forced to stub a
capability it lacks.

Keep both modes. They trade off in opposite directions: segments gives exact gaps, per-speaker
`voice_settings`/`speed`, and independent per-segment retry — which is precisely what #35's verify
pass needs. Dialogue gives natural flow but is atomic per request, takes one `stability` for the
whole run, and ignores intra-run `pause_after`.

Client seam: `config.get_elevenlabs_client()` mirrors `get_client()`, but builds lazily and caches
rather than being installed eagerly by the CLI runtime, so `sanzaru --help` never imports the SDK.
Missing key raises `RuntimeError`, missing extra raises `ImportError` — both map to CLI exit 3 via
`_classify`. `ConfigurationError` would *not* (it falls through to exit 1).

### Podcast Verification

`generate_podcast(verify=True)` transcribes each rendered unit **before stitching** and checks it
against the script. Verifying there rather than against the finished episode is what makes
attribution exact: `segment_bytes_list` is still index-aligned to `units`, every unit is far under
the 25MB/duration limits so no windowing is involved, and no offset arithmetic against the mixed
file is needed. It does not prove the stitch — but the documented failure is TTS dropping speech,
and stitching is a deterministic pydub concatenation.

Two checks, because there are two documented failures: a fuzzy match on the last
`VERIFY_TAIL_WORDS` (tails are what drop), and presence anywhere in the unit's audio for segments
of `VERIFY_SHORT_SEGMENT_WORDS` or fewer (a three-word segment vanished whole). Thresholds are
loose on purpose — this looks for *absent speech*, not a perfect reading, and ASR reorders
currency and wobbles on proper nouns.

Failures re-render **once**, then are reported. Drops are per-render random, not per-segment
sticky, so a segment that fails twice wants its tail rewritten rather than a third render. A
dialogue unit is atomic — a failure anywhere in a batched run re-renders the whole run, since the
model paced those turns together.

A `SegmentVerdict` carries `checked` separately from `ok`, and `verified=True` requires both.
Transcription is retried `VERIFY_TRANSCRIBE_ATTEMPTS` times (cheap, idempotent, mostly transient
429s); only then is the segment downgraded to `not_transcribed` with `checked=False`, and a unit
over `TRANSCRIBE_MAX_BYTES` is `too_large_to_verify` without a call. Neither ever loses the
episode — the same property `run_qc` has — but both make it **unverified** (`verified=False`),
never verified: folding "nothing was found" into "nothing was looked at" made the control assert
its own success (CWE-636). The CLI reports the two as separate blocks for the same reason.

Scoring runs off the event loop and the window scan strides by `span // VERIFY_WINDOW_STRIDE_DIVISOR`
for long needles (tails and short segments still slide one word at a time), so a 40k-char segment
is not minutes of uninterruptible CPU per unit.

Bounds are enforced in `_validate_script` before any synthesis: `MAX_SEGMENTS`, `MAX_SILENCE_MS`
per knob, `MAX_TOTAL_SILENCE_MS` over the sum (deliberately over-counting the final pause and
intra-dialogue pauses — a ceiling wants the upper bound), and `DEFAULT_PODCAST_MAX_CONCURRENCY`
where a provider reports no cap of its own. A JSON `null` on a pause knob means "not set", the
same as an absent key (`_default_pause_ms` / `_pause_after_ms`; `dict.get(key, default)` alone
does not do that). The output name runs `reject_reserved_name`, `safe_audio_format` against
`SAFE_AUDIO_EXTENSIONS`, and the checkpoint-sidecar pre-flight; all three raise `ValueError` so
the CLI classifies them alike (usage, exit 2).

### Simulated Podcasts (realtime)

A third, categorically different mode: the conversation is **generated, not read**. N
`gpt-realtime` sessions get personas and a rundown; a producer gives one the floor and plays its
PCM frames into the others' `input_audio_buffer`, so they respond to delivery, not to a transcript.
`audio/realtime/` holds the machinery, `tools/simulate_podcast.py` the tool, and
[`docs/audio/simulated-podcasts.md`](docs/audio/simulated-podcasts.md) the full rationale with
measured numbers.

Non-obvious things that are easy to break:

- **The producer is load-bearing, and its output is a default, not a fixture.** Without per-turn
  steering, agents drift to 30s turns and mutual agreement — so `producer.py` owns floor control,
  walking talking points across the act, and landing the close (steering notes go out as system
  messages, never heard). But the caller is usually another agent and is a better producer than
  our f-strings, so `ActBrief.direction` / `turn_notes` / `speaking_order` each *replace* the
  generated behavior. It can't steer live (a round-trip per turn would break parallel acts), so
  direction is declarative and set before recording.
- **`max_output_tokens` counts the transcript, not just the audio.** Audio is a very steady 20
  tok/s, but the returned transcript adds another 9–17. Sizing the cap from the audio rate alone
  truncated 17 of 29 turns mid-sentence. `turn_token_cap()` budgets both and applies
  `TURN_TOKEN_HEADROOM` (1.5 — tuned, see the doc's table).
- **Acts drift forward.** Each act is recorded blind to the others, so `prior_context` keeps it
  from repeating earlier ground and `upcoming` keeps it out of *later* acts' material. Omitting
  `upcoming` made every act off-brief and act 3 a rerun. It's derived in
  `simulate_podcast.annotate_upcoming` (lookahead 2) so hand-edited rundowns get it too.
- **Chunking into acts is mandatory, not an optimization.** A Realtime WebSocket dies at 60
  minutes. Parallel acts also put a 30-minute episode at ~1 min wall clock, which is what makes a
  blocking tool acceptable when the repo has no job registry.
- **Every act is checkpointed before it can be lost** (mp3 + a turns/usage sidecar), plus a
  `simrun_<id>.json` manifest written before recording so `--resume <run_id>` needs nothing else.
  Storage has no delete op, so checkpoints persist — documented, not cleaned up.
- **Checkpoints bypass the `-o` storage override; deliverables don't.** `-o` repoints the whole
  "audio" path type for one invocation, but the resume hint carries no `-o`, so anything found by
  run id alone goes through `storage.get_default_storage()` (`simulate_podcast.checkpoint_storage`).
  Only the episode and stems follow `-o`. Loading a checkpoint decodes inside the try — a
  truncated or zero-audio act is re-recorded, never fatal to the whole resume. The two writes
  that make a checkpoint happen inside `anyio.CancelScope(shield=True)`: a sibling act's failure
  cancels the task group, and an mp3 with no sidecar is paid-for audio that resume throws away.
- **On `--resume`, caller-set fields win and the manifest fills the rest** (`model_fields_set` is
  the signal). That is why `cli/podcast.py` gives every simulate option a `None` click default and
  only puts what the user typed into the payload: a click-supplied `"mp3"` would be
  indistinguishable from the user asking for mp3, and a dropped `--max-cost` would make the
  ceiling abort's own resume hint re-run uncapped. The restored ceiling also counts the spend
  replayed from the checkpoints, so a bare resume under it would abort at the same total after
  paying to re-record: `_refuse_a_resume_that_cannot_finish` projects the remaining acts and
  stops first, and the CLI's ceiling envelope prints a resume command with a raised
  `--max-cost` (`CostBudget.suggested_limit_usd`).
- **`gpt-live-1` is a different API, and its default mode is a different protocol.**
  `is_live_model()` (prefix `gpt-live`) routes a host to `live_agent.LiveAgent`:
  `client.live.connect()` with no model on the URL, one immutable `session.start`, every append
  acked with `*.appended` (matched by `event_id`). Measured facts the design rests on: **the
  session timeline only advances while input audio streams** (no input → no speech, no applied
  notes, one `context_injection_incomplete` per pending append at close), so every agent runs a
  100 ms input clock from `session.started` to `session.close` — *an act runs in real time*,
  parallel across acts; **output is a continuous stream with exact-zero silence frames**, so
  speech is detected by loudness (`SPEECH_RMS_THRESHOLD` 300; speech is 500–3500);
  **`instructions.append` is spoken aloud, `thinking.append` is not** — all producer notes go
  through `LiveAgent.think()`. `SimulationSettings.live_mode` (default `duplex`, CLI
  `--live-mode`) picks the protocol for an all-Live table: `duplex.run_duplex_act` opens every
  host, chain-feeds each host's output frames into every other host's clock for the whole act
  (per-source mixing for N>2), sends the producer's plan as thinking notes on the *act clock*
  (`plan_notes`: opener at 0, points spread to 85 %, close to the designated closer at 85 %, wrap
  at 100 %), counts overlap into `collision_seconds` and tells the later starter to yield after
  1.5 s, ends on quiet after the close or at `target × 1.5 + 30 s`, and returns an `ActResult`
  with `mixed_pcm` + per-host `stems` + `Turn`s grouped from timestamped transcript deltas
  (`group_utterances`). `join_pcm()`/`seconds` prefer the mix, `_build_timeline` emits a
  `MixedBlock`, the sidecar carries `mode`/`collision_seconds`, and a resumed duplex act is the
  mix whole (no stems). `cued` keeps the floor-control loop (used automatically for a mixed
  Realtime/Live table) with a 1 s settle gap before each cue and a ≤1 s false-start trim. Billing
  is **$0.05 per session-minute per host**: `usage.seconds` meters streamed session time and is
  cumulative (never sum it); `RealtimeUsage.live_seconds` carries deltas (max of reported and
  wall clock), `ModelPrices.per_minute` prices it, `project_usage(model=...)` projects
  `target_seconds × hosts`, `live_seconds` and `live_mode` are signed (`SIGNATURE_VERSION` 2), and
  `SANZARU_REALTIME_PRICE_*` accepts an optional 7th value (`per_minute`).
- **Every turn runs under `anyio.fail_after`.** Nothing in the Realtime protocol bounds a turn, and
  a stalled session would hold a `CapacityLimiter` slot forever inside a blocking tool. The bound
  is 6x `turn_seconds` (min 60s), overridable via `SANZARU_REALTIME_TURN_TIMEOUT` /
  `turn_timeout_s`; a breach becomes `RealtimeAPIError`, which fails the run (no per-act retry)
  but leaves finished acts checkpointed — the CLI attaches `run_id` + `resume` to *every*
  post-recording failure envelope, not just the ceiling. Relatedly, `speak()` requires a
  `response.done`: the SDK's `__aiter__` *returns* on `ConnectionClosedOK`, so a graceful mid-act
  close would otherwise look like a successful silent turn.
- **The act clock ends the act; it does not abort it.** A turn breach raises, but the act-level
  bound (`act_wall_budget_seconds`, default 3000s under the 60-minute session close) instead
  forces the closing turn and reports `stop_reason="wall_clock"`. It has to: `run_act` writes no
  checkpoints — `_record_act` encodes the mp3 and writes the sidecar *after* it returns — so
  anything that unwinds mid-act discards every turn already paid for. There is no partial-act
  persistence path. Same reason `close_is_due` exists, against the other clock.
- **`_stitch_audio` takes a `decode` callable.** Realtime is PCM16/24k end to end with no mp3
  round-trip; the scripted path still passes mp3. Everything downstream is shared.
- **QC inverts the usual check.** No script means no ground truth, so verification compares the
  API's own per-turn transcript against `gpt-transcribe` on the rendered audio (catches dropped
  audio deterministically), then judges the transcript against the rundown. `gpt-live-transcribe`
  is realtime-only and 404s on `/v1/audio/transcriptions` — do not swap it in.
- **The cost ceiling is charged every turn** via a shared `CostBudget`, and arrives at the CLI
  inside an `ExceptionGroup` — possibly one per act. `find_in_group()` in `cli/_runtime.py` is what
  makes that still exit 6 with a resume hint instead of an opaque "3 parallel tasks failed".
  Replayed checkpoints are charged at the *dearest* priced billable model (`_replay_model`): a
  checkpoint stores one aggregated `usage` with no per-model split while the live run charged
  each turn at its host's model, so replaying at the episode model let per-host overrides onto a
  dearer model under-count the restored ceiling. Erring closed is the short-term answer; the real
  one is a per-model breakdown in the checkpoint.

### Runtime Path Configuration
Paths are validated lazily via the `get_path()` function when tools are called:
- `get_path("video")`: Returns validated path for video downloads
- `get_path("reference")`: Returns validated path for reference images
- `get_path("audio")`: Returns validated path for audio files

Resolves from `SANZARU_MEDIA_PATH/{subdir}` (with auto-creation) or individual env vars. Paths are cached with `@lru_cache` for performance.

### Two API Integration Patterns

**0. Waiting on jobs (`wait_for`)**
- One call over any mix of `video_*` and `resp_*` ids, waited concurrently via `polling.py`
- Reports progress to the client on every poll (that resets clients' idle timers — Claude Code
  aborts a silent HTTP tool call at 5 min and backgrounds calls past 2 min)
- The deadline *returns* (`timed_out=true` per job, last-seen status) instead of raising; call again
  to resume. Default 240 s (under the idle window), max 1800 s. `download=true` saves completed jobs
- MCP Tasks are not used: mcp 2.x ships only the wire types and no Claude client implements the
  extension (see `docs/async-task-tracking.md`)

**1. Sora Video API (client.videos.*)**
- Async jobs with polling: `create()` → `retrieve()` → `download_content()`
- Status progression: `queued` → `in_progress` → `completed` or `failed`
- Progress tracking: 0-100 integer
- Uses OpenAI SDK types: `Video`, `VideoModel`, `VideoSize`, `VideoSeconds`
- Download supports optional custom filenames with path traversal protection

**2. Responses API (client.responses.*)**
- Background image generation: `create(background=True)` with `tools=[{"type": "image_generation"}]`
- Iterative refinement via `previous_response_id` parameter
- Returns `Response` object with `output` array containing `ImageGenerationCall` items
- Base64-encoded image in `ImageGenerationCall.result`

### Security Model
All file operations use centralized security utilities from `security.py`:

**`validate_safe_path(base_path, filename, allow_create=False)`**
- Prevents path traversal attacks (e.g., `../../etc/passwd`)
- Ensures resolved path stays within base_path
- Optionally validates file existence

**`check_not_symlink(path, error_context)`**
- Prevents symlink exploitation
- Raises ValueError if path is a symbolic link

**`safe_open_file(path, mode, error_context, check_symlink=True)`**
- Context manager for safe file I/O
- Standardized error handling (FileNotFoundError, PermissionError, OSError)
- Optional symlink checking

Example usage:
```python
from security import validate_safe_path, safe_open_file
from config import get_path

base_path = get_path("reference")
file_path = validate_safe_path(base_path, user_filename)

with safe_open_file(file_path, "rb", "reference image") as f:
    data = f.read()
```

Additional security:
- Symlinks rejected in environment variable paths (`get_path()` validation)
- Empty/whitespace-only env vars rejected
- User filenames validated against allowed extensions where applicable

### Invariants that are load-bearing, not stylistic

Each of these closed a specific hole. They look like ordinary defensive code, so
they are easy to "simplify" away.

- **`check_not_symlink` uses `lstat`, never `exists() and is_symlink()`.** `exists()`
  follows the link, so the compound form returned False for a *dangling* symlink and
  skipped the check — passing exactly the links a `"wb"` open would then follow and
  create the target of. `os.lstat` rather than `Path.is_symlink()` because the latter
  swallows every `OSError`, which made the documented permission-denied branch
  unreachable. `config.get_path` and `audio/config.py` follow the same rule for the
  operator-supplied media path.
- **Every storage open passes `opener=open_nofollow`, and every writer also calls the
  pre-open check.** Validation and open are not atomic; `O_NOFOLLOW` is what refuses a
  symlink swapped in between them (CWE-367). It is POSIX-only, so on a platform
  without it the pre-open `check_not_symlink` in `write`/`write_stream` is the only
  thing between a `"wb"` open and a planted link's target. `safe_open_file` and
  `async_safe_open_file` use the same opener, so `check_symlink=False` skips the
  pre-check only — it never means "follow links".
- **`reject_reserved_name` matches the basename storage would write, not the raw
  string.** Both backends canonicalise (`PurePosixPath(name).name`,
  `validate_safe_path` resolving `./x`, `x/`, `a/../x` to `<base>/x`), so a check on the
  raw string let `./simrun_<id>.json` clobber another run's manifest. Any name check
  has to run on the same reduction as the write.
- **`SANZARU_REQUIRE_USER_CONTEXT` is a strict boolean that fails closed.** The
  switch's failure mode is "every user shares one namespace", so an unrecognised
  value (`enabled`, `y`) is a configuration error, never a silent "off" — the original
  `1/true/yes` parse treated `on` as off.
- **The ffmpeg demuxer is never chosen from an untrusted extension.** pydub turns
  `format=<ext>` into `ffmpeg -f <ext>`, and playlist demuxers (`hls`, `concat`,
  `dash`) read *other local files* named by the file's content. Every
  `AudioSegment.from_file` whose format derives from a filename goes through
  `demuxer_for()` (which runs `safe_audio_format()` and then maps the extensions
  ffmpeg does not know by name — `.opus`→`ogg`, `.m4b`→`mp4`, `.wma`→`asf` — onto
  a self-contained demuxer). Current ffmpeg blunts the playlist escapes on its
  own (`concat` defaults to `safe=1`; `hls` checks `allowed_extensions`), so for
  that class the allowlist holds regardless of the host build. It is not a
  complete answer: `mov`/`mp4` stay allowlisted, and that demuxer can follow
  external data references (`dref`) — gated only by ffmpeg's `enable_drefs=0`
  default, which pydub gives us no way to pass as an input option. That one does
  depend on the host's ffmpeg defaults.
- **Two extension sets, and they are not interchangeable.**
  `DECODABLE_AUDIO_EXTENSIONS` gates what ffmpeg may *read* — the rule is only
  "a self-contained container", so `.aac`/`.opus`/`.wma` belong here and
  narrowing it to the transcription formats broke `convert_audio`, whose whole
  job is unsupported input. `SAFE_AUDIO_EXTENSIONS` gates what this server may
  *create* under a caller-chosen name: `convert_audio`, `compress_audio` and
  `create_audio` all run `require_audio_name()` on the output before any work, so
  none of them can mint a `.json` or `.html` for `/media` to serve. It is not
  enforced by the storage layer (`storage.write` checks containment only), and
  `generate_podcast` only *warns* on a mismatched extension — so the claim is
  about those three tools, not every write.
- **Every convert/compress output goes through
  `FileSystemRepository.write_audio_file`.** pydub used to export straight into
  `storage.local_tempfile`, and the no-op copy was a `shutil.copyfile`; neither
  passed through the repository, so any write-time policy attached there (the
  checkpoint-overwrite refusal) silently did not cover them. The name check in
  `require_audio_name` does not protect act checkpoints and says so — they are
  ordinary `.mp3` names, and the write path is the only place that can tell.
- **`list_audio_files(pattern=)` is a substring or fnmatch glob, and refuses
  regex syntax.** The old `re.search` was a ReDoS (`(a+)+$` froze the event
  loop). fnmatch is itself `re`-backed but safe through its atomic-group
  translation of `*`, which is the property `compile_name_filter` relies on;
  regex-only metacharacters raise rather than fail open, because a caller on the
  old contract recovers from an error and not from an empty list.
- **Run checkpoints are protected by looking, not by their filename.** A
  name-shaped rule for `<slug>_<runid>_<actid>.mp3` cannot be written without
  false positives — eight hex characters is also what a date looks like, and it
  rejected `interview_20250826_part1.mp3`. `write_audio_file` instead probes for
  the act sidecar (one ranged read of its head) and refuses the mp3/wav *and*
  the sidecar itself — an unguarded sidecar defeated the guard in two calls,
  since the mp3 check reads it. `is_bookkeeping=True` is the run's own opt-out,
  which `--qc-retry` and the manifest write need. `reject_reserved_name` covers
  only `simrun_*.json`, which is unambiguous, and the write applies it too.
- **`/media` serves an allowlisted content type or `application/octet-stream`,
  always with `nosniff` + `Content-Disposition: attachment`.** The response type
  used to come from `mimetypes.guess_type()` of a caller-chosen name, which made a
  stored `.html` an executable document in the server's own origin — the origin the
  MCP SDK's rebinding allowlist trusts for `/mcp`. It also applies the same
  Host/Origin policy as `/mcp` by hand, because the SDK appends custom routes
  outside that middleware.
- **HTTP mode requires a bearer token on a non-loopback bind.** The Host allowlist
  is not authentication — a network peer writes that header themselves.
- **`build_http_app()` is the only HTTP artifact.** `mcp.streamable_http_app()` alone
  carries no bearer or identity middleware; the docs used to hand that bare app out as
  the production mount, which left `/mcp` open with the token exported. `_run_http`
  and any embedding must both go through the seam.
- **Off loopback, the Origin check is kept, not dropped with the Host allowlist.**
  `SANZARU_ALLOWED_HOSTS` keeps the SDK's rebinding protection on with the real
  hostnames; the unauthenticated hatch *requires* it, so the unauthenticated path is
  never also the least-protected one. Only a token-protected bind without an allowlist
  turns the check off.
- **The identity header must appear exactly once.** `Headers.get()` returns the
  *first* copy, and an appending proxy forwards the client's copy first — so two
  copies (or a malformed one) are a 400, never "bind the first" or "bind nobody".
- **A resume only trusts bookkeeping written for *this* run.** `RunManifest` is the
  resumed run's configuration (ceiling, models, filename, rundown), so `run_id` must
  match and, when `SANZARU_RUN_SECRET` is set, the HMAC must verify. An
  `ActCheckpoint` carries its `run_id` and a digest of its mp3, and
  `_load_checkpoint` checks both whenever they are non-empty — **secret or not**,
  because the cheap attack needs no forgery (copy a genuine pair from your own
  run over the victim's act names). They are inside the signed tuple too: on a
  shared installation the attacker can have this code mint them a validly signed
  checkpoint, so a signature that binds neither the run nor the audio proves
  nothing. The signature covers an *explicit, versioned tuple* (`signed_fields`,
  `SIGNATURE_VERSION`), never `model_dump_json()`: signing the whole dump made any
  later defaulted field invalidate every signed file, stranding paid acts on the
  first upgrade — the same failure class the pricing refusal was moved out of a
  validator to avoid. `compare_digest` is fed bytes, since a non-ASCII `sig` makes
  the str form raise TypeError instead of returning False.
- **A ceiling that cannot price a model refuses instead of proceeding** — in
  `check_ceiling_is_enforceable`, called from the tool body (once on the brief
  alone before the planner is billed, again once the rundown's per-host models
  are known), and again at `charge()`, which raises `UnpricedModelError` so the
  CLI does not answer it with a `--max-cost` hint that cannot work. Deliberately
  *not* a pydantic validator: `RunManifest.brief` is a `SimulationBrief`, so
  raising during validation made reading a manifest fail and turned a missing
  price into an unresumable run with paid acts stranded. The `charge()` copy is
  what covers resume, since `model_copy(update=...)` does not re-run validators.
  A dry run is exempt — it spends nothing and its projection is how you discover
  the model is unpriced (`project_run` reports `unpriced_models`, `usd=None`).
- **`verified=True` means checked, not merely un-flagged.** A `SegmentVerdict`
  carries `checked` separately from `ok`; folding them made a transcription failure
  report success.

## Prompting Sora with Reference Images

**CRITICAL**: When using `input_reference_filename`, keep prompts simple and focused on motion/action ONLY.

❌ **Bad**: Re-describing what's already in the image
```python
create_video(
    prompt="A pilot in orange suit sitting in cockpit with instruments glowing...",
    input_reference_filename="pilot.png"
)
```

✅ **Good**: Describing only the action/transformation
```python
create_video(
    prompt="The pilot glances up, takes a breath, then returns focus to the instruments.",
    input_reference_filename="pilot.png"
)
```

The reference image already contains: character, setting, framing, style, lighting.
The prompt should only describe: what happens next, motion, camera movement.

See `docs/sora-prompting-guide.md` and `docs/sora2_prompting_guide.ipynb` for complete prompting guidelines.

## Typical Workflows

### Generate Reference Image → Animate with Sora
```python
# 1. Generate reference image
resp = create_image(prompt="futuristic pilot in mech cockpit", size="1536x1024")
get_image_status(resp.id)  # poll until completed
download_image(resp.id, filename="pilot.png")

# 2. Resize for Sora if needed
prepare_reference_image("pilot.png", target_size="1280x720", resize_mode="crop")

# 3. Create video with simple motion prompt
create_video(
    prompt="The pilot looks up and smiles.",
    input_reference_filename="pilot_1280x720.png",
    size="1280x720",
    seconds="8"
)
```

### Iterative Image Refinement
```python
# Generate initial concept
resp1 = create_image(prompt="a cyberpunk character")

# Refine with previous_response_id
resp2 = create_image(
    prompt="add more neon details and a cityscape background",
    previous_response_id=resp1.id
)

# Continue refining
resp3 = create_image(
    prompt="change camera angle to show profile",
    previous_response_id=resp2.id
)
```

## Image Resize Modes

Three modes available in `prepare_reference_image`:
- **crop**: Preserve aspect ratio, scale to cover target, center crop excess (no distortion, may lose edges)
- **pad**: Preserve aspect ratio, scale to fit, add black letterbox bars (no distortion, full image preserved)
- **rescale**: Stretch/squash to exact dimensions (may distort, no cropping/padding)

## Environment Configuration

Required:
```bash
OPENAI_API_KEY="sk-..."
```

### Media Storage (choose one)

**Option 1 — Unified path (recommended):**
```bash
SANZARU_MEDIA_PATH="/absolute/path/to/media"  # Auto-creates videos/, images/, audio/ subdirs
```

**Option 2 — Individual paths (legacy, still supported):**
```bash
VIDEO_PATH="/absolute/path/to/videos"
IMAGE_PATH="/absolute/path/to/references"
AUDIO_PATH="/absolute/path/to/audio"
```

Individual paths take precedence over `SANZARU_MEDIA_PATH` when both are set.

Optional:
```bash
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR (defaults to INFO)

# ElevenLabs TTS provider — needs `uv pip install 'sanzaru[elevenlabs]'`
ELEVENLABS_API_KEY="..."
SANZARU_ELEVENLABS_MAX_CONCURRENCY=2  # Free-tier cap (4 on flash/turbo); raise on a paid tier
SANZARU_OPENAI_MAX_CONCURRENCY=0      # 0 = unbounded (default)

# Simulated podcasts (needs only OPENAI_API_KEY + the [audio] extra)
SANZARU_TRANSCRIBE_MAX_CONCURRENCY=4  # windows in flight when transcribing a long file
SANZARU_REALTIME_MAX_SESSIONS=6       # concurrent realtime sessions across all acts
SANZARU_REALTIME_TURN_TIMEOUT=120     # per-turn stall bound; default 6x turn_seconds, min 60s
SANZARU_REALTIME_ACT_BUDGET=3000      # per-act wall clock; default 3000s, under the 60-min close
# Override stale list pricing: text_in,cached_text_in,audio_in,cached_audio_in,audio_out,text_out
# (USD per 1M tokens), optionally followed by a 7th per_minute value (USD per session-minute).
# Also the way to make an unlisted model usable *with* max_cost_usd: a ceiling
# over a model nothing can price is refused, not silently un-enforced.
SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1=4,0.4,32,0.4,64,24
SANZARU_REALTIME_PRICE_GPT_LIVE_1=0,0,0,0,0,0,0.05

# HTTP transport security (ignored on stdio). None of these load from `.env` —
# the loader is an allowlist and a planted file must not weaken transport auth;
# export them, or inject with `npx dotenv-cli -- sanzaru ...`.
SANZARU_HTTP_TOKEN="..."               # bearer token required on /mcp and /media
SANZARU_ALLOW_UNAUTHENTICATED_HTTP=1   # only when a proxy already authenticates; a
                                       # non-loopback bind refuses to start without one.
                                       # Requires SANZARU_ALLOWED_HOSTS.
SANZARU_ALLOWED_HOSTS=sanzaru.example.com,sanzaru.example.com:*  # Host values clients send;
                                       # keeps DNS-rebinding (Host + Origin) checks on off loopback
SANZARU_ALLOWED_ORIGINS=https://app.example  # optional; default is http(s):// of each host
SANZARU_IDENTITY_HEADER=x-forwarded-email  # opt-in: trust this proxy-injected header for caller
                                       # identity (multi-tenant). Unset = no header is trusted.
                                       # Set only behind a proxy that strips client copies;
                                       # a request carrying two copies is refused (400).
SANZARU_REQUIRE_USER_CONTEXT=1         # Databricks backend: refuse (403) rather than fall back
                                       # to the shared volume root when a request has no identity

# Signs simulated-podcast manifests and checkpoints so `--resume` refuses
# bookkeeping this installation did not write. Set it on shared filesystems.
SANZARU_RUN_SECRET="..."
```

**For MCP servers (Claude Desktop):**
Set environment variables explicitly in `.mcp.json` using template variables:
```json
{
  "mcpServers": {
    "sanzaru": {
      "command": "uv",
      "args": ["run", "sanzaru"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "SANZARU_MEDIA_PATH": "${SANZARU_MEDIA_PATH}"
      }
    }
  }
}
```

**Note:** `LOG_LEVEL` can be optionally added to the `env` object if you want to override the default (INFO).

**For local development with .env files:**
1. Run `./setup.sh` for interactive setup, or manually copy `.env.example` to `.env`
2. python-dotenv ships with sanzaru as a runtime dependency; nothing to install
3. Run Claude with dotenv-cli to inject env vars: `npx dotenv-cli -- claude` (or `bunx dotenv-cli -- claude`)

This approach makes environment configuration explicit and avoids confusion from implicit `.env` loading.

## Storage Backend

All file I/O goes through a pluggable `StorageBackend` protocol (`src/sanzaru/storage/protocol.py`). Tools call `get_storage()` to get the singleton backend — they never touch the filesystem directly.

### Configuration

```bash
STORAGE_BACKEND="local"       # Default — uses SANZARU_MEDIA_PATH (or individual paths)
STORAGE_BACKEND="databricks"  # Databricks Unity Catalog Volumes via Files API
```

**Databricks backend** requires:
```bash
DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
DATABRICKS_CLIENT_ID="..."
DATABRICKS_CLIENT_SECRET="..."
DATABRICKS_VOLUME_PATH="/Volumes/catalog/schema/volume"

# Optional, multi-tenant deployments only — see Multi-Tenant Support below
SANZARU_REQUIRE_USER_CONTEXT=1
```

### Protocol Methods

| Method | Purpose |
|--------|---------|
| `read(path_type, filename)` | Read full file → `bytes` |
| `write(path_type, filename, data)` | Write file → display path |
| `write_stream(path_type, filename, chunks)` | Stream write (async iterator) |
| `stat(path_type, filename)` | Get `FileInfo(name, size_bytes, modified_timestamp)` |
| `exists(path_type, filename)` | Check existence → `bool` |
| `list_files(path_type, pattern, extensions)` | List with filtering → `list[FileInfo]` |
| `local_path(path_type, filename)` | Context manager yielding `pathlib.Path` |
| `local_tempfile(path_type, filename)` | Context manager for writing (uploads on exit) |

### Multi-Tenant Support (Databricks)

For deployments where multiple users share one server (e.g., Databricks Apps), the `user_context` module provides per-user storage isolation. When middleware sets a `UserContext` via `set_user_context()`, the Databricks backend automatically prefixes volume paths with a user slug:

```
/Volumes/{volume_path}/{user_slug}/videos/{filename}
```

The user slug is `<readable>-<hash>`: the lowercased email local part with anything outside `[a-z0-9_]` folded to `_`, then the first 12 hex characters of a SHA-256 over the full address (e.g., `rcaputo3@tjclp.com` → `rcaputo3-058695b50073`). The hash is what makes the prefix injective — the readable half alone dropped the domain and folded punctuation, so `jane.doe@corp.com`, `jane-doe@corp.com` and `jane.doe@other.com` all shared one directory (CWE-706). A local part with nothing in `[a-z0-9_]` yields `user-<hash>` rather than an error. The slug **is** a storage path: do not change `user_slug` without a migration.

**Migration from 0.10.x and earlier:** the slug used to be the bare local part (`rcaputo3`), so every tenant's existing files live under a prefix the code no longer resolves to (`/Volumes/{volume_path}/rcaputo3/videos/…` → now `/Volumes/{volume_path}/rcaputo3-058695b50073/videos/…`). The storage protocol has no move or delete, so either move each `{old_slug}/` tree to `{user_slug(email)}/` with Databricks tooling (`user_slug` is importable from `sanzaru.user_context` to compute the new names), or re-derive the old slug read-side in a deployment-local subclass until the move is done. Nothing in the server does this for you.

When no user context is set (default), paths are unchanged. On a shared deployment that fallback is the wrong answer — a request with no identity would be served out of everybody's directory — so set `SANZARU_REQUIRE_USER_CONTEXT=1` to refuse it instead. Strict boolean: `1/true/yes/on` requires an identity, `0/false/no/off` (or unset) allows the shared root, and any other value is a configuration error rather than a silent "off". The refusal is `UserContextRequiredError` (a `RuntimeError`, so the CLI reports it as `config`/exit 3 — the CLI never carries an identity, so setting the switch there is a mismatch, not an internal failure); an authenticating HTTP transport should catch it by name and answer 403.

### Known Limitations (Databricks)

- **`write_stream()` buffers in memory** — Databricks Files API requires a complete PUT body. For typical Sora videos (20-60 MB) this is acceptable; monitor memory for very large files.
- **`stat()` returns `modified_timestamp=0.0`** — HEAD response doesn't include mtime.
- **`local_path()` downloads to temp file** — Libraries needing filesystem access (PIL, pydub) get a temp copy that's cleaned up on context exit.

### Known Limitations (Podcast)

- **Memory footprint**: All segment audio is held in memory during generation. For a 30-minute podcast (~45 MB at 192k MP3), peak memory is approximately 2x the final file size. Monitor for very long podcasts (60+ minutes).

## Media Viewer (MCP App)

The `view_media` tool opens an interactive media player rendered directly in the conversation via the MCP Apps protocol.

### Architecture

```
view_media(media_type="audio", filename="track.mp3")
  → Returns metadata + meta.ui.resourceUri
  → Host loads ui://sanzaru/media-viewer.html (bundled React app)
  → React app calls _get_media_data via callServerTool (2MB chunks)
  → Assembles chunks → Blob URL → <video> / <audio> / <img>
```

### HTTP Route

In HTTP transport mode, a direct route serves raw bytes with no base64 overhead:
```
GET /media/{type}/{name}  →  raw bytes + Content-Type header
    Authorization: Bearer <SANZARU_HTTP_TOKEN>
```

It is for **programmatic clients that can send the bearer header** — an agent
fetching a rendered file, a proxy. A browser media element (`<video src>`, `<img src>`)
cannot attach `Authorization`, and every response carries `Content-Disposition:
attachment` and `nosniff`, so it is not a playback URL: in an authenticated
deployment the `_get_media_data` chunking path (which the bundled React app already
uses) is the only one that works in the viewer, over both stdio and HTTP. The route
applies the same Host/Origin policy as `/mcp` and serves an allowlisted content type
or `application/octet-stream`.

### Frontend Development

The React app lives in `src/sanzaru/app/media-viewer/`. The built HTML (`dist/mcp-app.html`) is committed to the repo and shipped as Python package data — no Node/Bun needed at install time.

```bash
cd src/sanzaru/app/media-viewer
bun install && bun run build   # Only needed when modifying the frontend
```

## Transport Modes

Sanzaru supports two transport modes for different deployment scenarios:

### stdio (Default)
Standard I/O transport for MCP clients like Claude Desktop:
```bash
uv run sanzaru
```

**Use cases:**
- Claude Desktop integration
- Local MCP client connections
- Development and testing

**Configuration:** Set environment variables in `.mcp.json` or via dotenv-cli

### http (Stateless HTTP Streaming)
HTTP streaming transport for web clients and remote access:
```bash
# Local HTTP server (loopback: token optional, the SDK's loopback Host/Origin allowlist applies)
uv run sanzaru --transport http

# Any other bind requires a bearer token; without one the server refuses to start (exit 3)
export SANZARU_HTTP_TOKEN="$(openssl rand -hex 32)"
uv run sanzaru --transport http --host 0.0.0.0 --port 3000
# ...and clients send: Authorization: Bearer <token>   (on /mcp and /media)

# Behind a proxy that already authenticates (Databricks Apps): no token, but the
# proxy's hostnames must be named so the Host/Origin checks stay on
SANZARU_ALLOW_UNAUTHENTICATED_HTTP=1 SANZARU_ALLOWED_HOSTS=myapp.example.com \
  uv run sanzaru --transport http --host 0.0.0.0 --port 8000
```

**Authentication model:** `SANZARU_HTTP_TOKEN` is checked on every request by
`BearerTokenMiddleware` (constant-time, as bytes). Off loopback, `SANZARU_ALLOWED_HOSTS`
keeps the SDK's DNS-rebinding protection on for the deployment's real hostnames —
Origin included, which a browser page cannot forge; with a token and no allowlist the
check is switched off and the token is the control. `SANZARU_ALLOW_UNAUTHENTICATED_HTTP=1`
drops the token requirement only together with an allowlist. See `build_http_app` in
`server.py` for the exact policy.

**Use cases:**
- Web-based MCP clients
- Remote server deployments
- Multi-user access (stateless, no session management)
- Browser-based integrations

**Endpoints:** MCP tools available at `http://{host}:{port}/mcp`

**Key features:**
- **Stateless by design:** No session IDs required (all state lives in OpenAI's cloud)
- **SSE streaming:** Server-Sent Events for real-time communication
- **CORS support:** Can be configured via Starlette middleware (see Python MCP SDK docs)

**Production deployment:**
For advanced deployments with CORS or custom middleware, build the app with
`build_http_app()` and hand it to your own ASGI server. It is the *same* app
`sanzaru --transport http` serves — bearer auth, identity middleware, the `/media`
route and the bind-appropriate Host/Origin policy included. Do **not** mount the bare
`mcp.streamable_http_app()`: it carries none of the middleware, so `/mcp` is open even
with `SANZARU_HTTP_TOKEN` exported (and a `Mount` never runs its lifespan, so the
session manager is never started either).
```python
from starlette.middleware.cors import CORSMiddleware
from sanzaru.server import build_http_app

# Pass the address you will bind — the token requirement and Host/Origin policy
# are derived from it. Raises ConfigurationError for an unsafe combination.
app = build_http_app(host="0.0.0.0", port=8000)
app = CORSMiddleware(app, allow_origins=["https://client.example"], expose_headers=["Mcp-Session-Id"])
```
It is safe to call more than once: mcp 2.x builds a fresh session manager on every
`streamable_http_app()` call, with the stateless flag and Host/Origin policy passed as
arguments rather than read from server settings (`mcp.settings.stateless_http` and
`mcp.settings.transport_security` no longer exist — that is the 1.x API).

**Note:** The `/media` route does not include CORS headers by default. If you need cross-origin access to media files (e.g., from a browser-based client), wrap with CORSMiddleware as shown above.

## Model Selection Guidelines

### Video Generation (Sora)
**sora-2**: Faster, cheaper, good for iteration and testing
**sora-2-pro**: Slower, higher quality, for final production (supports larger resolutions)

**Supported video sizes:**
- Both models: `720x1280`, `1280x720`
- Pro only: `1024x1792`, `1792x1024`

### Image Generation

**Three tools available:**

| Tool | API | Best For |
|------|-----|----------|
| `generate_image` | Images API | Simple one-shot generation — no polling needed (RECOMMENDED DEFAULT) |
| `create_image` | Responses API | Parallel generation, iterative refinement chains |
| `edit_image` | Images API | Editing existing images |

Generation (`generate_image`, `create_image`) defaults to `DEFAULT_IMAGE_MODEL` (gpt-image-2.5-flare);
`edit_image` defaults to `DEFAULT_IMAGE_EDIT_MODEL` (gpt-image-2.5-sunburst). Both live in `config.py`.

**Image generation models:**
- **gpt-image-2.5-flare**: DEFAULT for generation — OpenAI's pick for fast, high-quality everyday images
- **gpt-image-2.5-sunburst**: DEFAULT for edits — tuned for editing precision. Both 2.5 variants (snapshot
  `2026-09-08`) are priced like gpt-image-2 and add transparent backgrounds plus `quality="xhigh"|"max"`
- **gpt-image-2**: Previous flagship — ~99% text accuracy, up to 4K output, any valid resolution
- **gpt-image-1.5**: Older gen — transparent backgrounds and `input_fidelity`, fixed sizes only
- **gpt-image-1**: High quality
- **gpt-image-1-mini**: Fast, cost-effective
- **dall-e-3** / **dall-e-2**: Legacy

**Supported image sizes:**
- Common: `1024x1024`, `1024x1536`, `1536x1024`, `auto`
- gpt-image-2.5 and gpt-image-2 also: `2048x2048`, `2048x1152`, `3840x2160`, `2160x3840`, plus any
  resolution with max edge ≤3840px, multiples of 16, ratio ≤3:1, and 655,360 ≤ pixels ≤ 8,294,400.

**Per-model rules live in `image_models.py`, not at the call sites.** `capabilities_for()` resolves a
model (dated snapshots included) to one row; `check_background` / `check_quality` /
`honors_input_fidelity` are what the three tools call. The rules the table encodes:
- gpt-image-2 does NOT support `background="transparent"` (raises before the request)
- gpt-image-2 **and gpt-image-2.5** reject `input_fidelity` (always high fidelity) — the wrappers strip it
  rather than let the API 400. The SDK docstring claims 2.5 honours it; the endpoint said otherwise on
  2026-09-15, and the endpoint wins
- `quality="xhigh"` and `"max"` exist only on gpt-image-2.5; other models raise with the accepted list
- Unknown models (dall-e-*, anything newer than the table) get no client-side rules; the API decides

**Example with generate_image (recommended default — synchronous):**
```python
# Images API - blocks until done, returns token usage
generate_image(
    prompt="a futuristic cityscape at sunset",
    size="1536x1024",
    quality="high",
)  # defaults to model="gpt-image-2.5-flare"
```

**Example with create_image (parallel/refinement workflows):**
```python
# Responses API - async, supports iterative refinement and action field
resp = create_image(
    prompt="a futuristic cityscape at sunset",
    tool_config={
        "type": "image_generation",
        "model": "gpt-image-2.5-flare",
        "quality": "max",
        "size": "1536x1024",
    },
)
# poll with get_image_status(resp.id), then download_image(resp.id)
```

## Type Safety Notes

- `VideoSeconds` must be string literal: `"4"`, `"8"`, or `"12"` (NOT integers)
- `VideoSize` and image sizes are string Literals enforced by type system
- Use `omit` from `openai._types` when converting `None` to SDK parameters
- All async functions use `AsyncOpenAI` client

## Code Style

- Line length: 120 characters
- Format with ruff (double quotes, space indentation)
- Tool descriptions support E501 ignore for readability
- Comprehensive docstrings on all MCP tools
- Security-first: path traversal protection on all file operations

### Naming Conventions

**Function Names: Verb-First (Predicate-First)**
- Internal functions use `verb_noun` pattern (action comes first)
- Examples:
  - ✅ `create_video()` - verb first
  - ✅ `download_image()` - verb first
  - ✅ `list_reference_images()` - verb first
  - ✅ `get_video_status()` - verb first

**MCP Tool Names (Public API): Keep "sora" prefix for branding**
- Server wrapper functions can keep descriptive names for MCP tools
- Example: MCP tool `create_video` → calls internal `create_video()`

**Description Constants: Match tool names**
- `CREATE_VIDEO`, `DOWNLOAD_IMAGE`, `LIST_REFERENCE_IMAGES`
- ALL_CAPS with underscores
- Verb comes first, matches function structure

## Testing

**Test Structure:**
```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests for pure functions (46 tests)
│   ├── test_utils.py
│   ├── test_security.py
│   └── test_image_processing.py
└── integration/         # Integration tests with mocked clients (12 tests)
    ├── test_video_tools.py
    ├── test_image_tools.py
    └── test_reference_tools.py
```

**Style Notes:**
- **NO `__init__.py` files in test directories** - tests are not a package
- Use `@pytest.mark.unit` for unit tests (pure functions, no mocking)
- Use `@pytest.mark.integration` for integration tests (mocked OpenAI client)
- Use `pytest-mock` (mocker fixture) for all mocking
- Ignore SIM117 in tests (nested `with` intentional for pytest.raises)

**Running Tests:**
```bash
pytest                        # All tests
pytest tests/unit -m unit     # Unit tests only (fast)
pytest tests/integration      # Integration tests only
pytest --cov=src              # With coverage report
```

**Coverage Goals:**
- Pure functions: 100% (achieved)
- Tools (business logic): 80%+ (achieved: 82-88%)
- Overall: 65%+ (achieved)
- Always remember to read `.venv` files for type information about external libraries. NEVER use `typing.Any`.
