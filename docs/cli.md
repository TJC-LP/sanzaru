# Sanzaru Agent CLI

`sanzaru <group> <verb>` exposes every Sanzaru capability as a shell command designed for AI
agents: machine-readable JSON on stdout, deterministic exit codes, one-shot workflows for
long-running jobs, and output to any path you choose. The CLI is **stateless** — job IDs are the
only handles, every command is independently resumable, and nothing is tracked between
invocations.

Bare `sanzaru` (no subcommand) still starts the MCP server, exactly as before — existing
`.mcp.json` and Claude Desktop configs are unaffected. `sanzaru serve` is the explicit alias.

```bash
uv tool install sanzaru        # or: uvx sanzaru ..., pipx install sanzaru
export OPENAI_API_KEY=sk-...
sanzaru capabilities            # no API key needed — discover what works here
```

## The contract

**stdout** carries exactly one JSON envelope per input — nothing else. Fan-out commands stream
one envelope per line (JSONL) in completion order. **stderr** carries progress, heartbeats, and
human-readable hints. A TTY only switches formatting (pretty vs compact), never structure.

```json
{"v": 1, "ok": true,  "command": "video.create", "result": {"id": "video_x", "...": "...", "file": {"path": "/abs/clip.mp4", "bytes": 48211939}}, "elapsed_s": 184.2}
{"v": 1, "ok": false, "command": "video.wait", "error": {"type": "timeout", "message": "..."}, "resume": "sanzaru video wait video_x --download -o ./clip.mp4", "id": "video_x", "last_status": "in_progress", "last_progress": 78}
```

Errors are **also** emitted as envelopes on stdout (`"ok": false`) so `jq` pipelines never hang,
with a one-line summary on stderr. `error.type` is one of: `usage`, `config`, `api_error`,
`not_found`, `job_failed`, `timeout`, `download_error`, `internal`. A `resume` field is present
whenever a follow-up command recovers the situation.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime/API error (network, 4xx/5xx, write failure, unknown ID) |
| 2 | Usage error (bad flags/arguments; also click's own errors) |
| 3 | Configuration error (missing `OPENAI_API_KEY`, missing optional extra) |
| 4 | Timeout — the job **keeps running server-side**; re-run the `resume` command |
| 5 | Job failed server-side (moderation, generation error) |
| 6 | Partial batch failure (fan-out with ≥1 success and ≥1 failure) |
| 130 | Interrupted (Ctrl-C) — job keeps running; resume hint on stderr |

When **every** input of a fan-out fails, the exit code is deterministic: 4 if any input timed out
(resumable work remains), otherwise the highest per-input code. Per-line envelopes carry the
detail either way.

### Global flags

`--json` (accepted no-op — output is always JSON) · `-q/--quiet` (suppress stderr progress) ·
`-v/--verbose` (debug logging) · `--media-dir DIR` (override `SANZARU_MEDIA_PATH` for this
invocation). Durations accept `90`, `90s`, or `5m` forms.

## Async jobs: create → wait → download

`video create`/`video remix`/`image create` submit a job and return its ID in ~1 second.
`status` peeks (never blocks); `wait` blocks with adaptive polling; `download` fetches the
artifact. **Flag implication: `-o` ⇒ `--download` ⇒ `--wait`** — so one command composes all
three:

```bash
sanzaru video create "the pilot looks up and smiles" --seconds 8 --size 1280x720 \
  -o ./assets/pilot.mp4 | jq -r .result.file.path
```

While waiting, stderr gets a line per state change plus a 30s heartbeat
(`sanzaru: video_x in_progress 42% t=95s`). Polling adapts per job type (video: 5s → 20s cap,
default timeout 30m; image: 2s → 10s cap, default 10m); `--poll-interval` fixes the cadence and
`--timeout` sets the deadline.

**Waiting is idempotent.** On exit 4 the job is still running — the envelope's `resume` field is
a complete command to attach again. This is the crash/timeout recovery loop for harnesses that
cap foreground commands:

```bash
ID=$(sanzaru video create "..." --seconds 8 | jq -r .result.id)   # returns in ~1s
# ... do other work, then repeat until exit != 4:
sanzaru video wait "$ID" --download -o ./out/clip.mp4 --timeout 100s
```

`sanzaru wait` polls **mixed** job types concurrently — `video_*` and `resp_*` IDs are
dispatched by prefix (`--type` is the escape hatch) — and streams JSONL as each finishes:

```bash
sanzaru wait video_a1 video_b2 resp_c3 --download -o ./media/ > done.jsonl
```

## Output paths and inputs

Precedence for where artifacts land: `-o PATH` → `--media-dir` → individual env vars
(`VIDEO_PATH`/`IMAGE_PATH`/`AUDIO_PATH`) → `SANZARU_MEDIA_PATH/{videos,images,audio}` → the
**current directory** (with a stderr note — never a hard error). `-o` pointing at a directory
(existing, or with a trailing `/`) auto-names the file inside; parent directories are created.

Inputs (`--input-ref`, `--input-image`, `--mask`, audio `FILE`s) accept either a **path** (used
as-is) or a **bare filename** resolved from the configured media dir, matching MCP behavior.
v1 constraints: same-type path inputs must share one directory; bare filenames and paths can't
be mixed for the same media type; with `STORAGE_BACKEND=databricks`, `-o` always produces a
local file (bytes are copied out of the volume when needed).

## Long content

Prompts, TTS text, and podcast scripts accept three forms: inline string, `@file`, or `-`
(stdin). A literal leading `@` escapes as `@@`.

```bash
sanzaru audio speak @chapter1.txt --voice nova -o ch1.mp3
sanzaru podcast generate - < episode.json -o ./out/episode.mp3
```

## Command reference

### `sanzaru video` — Sora jobs
| Command | Purpose |
|---------|---------|
| `create PROMPT` | Submit a job. `--model sora-2\|sora-2-pro`, `--seconds 4\|8\|12`, `--size`, `--input-ref`, one-shot flags |
| `remix ID PROMPT` | Submit a remix of a completed video (new job ID; same one-shot flags) |
| `status ID` | Peek at status + progress (never blocks) |
| `wait ID...` | Block until terminal; concurrent multi-ID, `--download`, JSONL output |
| `download ID` | Fetch artifact: `--variant video\|thumbnail\|spritesheet`, `-o` |
| `list` | Cloud jobs (`--limit/--after/--order`) |
| `delete ID` | Permanently delete from OpenAI storage |
| `files` | Locally downloaded videos (`--pattern/--type/--sort/--order/--limit`) |

With `--input-ref`, keep the prompt motion-only — the image already carries character, setting,
and style (see `docs/sora-prompting-guide.md`).

### `sanzaru image` — two generation paths
| Command | Purpose |
|---------|---------|
| `generate PROMPT...` | **Synchronous** Images API — returns file + token usage. RECOMMENDED for one-off images. Multi-prompt × `--count` fan-out with `--concurrency` |
| `edit PROMPT` | Synchronous edit/composition of existing images (`--input-image`, `--mask`) |
| `create PROMPT` | **Async** Responses job — for refinement chains (`--previous-id`) and parallel jobs. `--image-model` (default gpt-image-2), `--input-image`, `--mask`, one-shot flags |
| `status ID` / `wait ID...` / `download ID` | The async job trio |
| `prepare INPUT` | Resize to Sora dimensions (`--size`, `--mode crop\|pad\|rescale`) |
| `files` | Images in the media dir |

Iterative refinement:

```bash
R1=$(sanzaru image create "a cyberpunk courier, full body" --size 1024x1536 --wait | jq -r .result.id)
sanzaru image create "add neon rain and a cityscape" --previous-id "$R1" -o ./art/courier_v2.png
```

gpt-image-2 (the default) does not support `--background transparent` — the guard raises a clear
usage error pointing to gpt-image-1.5.

### `sanzaru audio` — synchronous audio ops (requires `sanzaru[audio]`)
| Command | Purpose |
|---------|---------|
| `transcribe FILE...` | Whisper/GPT-4o transcription; `--enhance detailed\|storytelling\|professional\|analytical`, `--format`, `--timestamps`; multi-file fan-out |
| `chat FILE` | Ask questions about audio content (`--prompt`, `--system`) |
| `speak TEXT` | TTS (`--voice`, `--instructions`, `--speed`); long text auto-chunks |
| `convert FILE` | To mp3/wav (`--to`) |
| `compress FILE` | Fit a size budget (`--max-mb`, default 25) |
| `files` | List with filters (`--pattern/--format/--min-duration/...`); `--latest` prints only the newest |

### `sanzaru podcast`
`generate SCRIPT` renders a multi-voice podcast from a PodcastScript JSON
(`{"title", "speakers": [...], "segments": [...], "config": {...}}`); segments TTS in parallel
internally. `config` **requires** `default_pause_ms` (int), `normalize_loudness` (bool), and
`output_format` (`"mp3"|"wav"`); optional: `intro_silence_ms`, `outro_silence_ms`,
`output_bitrate`. The envelope includes the full transcript — pipe to a file for long episodes.

### Top-level
`wait ID...` (mixed-type poller) · `capabilities` (version, per-feature availability with
reasons, configured paths, storage backend, API-key presence, command map) · `serve` (MCP
server).

## Recipes

```bash
# Reference image → Sora pipeline
IMG=$(sanzaru image generate "futuristic pilot in mech cockpit" --size 1536x1024 -o ./work/ | jq -r .result.file.path)
REF=$(sanzaru image prepare "$IMG" --size 1280x720 --mode crop | jq -r .result.file.path)
sanzaru video create "the pilot glances up, takes a breath" --input-ref "$REF" --size 1280x720 --seconds 8 -o ./out/

# Batch assets with partial-failure retry
sanzaru image generate "app icon" "hero banner" "404 art" --quality high -o ./art/ > results.jsonl
jq -r 'select(.ok | not) | .input.prompt' results.jsonl    # exit 6 → retry just these

# Background one-shot for agent harnesses (progress lands in the log)
sanzaru video create "..." -o ./out/clip.mp4 --timeout 25m 2> progress.log &
```

## Implementation notes

- One `AsyncOpenAI` client per invocation is shared across every call (a 10-minute poll loop
  reuses one connection pool instead of a TLS handshake per poll).
- The polling loops live in `sanzaru/polling.py` (`wait_for_video`/`wait_for_image`) — pure
  async, adaptive backoff with jitter, transient 408/409/429/5xx retried until the deadline,
  404 fails fast.
- `-o` works by installing a per-invocation `LocalStorageBackend(path_overrides=...)` via
  `sanzaru.storage.set_storage_backend()`; the tool layer still validates basenames. The MCP
  server never touches these overrides.
- All commands are registered regardless of installed extras — a missing extra returns a
  `config` envelope (exit 3) with the exact install command, so `--help` output is stable
  everywhere.
