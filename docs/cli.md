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

`result.file.path` is the canonical location of a written artifact and is always present. Where a
result also carries a bare name (`output_file`, `output_filename`, `filename`), it is the basename
of that same path — including when `-o` renamed the file or it was staged under a temporary name
first. The two never disagree, so `jq -r .result.file.path` and `jq -r .result.output_file` always
describe one file. `video` envelopes carry only `file.path`, no bare name; prefer `file.path` in
scripts that handle more than one media type.

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
| `speak TEXT` | TTS (`--provider`, `--model`, `--voice`, `--instructions`, `--speed`, `--voice-settings`); long text auto-chunks |
| `convert FILE` | To mp3/wav (`--to`) |
| `compress FILE` | Fit a size budget (`--max-mb`, default 25) |
| `files` | List with filters (`--pattern/--format/--min-duration/...`); `--latest` prints only the newest |

### TTS providers

`audio speak` and `podcast generate` both take `--provider openai|elevenlabs` (default `openai`).

| | `openai` | `elevenlabs` |
|---|---|---|
| `--model` | `gpt-4o-mini-tts` (default), `tts-1`, `tts-1-hd` | `eleven_v3` (default), `eleven_multilingual_v2`, `eleven_flash_v2_5`, `eleven_turbo_v2_5` |
| `--voice` | named voice (default `alloy`) | a voice id from your library — **required** |
| `--instructions` | style direction | ignored; use inline `[audio tags]` in the text with `eleven_v3` |
| `--speed` | 0.25–4.0 | 0.7–1.2, and `eleven_v3` rejects any change |
| `--voice-settings` | rejected | JSON: `stability`, `similarity_boost`, `style`, `use_speaker_boost`, `speed` |

ElevenLabs needs `ELEVENLABS_API_KEY` and `uv pip install 'sanzaru[elevenlabs]'`; either missing is
a config error (exit 3). Set `SANZARU_ELEVENLABS_MAX_CONCURRENCY` if you hit HTTP 429 — their cap
is per subscription tier. `ELEVENLABS_BASE_URL` overrides the API endpoint (the ElevenLabs
counterpart to `OPENAI_BASE_URL`), for sandboxes that reach the API through a credential proxy.

```bash
sanzaru audio speak "[excited] You will not believe this." \
  --provider elevenlabs --voice 21m00Tcm4TlvDq8ikWAM \
  --voice-settings '{"stability":0.4,"similarity_boost":0.85}' -o hook.mp3
```

### `sanzaru podcast`

Three verbs, in the order you use them:

| verb | what it does | cost |
| --- | --- | --- |
| `rundown PREMISE` | plans an episode into acts; emits editable JSON | one text call |
| `simulate BRIEF` | records realtime agents actually conversing | real; see below |
| `generate SCRIPT` | speaks a script you wrote (multi-voice TTS) | TTS rates |

Reach for `simulate` when you have a **topic** and want a real conversation, `generate
--render-mode dialogue` when you have a **script** and want it performed naturally, and
`generate` (segments) when you need exact control over gaps and per-segment retry.

#### `rundown`

`rundown PREMISE` (inline text, `@file`, or `-`) expands a premise into acts and writes JSON you
can hand-edit before spending anything. Options: `--acts`, `-m/--target-minutes`, `--title`,
`--style`, `--host NAME[:VOICE[:PERSONA]]` (repeatable), `--turn-seconds`, `--model` (a *text*
model, not a realtime one), `-o`.

It exists as its own command because planning is cheap and recording is not. Acts record in
parallel and cannot hear each other, so each carries `prior_context` (what earlier acts covered),
`upcoming` (what later acts own), and `handoff` (where to leave off) — that wiring is the whole
point of planning first, and it is easier to fix in an editor than in a prompt.

```bash
sanzaru podcast rundown "why TTS providers drop sentence tails" \
  --acts 3 -m 6 \
  --host "Avery::You host and translate jargon." \
  --host "Rory:cedar:You chased the bug. Dry, specific." \
  -o rundown.json
```

#### `simulate`

`simulate [BRIEF]` records the episode. BRIEF is a rundown or a full SimulationBrief (inline JSON,
`@file`, `-`); flags override it. Or skip BRIEF and pass `--premise` to plan and record in one go.

Nothing is scripted: each host is a `gpt-realtime` session with a persona, and one host's audio is
played into the others' ears. The transcript comes back in the envelope as an *output*.

**Always `--dry-run` first.** It plans, projects turns/duration/tokens/dollars, and records
nothing:

```bash
sanzaru podcast simulate @rundown.json --dry-run
```
```
sanzaru: dry run — 'The Hard Part Isn't the Model': 3 acts, up to 27 turns
sanzaru:   act1: The Model Is the Easy Part to Demo — 120s, up to 9 turns
sanzaru: projected ~6 min audio, 16,593 input / 10,800 output tokens
sanzaru: projected cost ~$0.20 (estimate, not a quote)
sanzaru: nothing was recorded; drop --dry-run to record
```

Then record with a ceiling. `--max-cost` is checked after every turn across every parallel act:

```bash
sanzaru podcast simulate @rundown.json --model gpt-realtime-2.1-mini \
  --max-cost 2.00 --stems -o ./out/ep1.mp3
```

Progress is one greppable stderr line per turn and per act, each carrying elapsed wall clock —
during a multi-minute blocking run that is the only signal it is alive:

```
sanzaru: run d33730ea — resume with: sanzaru podcast simulate --resume d33730ea
sanzaru: act 1/3 turn 4 [Rory] 14.2s t=26s
sanzaru: act 1/3 recorded 9 turns, 122s audio (complete) t=28s
sanzaru: qc: transcribing 3 acts with gpt-transcribe t=32s
sanzaru: episode d33730ea: 3 acts, 26 turns, 6.8 min
sanzaru: spend $0.21
sanzaru: qc warn: act2 — see result.qc for why (--qc-retry re-records just those)
```

**Directing it.** A producer inside the tool gives one host the floor at a time, pushes talking
points across each act, and steers the last turns to a landing — but those are defaults. Each act
in the rundown takes `direction` (free text, how to play it), `turn_notes` (`{"0": "..."}` by turn
index, replacing the generated note) and `speaking_order` (host ids, cycled, instead of strict
alternation). `turn_notes` is the strongest lever here: it is the difference between "move onto
the next point" and "object to what they just said". Edit them in the rundown JSON — the tool
blocks while acts record in parallel, so there is no live steering.

**Recovery.** The run id prints *before* recording starts, and every act is checkpointed to the
audio dir the moment it finishes. An interrupt, a crash, or a `--max-cost` abort never loses audio
you paid for:

```bash
sanzaru podcast simulate --resume d33730ea   # records only the missing acts
```

The printed id goes to **stderr**, so choose it yourself when a harness parses only stdout:
`--run-id ep1`, or a top-level `"run_id"` in the rundown JSON (the flag wins). Then `--resume ep1`
is predictable even if the shell died before you read anything.

`--qc-retry` re-records flagged acts, and the take it replaces is preserved beside it as
`<slug>_<run>_<act>_take1.mp3` (`_take2`, … on later retries) — never read by `--resume`, so the
run keeps exactly one truth while you stay free to assemble the best cut per act. QC verdicts do
disagree run-to-run, so a retry is not automatically the better take.

This composes with `-o`: the episode and stems go where you asked, while the manifest and the act
checkpoints always stay in the media dir, so the printed resume command works verbatim with no
`-o` of its own. A resume also reinstates the run's settings from the manifest — including
`--max-cost`, so following the ceiling abort's hint does not re-run uncapped. Anything you pass on
the resume itself wins. Because the restored ceiling also counts the spend replayed from the
checkpoints, the ceiling abort prints a resume command with a *raised* `--max-cost`, and a resume
that cannot fit under the restored one stops before it records anything.

Options: `-p/--premise`, `--acts`, `-m/--target-minutes`, `--title`, `--style`, `--host`,
`--model`, `--planner-model`, `--turn-seconds`, `--turn-tokens`, `--max-cost`, `--max-sessions`,
`--resume RUN_ID`, `--run-id RUN_ID`, `--stems`, `--qc/--no-qc`, `--qc-retry`, `--dry-run`,
`--act-gap`, `--format`, `--bitrate`, `-o`.

Exit codes are the usual contract plus one: **6** means the cost ceiling stopped the run — the
envelope carries `spent_usd`, `suggested_limit_usd`, `completed_acts`, and a `resume` command.
Every other failure after recording starts carries `run_id` and a `resume` command too.

Full rationale, measured numbers, and tuning notes:
[`docs/audio/simulated-podcasts.md`](audio/simulated-podcasts.md).

#### `generate`

`generate SCRIPT` renders a multi-voice podcast from a PodcastScript JSON; segments TTS in
parallel internally, bounded per provider. Only `speakers` and `segments` are required — the
smallest script that renders is
`{"speakers": [{"name": "Alex", "voice": "ash"}], "segments": [{"speaker": "Alex", "text": "Hi."}]}`.
A speaker's `id` defaults to its `name` (so segments can reference it by name) and `speed` to
`1.0`; `instructions` is optional and OpenAI-only. `title` defaults, and `config` is optional
in full: `default_pause_ms` (600), `normalize_loudness` (true), `output_format` (`"mp3"`),
plus `intro_silence_ms`, `outro_silence_ms`, `output_bitrate`, `provider`, `max_concurrency`,
`render_mode`, `dialogue_stability`. An invalid script reports every problem at once rather
than one per run. Speakers accept optional `provider`, `model`, and `voice_settings`, resolved
as `speaker.provider > config.provider > --provider` — so one episode can mix OpenAI and ElevenLabs
voices. The envelope includes the full transcript — pipe to a file for long episodes.

#### Render modes

`--render-mode segments|dialogue` (or `config.render_mode`; default `segments`).

- **`segments`** — one TTS request per turn, joined with your configured silence gaps. Full control,
  and every segment is independent so a single bad render can be retried on its own.
- **`dialogue`** — consecutive turns sharing a dialogue-capable provider and model (currently
  ElevenLabs `eleven_v3`) are sent as **one** request, so the model paces the exchange itself.
  Noticeably more natural back-and-forth.

Grouping is per-run, not per-episode: turns that can't join a run — OpenAI speakers, other models,
a lone turn, a stretch in one voice, a turn that alone fills the request budget — still render per segment,
so mixed episodes keep working. On an 11-segment demo with an OpenAI host and two `eleven_v3`
guests, 8 segments batched into 3 dialogue requests while the 3 host turns rendered individually.

Inside a dialogue run, `pause_after` is ignored (the model owns pacing) and per-speaker
`voice_settings`/`speed` don't apply — the endpoint takes a single `config.dialogue_stability`
(0–1) for the whole request.

**The trade: dialogue buys pacing and sells partial retry.** A run is one request, so it is
all-or-nothing. If a single line comes out wrong there is no way to re-render just that line —
fixing it re-spends every character in the batch. `segments` renders each turn independently, so a
bad one costs only itself.

That bites hardest on ElevenLabs, where quota is drawn down by the characters you submit and tiers
can be small (the free tier was 10,000 characters/month as of 2026-08; check your account rather
than trusting this number). Compounding it: inside a batched run all your direction has to live in
**inline audio tags** (`[whispers]`), since `pause_after` and per-speaker `voice_settings` are inert
there — and tags count too. So the expressive mode is also the one where a retry costs the most. The
production run behind this note was a single 11-turn dialogue request totalling 1,730 characters;
one bad line would have cost all 1,730 again. (Per-run, not per-episode — an episode split across
three requests only re-spends the one containing the bad line.)

**Watch total run length, not just turn length.** The 2000-character budget is per *request* —
the sum of consecutive turns — so where a run splits decides what actually gets batched. The
planner never emits an over-budget request, so nothing fails; the consequence is quieter than a
failure. A turn left alone in its own voice after a split renders as an ordinary segment: exactly as
`segments` mode would, so it *regains* its `pause_after` and per-speaker `voice_settings`/`speed`,
but without the model-paced turn-taking you chose `dialogue` for. It costs no extra characters —
what you lose is pacing, not quota.

Two ways to land there, and the first is both likelier and quieter:

- **A split strands the tail.** Three 900-character turns (`a, b, a`) batch as one request of turns
  1–2, leaving turn 3 alone as a segment — with no turn anywhere near the ceiling.
- **One over-long turn takes its neighbours with it.** `a, b, a` with a 2500-character middle
  renders *all three* as segments, because the flush leaves each short turn single-voice too. This
  one at least announces itself: with no run left to batch, the render logs a warning.

So "keep every turn short" is not sufficient advice on its own — plan the running total, and expect
the tail of a split run to fall back when it lands single-voice.

**You don't have to work it out in your head.** Every dialogue render logs

```
Dialogue mode: N/M segments batched into K conversation request(s)
```

on stderr, where `M - N` is how many turns did *not* batch. In an all-ElevenLabs episode those are
exactly your stranded turns: the first example above reports `2/3 segments batched into 1
conversation request(s)`, and the 1 is turn 3. In a **mixed** episode the gap also counts turns that
were never eligible — OpenAI speakers, non-`eleven_v3` models — so treat it as an upper bound there
and read the shortfall against the turns you expected to batch.

(The ceiling exists because an over-budget request can terminate the stream mid-conversation,
indistinguishable from a complete take, so the provider layer refuses one outright rather than
return a short one.)

Rules of thumb: `segments` for exact gap control, per-speaker tuning, cheap retry, or a tight
character budget; `dialogue` for natural conversation on a script you're confident in.

```bash
sanzaru podcast generate @episode.json --render-mode dialogue -o ep.mp3
```

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
