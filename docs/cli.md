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
Paths in one batch may span directories — each is validated individually under its own parent —
so an episode and its QC windows can be transcribed in one call:

```bash
sanzaru audio transcribe ./episodes/ep.mp3 ./windows/w1.mp3 ./windows/w2.mp3
```

Constraints: two inputs of the same media type cannot share a **basename** (tools are handed
bare names and the envelope reports bare names, so one would be unaddressable); bare filenames
and paths can't be mixed for the same media type; `-o` still resolves against a single directory
per type, the first input's; with `STORAGE_BACKEND=databricks`, `-o` always produces a local
file (bytes are copied out of the volume when needed).

Two rules exist because agents run this from workspaces holding untrusted files:

- **A bare name captured by the current directory is announced, never silent.** A bare name
  normally addresses the media library, but a same-named file in the working directory still
  wins (chained commands depend on it). When both exist as *different* files, the stderr note
  names both candidates and how to spell each, so a planted `episode.mp3` cannot quietly
  substitute itself for the library's (running from inside the media directory itself is one
  file, not two, and says nothing). Write `./name` to mean the local file with no note. A bare
  name whose local match is a **symlink** is refused outright — `./name` opts into that
  explicitly.
- **`-o` never writes through a symlink.** The destination, the staging file, and
  `podcast rundown -o` all open with `O_NOFOLLOW`, so a symlink pre-planted at the output path
  is an error (exit 2) rather than a write to whatever it points at — decided *before* anything
  is generated or billed, `rundown` included. When the refusal happens after the tool has
  already written (a relocation onto a planted link), the error names where the artifact still
  is and carries it as `file.path`. The staging name is random rather than derived from the
  output name, so it cannot be predicted and swapped mid-run. The direct-write case (no
  relocation) is checked at plan time here and closed against the race by the local storage
  backend's own `O_NOFOLLOW` opener.

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
| `create PROMPT` | **Async** Responses job — for refinement chains (`--previous-id`) and parallel jobs. `--image-model` (default gpt-image-2.5-flare), `--input-image`, `--mask`, one-shot flags |
| `status ID` / `wait ID...` / `download ID` | The async job trio |
| `prepare INPUT` | Resize to Sora dimensions (`--size`, `--mode crop\|pad\|rescale`) |
| `files` | Images in the media dir |

Iterative refinement:

```bash
R1=$(sanzaru image create "a cyberpunk courier, full body" --size 1024x1536 --wait | jq -r .result.id)
sanzaru image create "add neon rain and a cityscape" --previous-id "$R1" -o ./art/courier_v2.png
```

`generate`/`create` default to gpt-image-2.5-flare and `edit` to gpt-image-2.5-sunburst; both accept
`--background transparent` (png/webp output) and the extra `--quality xhigh|max` levels. Passing
`--model gpt-image-2` with either raises a clear usage error before any request is made.

### `sanzaru audio` — synchronous audio ops (requires `sanzaru[audio]`)
| Command | Purpose |
|---------|---------|
| `transcribe FILE...` | Whisper/GPT-4o transcription; `--enhance detailed\|storytelling\|professional\|analytical`, `--format`, `--timestamps`; multi-file fan-out; files over 8 min are auto-windowed (`result.chunked`) |
| `chat FILE` | Ask questions about audio content (`--prompt`, `--system`) |
| `speak TEXT` | TTS (`--provider`, `--model`, `--voice`, `--instructions`, `--speed`, `--voice-settings`); long text auto-chunks; `-o FILE` without an extension gets `.mp3` |
| `convert FILE` | To mp3/wav (`--to`); `-o FILE` without an extension gets `.<to>` |
| `compress FILE` | Fit a size budget (`--max-mb`, default 25); `-o FILE` must carry an audio extension — the result is mp3 when re-encoded but keeps the input's format when it was already small enough |
| `files` | List with filters (`--pattern/--format/--min-duration/...`); `--latest` prints only the newest. `--pattern` is a case-insensitive substring or glob (`"*.mp3"`, `"ep0?"`), **not a regex** — `^ $ | ( ) + \` are a usage error |

Audio names are checked before any work is done. An input to `convert`/`compress` must be a
self-contained audio container (`mp3 wav flac aac ogg oga opus m4a m4b mp4 mov aiff aif aifc wma mka
webm amr au caf mpeg mpga`); playlist formats such as `hls`, `m3u8`, `concat` and `dash` are refused
because their *content* names other local files for ffmpeg to read. A caller-named output of
`convert`, `compress` or `speak` must carry one of the narrower output extensions (`flac m4a mp3 mp4
mpeg mpga ogg wav webm`) — none of these commands can create a `.json` or `.html`.

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

#### Character usage and quota

ElevenLabs bills **characters submitted**, inline audio tags included, against a monthly
allowance that can be small (the free tier was 10,000 characters/month as of 2026-08). Every
render now reports what it spent, so you no longer have to count the script by hand first:

```jsonc
// audio speak
"result": {"output_file": "...", "provider": "elevenlabs", "model": "eleven_v3",
           "characters": 1730, "requests": 1}
// podcast generate — a list, since one episode can mix providers
"result": {"usage": [{"provider": "elevenlabs", "model": "eleven_v3",
                      "characters": 1730, "requests": 1}]}
```

The count is computed from the text actually submitted, so it is right even when a request
later fails, and it reflects chunking (long text is split into several requests).

To check the allowance *before* spending any of it:

```bash
sanzaru capabilities --quota | jq .result.elevenlabs_quota
# {"available": true, "tier": "free", "characters_used": 1754,
#  "character_limit": 10000, "characters_remaining": 8246, "resets_at_unix": ...}
```

`--quota` is the one part of `capabilities` that makes a network call and needs a key — the rest
stays safe as an agent's first command. A failed lookup is reported as
`{"available": false, "reason": "..."}` rather than failing the report.

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
| `generate SCRIPT` | speaks a script you wrote (multi-voice TTS); `--verify` checks the audio says it | TTS rates |

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

A dry run always projects, `--max-cost` or not. When a billable model — the episode's, or a
per-host `model` override in the rundown — has no known price, the projection reports it under
`cost.unpriced_models` and leaves `cost.usd` empty instead of quoting a figure it cannot stand
behind. That is also where you learn that the *recording* will be refused: a ceiling over a model
whose spend cannot be counted is not enforced silently, it exits **2** before anything is billed
(before the planner call, when only a premise was given). Set
`SANZARU_REALTIME_PRICE_<MODEL>` (`text_in,cached_text_in,audio_in,cached_audio_in,audio_out,text_out`
per 1M tokens, plus an optional seventh `per_minute` in USD per session-minute) or record without
a ceiling. If an unpriced model slips past that check and is
charged mid-run anyway, the run stops with exit 6 and the envelope names `unpriced_model` and
`price_env` — its `resume` command carries no `--max-cost`, because raising the cap cannot help.

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

A resume trusts only bookkeeping written for *that* run. The manifest must carry the run id it is
being resumed as, and each act checkpoint carries its run id and a digest of its mp3 — a pair copied
in from another run, or an mp3 swapped under its sidecar, is re-recorded rather than replayed. On a
shared media directory set `SANZARU_RUN_SECRET` as well: manifests and checkpoints are then
HMAC-signed and an edited or foreign one is refused (`not signed by this installation`, exit 2 for a
manifest; a checkpoint is re-recorded). The signature covers a fixed, versioned set of fields, so
upgrading sanzaru does not invalidate signed runs in flight; enabling the secret does invalidate
the unsigned files that already exist, so switch it on between runs.

Options: `-p/--premise`, `--acts`, `-m/--target-minutes`, `--title`, `--style`, `--host`,
`--model`, `--planner-model`, `--turn-seconds`, `--turn-tokens`, `--max-cost`, `--max-sessions`,
`--resume RUN_ID`, `--run-id RUN_ID`, `--stems`, `--qc/--no-qc`, `--qc-retry`, `--dry-run`,
`--act-gap`, `--format`, `--bitrate`, `-o`.

`--model gpt-live-1` (experimental) records on the full-duplex Live API instead: billed
**$0.05 per session-minute per host** with no tokens (the dry run prints session-minutes rather
than token counts); the act runs in **real time** (a 1-minute act takes about a minute, acts still
parallel) because the Live session only advances while input audio streams; floor control is
advisory (the model is asked to wait for its cue; anything it says out of turn is discarded and
logged); turn ends are detected by loudness after ~1.2s of near-silent output; and `--turn-tokens`
has no effect — a turn is cut at 2× `--turn-seconds` of speech. See
[`docs/audio/simulated-podcasts.md`](audio/simulated-podcasts.md#gpt-live-1-experimental).

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

A script is bounded before the first TTS request, because every one of these numbers is an
allocation or a connection: at most 2000 segments of 40000 characters each; `pause_after`,
`default_pause_ms`, `intro_silence_ms` and `outro_silence_ms` each 0–60000 ms; and no more than
an hour of silence in total (summed as if every segment's pause were inserted — the final
segment's is not, so the bound is conservative). A JSON `null` on any of those means "not set",
the same as leaving the key out. Segments fan out at most 32-wide per provider by default —
`config.max_concurrency` raises or lowers that, `SANZARU_OPENAI_MAX_CONCURRENCY` and ElevenLabs'
tier caps still win when set. The `-o` name must be a bare filename with an audio extension
(`mp3`, `wav`, `m4a`, `flac`, `ogg`, …) and may not be a simulated run's bookkeeping
(`simrun_<id>.json`) or the audio of a recorded act (a `<name>.json` act sidecar beside it); all
three refusals are usage errors (exit 2) and cost nothing.

#### `--verify`

TTS drops the tail of a segment, and occasionally a whole short segment, **at random and with
no error**. `transcript` in the envelope is just an echo of the script you sent, so it is no
evidence the audio contains the words — which is why callers built an external QC discipline
around this tool, budgeting a median of three renders per episode.

`--verify` transcribes each rendered unit *before* stitching and checks it against the script:
a fuzzy match on the last 8 words (the tail is where drops happen), or presence anywhere in the
audio for segments of 4 words or fewer. Anything that fails is re-rendered **once** and
re-checked. The episode is written either way.

```console
$ sanzaru podcast generate @episode.json --verify -o ep.mp3
sanzaru: verified: all 28 segments present in the audio
sanzaru:   (1 re-rendered to get there)
```

The envelope carries `verified`, `verify_retries`, and a `segment_verdicts` list. Each verdict
has `ok` (no problem was *found*), `checked` (the audio was actually transcribed and compared),
a `reason` and a `similarity`. Reasons: `tail_missing`, `segment_missing`, `diverged` (found
wrong, `ok: false`), or `not_transcribed` / `too_large_to_verify` (never looked at — `ok: true`
but `checked: false`). `verified` is `null` when you did not ask for it, `true` only when every
segment was both checked and found, and `false` when any segment was missing **or** could not be
checked — so a `false` is read from the verdicts, not assumed to mean "missing". The episode is
written in every case. On stderr the two show up as separate blocks: `verified: N of M segments
NOT found after a retry` for the first, `NOT verified: N of M segments could not be checked` for
the second.

Transcription is retried three times with a short back-off before a unit is given up as
`not_transcribed`, since those failures are mostly transient 429/5xx and the check is cheap and
idempotent. The *render* is still retried exactly once — that is the expensive half. A unit over
the 25 MB transcription upload limit is reported `too_large_to_verify` without a call: split the
segment to make it checkable.

Costs one transcription per unit, so it is off by default. Two things it does not do: drops are
per-render random rather than per-segment sticky, so a segment failing **twice** wants its tail
rewritten to be grammatically part of a longer sentence rather than a third render — the tool
says so and stops. And a dialogue-mode run is one request or nothing, so a failure anywhere in a
batched run re-renders the whole run.

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
