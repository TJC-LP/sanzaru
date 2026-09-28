---
name: sanzaru-cli
description: Generate videos (Higgsfield — Seedance 2.5, Kling 3.0; priced before submit), images (gpt-image-2.5), speech/transcription (OpenAI or ElevenLabs), scripted podcasts (ElevenLabs v4 recommended), and experimental simulated podcasts (realtime agents that actually converse) from the shell with the sanzaru CLI. Use for long-running media jobs (create → wait → download one-shots, resumable waits, JSON envelopes, batch fan-out) instead of loading the MCP tool surface.
---

# Sanzaru CLI for agents

`sanzaru <group> <verb>` wraps Higgsfield video, gpt-image-2.5, TTS/Whisper, and podcast APIs
for shell use. Requires `OPENAI_API_KEY` in the environment, and `HF_KEY` (a Higgsfield **API**
key, `key_id:key_secret` — prepaid, billed separately from a Higgsfield app subscription) for video. This skill is the **shell surface**;
the same server's MCP tools are covered by `sanzaru-mcp`, and how to *write* the prompts (video
prompt anatomy, the reference-image golden rule) by `prompt-guidance`. Start with:

```bash
sanzaru capabilities   # no API key needed: version, enabled features, command map
```

## Output contract (parse this, not the docs)

- **stdout**: exactly one JSON envelope per input — `{"v":1, "ok":true, "command":"...", "result":{...}}`.
  Fan-out commands stream one envelope per line (JSONL) in completion order.
- **stderr**: progress lines and hints (`sanzaru: hf_1c9e… in_progress t=95s`). Never parse it.
- Errors are envelopes too (`"ok":false`, `error.type`, often a `resume` command) — `jq` never hangs.
- Exit codes: `0` ok · `1` runtime/API · `2` usage · `3` config (missing key/extra) ·
  `4` **timeout — job still running, resumable** · `5` job failed server-side · `6` partial batch · `130` interrupted.
- Video error types worth branching on: `over_budget` (exit 2 — over `--max-cost`, nothing submitted or
  charged), `concurrency_limit` (exit 1 — 4 jobs already in flight; resubmitting is safe, or pass
  `--retry-busy 5m`), `insufficient_credits` (exit 1 — top up the API balance), and `api_error` with
  `maybe_submitted: true` (the submit may have gone through — check before resubmitting).

## The one-shot pattern (preferred)

`-o` implies `--download` implies `--wait`: one command submits, polls, downloads, and prints the
final path. Run it in the background if your shell caps foreground time.

```bash
sanzaru video create "the pilot looks up and smiles" --duration 8 --aspect-ratio 16:9 \
  --max-cost 5 -o ./out/pilot.mp4 --timeout 25m | jq -r .result.file.path
sanzaru image generate "an app icon, flat design" --quality high -o ./art/icon.png   # sync, ~10-60s
```

## The resume loop (harness-safe)

Submission returns in ~1s; waits are **idempotent**. On exit 4 the job keeps running server-side —
re-run the `resume` command from the envelope (or the same wait) until exit ≠ 4:

```bash
ID=$(sanzaru video create "..." --duration 8 --max-cost 5 | jq -r .result.id)
# ...do other work, then repeatedly:
sanzaru video wait "$ID" --download -o ./out/clip.mp4 --timeout 100s
# exit 0 → done · exit 4 → re-run · exit 5 → inspect .error
```

`sanzaru wait id1 id2 ...` polls mixed `hf_*`/`resp_*` ids concurrently, JSONL as each finishes.

## Video on Higgsfield — price first

Every submit is **estimated before it is sent** and the envelope carries `result.cost`
(`usd`, `basis`). Use `--dry-run` to see the price for free and `--max-cost USD` to refuse anything
over budget. Models (`--model`): `seedance-2.5` (default; 4-30 s, 480p/720p, ~$0.46/s @720p,
~$0.21/s @480p), `kling-3.0-std|pro|4k` (3-15 s, ~$0.35/$0.46/$1.16 per 5 s), `kling-3.0-turbo`,
or any catalog slug (`sanzaru video models --catalog`) with `--arg KEY=JSON` for its parameters.

```bash
sanzaru video create "waves at dusk, slow push-in" --duration 5 --resolution 480p --dry-run | jq .result.cost
ID=$(sanzaru video create "waves at dusk, slow push-in" --duration 5 --max-cost 3 -o ./out/w.mp4 | jq -r .result.id)
sanzaru video extend "$ID" "a gull crosses the frame" --duration 5 --max-cost 3 -o ./out/w2.mp4   # reuses the output
sanzaru video edit ./out/w.mp4 "make it snowing" --resolution 480p --max-cost 2 -o ./out/snow.mp4
```

`edit`/`extend` bill the source clip as well as the output. `video cancel ID` refunds a job still
queued. Higgsfield keeps outputs ~7 days — download (`-o` / `--download`) anything worth keeping.

## Choosing the right image command

- `image generate` — synchronous, RECOMMENDED for one-off images; returns file + token usage.
  Batch: `image generate "p1" "p2" --count 2 -o ./art/` (JSONL; exit 6 = partial, retry the
  failed `.input.prompt`s).
- `image create` — async job; use for refinement chains:
  `image create "add neon rain" --previous-id "$R1" -o v2.png`.
- gpt-image-2.5-flare is the default (sunburst for `edit`); `--background transparent` and `--quality xhigh|max` work on both. gpt-image-2 rejects them.

## Two TTS providers

`audio speak` and `podcast generate` take `--provider openai|elevenlabs` (default `openai`;
ElevenLabs needs `ELEVENLABS_API_KEY` and `uv pip install 'sanzaru[elevenlabs]'`). What differs,
and will bite if you assume otherwise:

- `--voice` is an opaque voice **id** from your library, not a name like `alloy`, and is required.
- `--instructions` is **ignored** — put inline audio tags (`[whispers]`, `[excited]`) in the text
  instead. The default `eleven_v4` performs stacked tags in order (`[whispers] [nervously]`).
- Speed is 0.7–1.2, and `eleven_v4`, `eleven_v4_turbo` and `eleven_v3` refuse any change (the API
  accepts it on those and silently ignores it). Out-of-range values raise rather than
  being rescaled from OpenAI's 0.25–4.0, so `--speed 2.0` never quietly means two things.
- `--voice-settings '{"stability":0.4,"similarity_boost":0.85}'` is ElevenLabs-only.

Podcast speakers choose independently (`speaker.provider` > `config.provider` > `--provider`), so
one episode can mix both. HTTP 429 means you exceeded your tier's concurrency — lower
`SANZARU_ELEVENLABS_MAX_CONCURRENCY` (defaults are Free-tier: 2, or 4 on flash/turbo; v4 Turbo is
in the v4 pool and gets 2).

ElevenLabs bills **characters submitted**, audio tags included, against a monthly allowance that
can be small. Do not count them by hand: every render reports `characters` (and `usage[]` per
provider on a podcast), and `sanzaru capabilities --quota` reads the remaining allowance without
spending any of it.

`podcast generate --render-mode dialogue` sends consecutive `eleven_v4` / `eleven_v4_turbo` /
`eleven_v3` turns as **one** request
so the model paces the exchange itself — distinctly more natural than fixed silence gaps. Turns
that cannot join a run (OpenAI speakers, other models, a lone turn, a stretch in one voice, a turn
that alone fills the 2000-character request budget) still render per segment, so mixed episodes
keep working. Inside a run, `pause_after` and per-speaker `voice_settings` do not
apply; use `config.dialogue_stability` (0–1) instead. Stay on the default `segments` when you need
exact gaps, per-speaker tuning, or cheap per-segment retry.

**Recommended engine for a scripted multi-voice show:** `--provider elevenlabs` (default model
`eleven_v4`) `--render-mode dialogue --verify`; `--model eleven_v4_turbo` for drafts (half the
character cost, roughly half the render time). In a blind listen v4 performed stacked tags and laughter that v3
dropped. OpenAI stays the default provider only because it needs no extra key. Premade ElevenLabs
voices that pair well (global ids, every account has them):

| Voice | id | Character |
|---|---|---|
| George | `JBFqnCBsd6RMkjVDRZzb` | warm British storyteller |
| Laura | `FGY2WhTYpPnrIDTdsKH5` | quirky enthusiast |
| Alice | `Xb7hH8MSUJpSbSDYk0k2` | clear British educator |
| Roger | `CwhRBWXzGAHq8TQ4Fs17` | laid-back, casual |
| Matilda | `XrExE9yKIg1WjnnlVkGX` | professional |

## Podcasts: scripted (recommended) vs simulated (experimental)

`podcast generate` is the way to make a podcast. With only a **topic**, write the script yourself
first — turns, inline direction tags (`[laughs]`, `[whispers]`), a clear open and close — then
render it. Reach for `simulate` only when the user explicitly wants an unscripted conversation.

| you have | use |
| --- | --- |
| a **topic** or a **script** (the default case) | `podcast generate --provider elevenlabs --render-mode dialogue --verify` (write the script first if you only have a topic) |
| a **script** needing exact gaps / per-segment retry | `podcast generate --render-mode segments --verify` |
| ElevenLabs not configured | `podcast generate --verify` (OpenAI, zero extra setup) |
| an explicit ask for an **unscripted** conversation | **experimental:** `podcast rundown` then `podcast simulate` (dry run first, `--max-cost`) |

**Use `podcast generate --verify` on anything you care about.** TTS drops segment tails, and
occasionally whole short segments, at random and with no error — the `transcript` in the result is
only an echo of your script, so it proves nothing. `--verify` transcribes each rendered unit,
checks it against the script, and re-renders what is missing once. A segment that fails *twice*
will not be fixed by a third render: rewrite its tail to be grammatically part of a longer
sentence. This replaces the hand-rolled QC loop that used to cost a median of three renders.

`simulate` is **experimental** and is not TTS: each host is a `gpt-realtime` session with a persona, and one host's audio
is played into the others' ears, so they react to delivery and disagree for real. The transcript
is an *output*, so results vary run to run. It is also **the most expensive thing sanzaru does** — roughly $0.20 for 7
minutes on `gpt-realtime-2.1-mini`, ~3x that on the full model.

```bash
# 1. Plan. One text call, and the JSON is yours to edit.
sanzaru podcast rundown "why TTS providers drop sentence tails" --acts 3 -m 6 \
  --host "Avery::You host and translate jargon." \
  --host "Rory:cedar:You chased the bug. Dry, specific." -o rundown.json

# 2. ALWAYS dry-run first: plans, projects turns/duration/tokens/cost, records nothing.
sanzaru podcast simulate @rundown.json --dry-run

# 3. Record with a ceiling. Acts run in parallel; ~30s of wall clock for 7 min of audio.
sanzaru podcast simulate @rundown.json --model gpt-realtime-2.1-mini \
  --max-cost 2.00 --stems -o ./out/ep1.mp3
```

**Name the run yourself.** The minted run id prints on **stderr only**, and you parse stdout — so
a crash between recording and reading strands audio you paid for. Pass `--run-id ep1` (or a
top-level `"run_id"` in the rundown JSON) and `--resume ep1` always works. Recording twice under
one id is refused rather than overwriting the first run; `--dry-run` against it is always fine.

**Recovery.** Every act is checkpointed as it lands. On an interrupt, a crash, or exit 6 (cost
ceiling — the envelope carries `spent_usd`, `completed_acts` and a `resume`), pick it back up with
`sanzaru podcast simulate --resume RUN_ID`; only the missing acts re-record.

**You are the producer.** A built-in producer handles floor control, walks the talking points, and
lands each act — but those are defaults, and you will usually direct better. Each act in the
rundown takes `direction` (how to play it), `turn_notes` (`{"0": "..."}` by turn index, replacing
the generated note) and `speaking_order` (host ids, cycled, instead of strict alternation).
`max_turns` is a *budget*, not a cap — an act extends up to 1.5x it to reach `target_seconds` — so
put the landing instruction on turn `max_turns - 1`, which takes over the close and follows it
wherever timing puts it. Setting any `turn_notes` also pins who opens the act.
`turn_notes` is the strongest lever: it is the difference between "move onto the next point" and
"object to what they just said". Edit them in the rundown — the tool blocks while acts record in
parallel, so there is no live steering.

**QC** runs by default (~$0.005/min): it transcribes the rendered audio and judges it against the
rundown, catching dropped audio, missed points, and the characteristic parallel-recording failure
of two acts covering the same ground. `result.qc.flagged_acts` names what to listen to;
`--qc-retry` re-records the ones a fresh take can fix.

A retry is **not** automatically better — QC verdicts disagree run-to-run — so the take it replaces
survives as `..._take1.mp3` beside it, listed in `result.preserved_takes`. Assemble the best cut per
act from those. An act flagged only for `tail_truncated` is *not* auto-retried: it was cut off by
the token cap, so raise `--turn-tokens` and resume instead of paying for the same defect twice.

## Media in, media out

- `-o` takes a file or directory (trailing `/`); parents are created; without it files land in the
  configured media dir, else the cwd (noted on stderr). The envelope always has the absolute path.
- Inputs (`--image`, `--end-image`, `--ref`, video `SOURCE`, `--input-image`, audio files) take real paths or bare
  media-dir filenames; video inputs also accept an `hf_…` id of a completed job.
- Long content: inline, `@file`, or `-` (stdin) — e.g. `sanzaru audio speak @ch1.txt -o ch1.mp3`,
  `sanzaru podcast generate - < episode.json -o ep.mp3` (a script needs only `speakers` and
  `segments`; `id` defaults to `name`, `speed` to 1.0, and `config`/`title` are optional —
  see `podcast generate -h`).
- Reference-image → video: keep the video prompt **motion-only** (the image already carries look):
  `sanzaru image prepare hero.png --aspect-ratio 16:9` (image-to-video framing follows the image) then
  `video create "she turns and smiles" --image ...`.

## Command map

`video` create/edit/extend/status/wait/download/cancel/models/files · `image`
generate/edit/create/status/wait/download/prepare/files · `audio`
transcribe(--enhance; auto-windows files over 8 min)/chat/speak(--provider)/convert/compress/files(--latest) ·
`podcast` rundown/simulate/generate(--provider, --render-mode) ·
`wait` (mixed ids) · `capabilities` · `serve` (MCP server; bare `sanzaru` does the same).
Every command supports `-h`; details in docs/cli.md.
