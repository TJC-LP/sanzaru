---
name: sanzaru-cli
description: Generate videos (Sora), images (gpt-image-2), speech/transcription (OpenAI or ElevenLabs), scripted podcasts, and simulated podcasts (realtime agents that actually converse) from the shell with the sanzaru CLI. Use for long-running media jobs (create → wait → download one-shots, resumable waits, JSON envelopes, batch fan-out) instead of loading the MCP tool surface.
---

# Sanzaru CLI for agents

`sanzaru <group> <verb>` wraps OpenAI's Sora video, gpt-image-2, TTS/Whisper, and podcast APIs
for shell use. Requires `OPENAI_API_KEY` in the environment. Start with:

```bash
sanzaru capabilities   # no API key needed: version, enabled features, command map
```

## Output contract (parse this, not the docs)

- **stdout**: exactly one JSON envelope per input — `{"v":1, "ok":true, "command":"...", "result":{...}}`.
  Fan-out commands stream one envelope per line (JSONL) in completion order.
- **stderr**: progress lines and hints (`sanzaru: video_x in_progress 42% t=95s`). Never parse it.
- Errors are envelopes too (`"ok":false`, `error.type`, often a `resume` command) — `jq` never hangs.
- Exit codes: `0` ok · `1` runtime/API · `2` usage · `3` config (missing key/extra) ·
  `4` **timeout — job still running, resumable** · `5` job failed server-side · `6` partial batch · `130` interrupted.

## The one-shot pattern (preferred)

`-o` implies `--download` implies `--wait`: one command submits, polls, downloads, and prints the
final path. Run it in the background if your shell caps foreground time.

```bash
sanzaru video create "the pilot looks up and smiles" --seconds 8 --size 1280x720 \
  -o ./out/pilot.mp4 --timeout 25m | jq -r .result.file.path
sanzaru image generate "an app icon, flat design" --quality high -o ./art/icon.png   # sync, ~10-60s
```

## The resume loop (harness-safe)

Submission returns in ~1s; waits are **idempotent**. On exit 4 the job keeps running server-side —
re-run the `resume` command from the envelope (or the same wait) until exit ≠ 4:

```bash
ID=$(sanzaru video create "..." --seconds 8 | jq -r .result.id)
# ...do other work, then repeatedly:
sanzaru video wait "$ID" --download -o ./out/clip.mp4 --timeout 100s
# exit 0 → done · exit 4 → re-run · exit 5 → inspect .error
```

`sanzaru wait id1 id2 ...` polls mixed `video_*`/`resp_*` ids concurrently, JSONL as each finishes.

## Choosing the right image command

- `image generate` — synchronous, RECOMMENDED for one-off images; returns file + token usage.
  Batch: `image generate "p1" "p2" --count 2 -o ./art/` (JSONL; exit 6 = partial, retry the
  failed `.input.prompt`s).
- `image create` — async job; use for refinement chains:
  `image create "add neon rain" --previous-id "$R1" -o v2.png`.
- gpt-image-2 is the default; `--background transparent` requires `--image-model gpt-image-1.5`.

## Two TTS providers

`audio speak` and `podcast generate` take `--provider openai|elevenlabs` (default `openai`;
ElevenLabs needs `ELEVENLABS_API_KEY` and `uv pip install 'sanzaru[elevenlabs]'`). What differs,
and will bite if you assume otherwise:

- `--voice` is an opaque voice **id** from your library, not a name like `alloy`, and is required.
- `--instructions` is **ignored** — put inline audio tags (`[whispers]`, `[excited]`) in the text
  instead, which the default `eleven_v3` understands.
- Speed is 0.7–1.2, and `eleven_v3` rejects any change. Out-of-range values raise rather than
  being rescaled from OpenAI's 0.25–4.0, so `--speed 2.0` never quietly means two things.
- `--voice-settings '{"stability":0.4,"similarity_boost":0.85}'` is ElevenLabs-only.

Podcast speakers choose independently (`speaker.provider` > `config.provider` > `--provider`), so
one episode can mix both. HTTP 429 means you exceeded your tier's concurrency — lower
`SANZARU_ELEVENLABS_MAX_CONCURRENCY` (defaults are Free-tier: 2, or 4 on flash/turbo).

`podcast generate --render-mode dialogue` sends consecutive `eleven_v3` turns as **one** request
so the model paces the exchange itself — distinctly more natural than fixed silence gaps. Turns
that cannot join a run (OpenAI speakers, other models, a lone turn, a stretch in one voice, a turn
that alone fills the 2000-character request budget) still render per segment, so mixed episodes
keep working. Inside a run, `pause_after` and per-speaker `voice_settings` do not
apply; use `config.dialogue_stability` (0–1) instead. Stay on the default `segments` when you need
exact gaps, per-speaker tuning, or cheap per-segment retry.

## Podcasts: scripted vs simulated

Three verbs, and picking the wrong one wastes either quality or money.

| you have | use |
| --- | --- |
| a **topic** and want a real conversation | `podcast rundown` then `podcast simulate` |
| a **script** you want performed naturally | `podcast generate --render-mode dialogue` |
| a **script** needing exact gaps / per-segment retry | `podcast generate` (default) |

`simulate` is not TTS: each host is a `gpt-realtime` session with a persona, and one host's audio
is played into the others' ears, so they react to delivery and disagree for real. The transcript
is an *output*. It is also **the most expensive thing sanzaru does** — roughly $0.20 for 7
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
- Inputs (`--input-ref`, `--input-image`, audio files) take real paths or bare media-dir filenames.
- Long content: inline, `@file`, or `-` (stdin) — e.g. `sanzaru audio speak @ch1.txt -o ch1.mp3`,
  `sanzaru podcast generate - < episode.json -o ep.mp3` (script `config` requires
  `default_pause_ms`, `normalize_loudness`, and `output_format` — see `podcast generate -h`).
- Reference-image → video: keep the video prompt **motion-only** (the image already carries look):
  `sanzaru image prepare hero.png --size 1280x720` then `video create "she turns and smiles" --input-ref ...`.

## Command map

`video` create/remix/status/wait/download/list/delete/files · `image`
generate/edit/create/status/wait/download/prepare/files · `audio`
transcribe(--enhance)/chat/speak(--provider)/convert/compress/files(--latest) ·
`podcast` rundown/simulate/generate(--provider, --render-mode) ·
`wait` (mixed ids) · `capabilities` · `serve` (MCP server; bare `sanzaru` does the same).
Every command supports `-h`; details in docs/cli.md.
