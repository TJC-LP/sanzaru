---
name: sanzaru-cli
description: Generate videos (Sora), images (gpt-image-2), speech/transcription, and podcasts from the shell with the sanzaru CLI. Use for long-running media jobs (create → wait → download one-shots, resumable waits, JSON envelopes, batch fan-out) instead of loading the MCP tool surface.
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

## Media in, media out

- `-o` takes a file or directory (trailing `/`); parents are created; without it files land in the
  configured media dir, else the cwd (noted on stderr). The envelope always has the absolute path.
- Inputs (`--input-ref`, `--input-image`, audio files) take real paths or bare media-dir filenames.
- Long content: inline, `@file`, or `-` (stdin) — e.g. `sanzaru audio speak @ch1.txt -o ch1.mp3`,
  `sanzaru podcast generate - < episode.json -o ep.mp3`.
- Reference-image → video: keep the video prompt **motion-only** (the image already carries look):
  `sanzaru image prepare hero.png --size 1280x720` then `video create "she turns and smiles" --input-ref ...`.

## Command map

`video` create/remix/status/wait/download/list/delete/files · `image`
generate/edit/create/status/wait/download/prepare/files · `audio`
transcribe(--enhance)/chat/speak/convert/compress/files(--latest) · `podcast` generate ·
`wait` (mixed ids) · `capabilities` · `serve` (MCP server; bare `sanzaru` does the same).
Every command supports `-h`; details in docs/cli.md.
