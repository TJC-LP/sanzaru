---
name: sanzaru-mcp
description: Use sanzaru's MCP tools (create_video, edit_video, extend_video, create_image, generate_image, wait_for, create_audio, generate_podcast, simulate_podcast, view_media …) correctly — which tool for which job, how to wait for async ids in one call instead of polling, model choices, and the mistakes that waste renders. Load when the sanzaru MCP server is connected and you are about to call its tools. For shell/agent-CLI use load sanzaru-cli instead; for how to *write* video and image prompts load prompt-guidance.
---

# Sanzaru MCP tools

How to drive the sanzaru MCP server. This skill is about the **tool surface**: which tool,
which arguments, how to wait. For prompt craft (what to say to a video or image model) load
`prompt-guidance`; for the shell CLI load `sanzaru-cli`.

## Tool Quick Reference

| Category | Tool | Pattern | Description |
|----------|------|---------|-------------|
| **Jobs** | `wait_for` | **blocking** | Wait for any mix of `hf_*`/`resp_*` ids in ONE call; `download=true` saves them too. Use this instead of polling `get_*_status` |
| **Video** | `create_video` | async | Text- or image-to-video on Higgsfield (default Seedance 2.5; returns an `hf_*` id and the estimated `cost` — then `wait_for` it) |
| | `edit_video` | async | Re-render an existing clip (Seedance 2.5); the source is billed too |
| | `extend_video` | async | Continue an existing clip by N seconds; the source is billed too |
| | `get_video_status` | one-off | Single status check; prefer `wait_for` for waiting |
| | `download_video` | sync | Save a completed job's output (usually done by `wait_for(download=true)`) |
| | `cancel_video` | sync | Cancel a job that is still queued (refunded) |
| | `list_local_videos` | sync | List downloaded video files |
| **Image** | `generate_image` | **sync** | Images API — returns the finished image (RECOMMENDED for one-shots) |
| | `edit_image` | **sync** | Edit/compose images (up to 16 inputs, optional mask) |
| | `create_image` | async | Responses API — refinement chains (`previous_response_id`) and parallel batches; `model` picks the mainline model (`gpt-6-astra` default, `gpt-5.6-sol`/`terra`/`luna`) |
| | `get_image_status` | one-off | Single status check; prefer `wait_for` for waiting |
| | `download_image` | sync | Download completed image |
| **Inspect** | `inspect_image` | sync | **See an image yourself** — returns it as visual content; `region` zooms in to read small text |
| | `inspect_video_frame` | sync | See frames from a video — the only way to check what a render actually shows (needs ffmpeg) |
| **Reference** | `list_reference_images` | sync | List start frames / edit inputs in the reference directory |
| | `prepare_reference_image` | sync | Crop/pad an image to a video aspect ratio (`aspect_ratio=`) or exact `size` |
| **Audio** | `create_audio` | sync | Text-to-speech — OpenAI (named voices) or ElevenLabs (voice id) |
| | `transcribe_audio` | sync | Transcription (long files are windowed automatically) |
| | `chat_with_audio` | sync | Audio understanding / Q&A over a file |
| | `list_audio_files` | sync | List and filter audio files (substring or glob `pattern`, not regex) |
| **Podcast** | `generate_podcast` | sync | **The way to make a podcast** — from a script, or from a topic (write the script yourself first). Recommended: ElevenLabs `eleven_v4`, `render_mode: "dialogue"`, `verify: true` |
| | `simulate_podcast` | sync | **Experimental.** Unscripted — realtime or gpt-live-1 agents converse from a rundown. Most expensive tool, variable results; only when an unscripted conversation is explicitly wanted. `dry_run: true` first, set `max_cost_usd` |
| **Viewer** | `view_media` | sync | Opens a video/audio/image inline (MCP App), with a Download button where the host supports it |

## Which image tool

| Need | Tool |
|------|------|
| One image from a prompt | `generate_image` — synchronous, nothing to wait for |
| Edit or compose existing images | `edit_image` — synchronous; defaults to `gpt-image-2.5-sunburst` (editing precision) |
| Several images in parallel, or a refinement chain | `create_image` → `wait_for` |

`generate_image`/`create_image` default to `gpt-image-2.5-flare`. Both 2.5 variants accept
`background="transparent"` (png/webp) and `quality` up to `"xhigh"`/`"max"`; gpt-image-2 refuses
those before any request. `input_fidelity` is honoured only by gpt-image-1/1.5.

## Waiting on jobs — one call, not a loop

`create_video`, `edit_video`, `extend_video` and `create_image` return an id immediately. Do **not** call
`get_video_status` / `get_image_status` in a loop. Hand the ids to `wait_for`, which blocks
server-side, reports progress to the client on every poll, and returns every job's final
state in one round trip:

```
# Video: create → wait_for (downloads too) → done
video = create_video(prompt="...", duration=5, max_cost_usd=3)
wait_for([video.id], download=True)          # returns when finished; file is on disk

# Several jobs at once, mixed types
a = create_video(prompt="...")
b = create_image(prompt="...")
result = wait_for([a.id, b.id], download=True)
# result.jobs[i]: status, done, timed_out, error, download (filename)

# Image (Images API): SYNCHRONOUS — nothing to wait for
result = generate_image(prompt="...")  # Returns the finished image
```

`wait_for` **returns on its deadline instead of failing**: a job still running comes back with
`timed_out=true` and its last status. Call `wait_for` again with the same ids to keep waiting
(default deadline 240 s, max 1800 s). A bad id fails only its own entry, never the batch.
Use `get_*_status` only for a one-off check when you are not going to wait.

## Video arguments and cost (Higgsfield)

- `model`: `"seedance-2.5"` (default; 4-30 s, 480p/720p, 6 aspect ratios, edit/extend),
  `"kling-3.0-std"` / `"-pro"` / `"-4k"` (3-15 s, end frame), `"kling-3.0-turbo"`, or any catalog id
  (`"vendor/model/operation"`) with model-specific arguments in `extra`.
- `duration` is an **integer** number of seconds within the model's range.
- **Every job is priced before submit** and returns `cost`. Pass `max_cost_usd` for anything
  beyond a few dollars (Seedance 720p is ~$0.46/s — 10 s ≈ $4.60), and `dry_run=true` to see the
  price without submitting. A refused job uploads and charges nothing.
- With `reference_image`, the prompt describes **motion only** (see `prompt-guidance`), and
  `aspect_ratio` is refused — framing follows the image, so crop it with
  `prepare_reference_image(aspect_ratio="9:16")` first.
- An `hf_*` id works as `reference_image` / `source_video`: chain an extend onto a finished render
  without downloading anything.
- `edit_video` / `extend_video` bill the source clip as well as the output.
- Four jobs may be in flight per Higgsfield account (the Higgsfield app counts too); past that,
  submits are refused with nothing charged — `wait_for` a running job, then resubmit.
- The Higgsfield **API** is prepaid and billed separately from a Higgsfield app subscription.

## Podcasts

- **`generate_podcast` is the podcast tool.** Given only a topic, write the script yourself (turns,
  inline direction tags, a clear open and close), then render it — don't reach for
  `simulate_podcast`. Use `verify: true` when the words matter; it re-renders a segment whose tail
  went missing, once.
- **Recommended engine:** every speaker `"provider": "elevenlabs"`
  (default model `eleven_v4`), `config.render_mode: "dialogue"`, `verify: true`. Set
  `"model": "eleven_v4_turbo"` for drafts (half the character cost, roughly half the render time). Direct delivery
  with inline tags in the text (`[laughs]`, `[whispers] [nervously]` — v4 performs stacked tags in
  order); `instructions` and `speed` do nothing on v4 (speed is refused). Use `segments` mode for
  exact gaps or per-speaker `voice_settings`. OpenAI stays the default provider only because it
  needs no extra key — it is the fallback when ElevenLabs is not configured. Premade ElevenLabs voices that pair well (global ids):

  | Voice | id | Character |
  |---|---|---|
  | George | `JBFqnCBsd6RMkjVDRZzb` | warm British storyteller |
  | Laura | `FGY2WhTYpPnrIDTdsKH5` | quirky enthusiast |
  | Alice | `Xb7hH8MSUJpSbSDYk0k2` | clear British educator |
  | Roger | `CwhRBWXzGAHq8TQ4Fs17` | laid-back, casual |
  | Matilda | `XrExE9yKIg1WjnnlVkGX` | professional |
- `simulate_podcast` is **experimental**: realtime agents improvise an unscripted conversation.
  Use it only when the user explicitly asks for an unscripted/simulated conversation (or names
  the tool). Always `dry_run: true` first (plans, projects
  cost, spends nothing), then set `max_cost_usd`. gpt-realtime bills tokens; `gpt-live-1` bills
  $0.05 per host-minute and runs in **duplex** mode by default (hosts hear each other live). Every
  act is checkpointed; the result carries a `resume_command` if the run stops early.

## Checking your own work

`generate_image` and friends hand back a filename, not a picture. To judge what you
made, look at it:

```
generate_image(prompt="a poster reading 'GRAND OPENING'")
inspect_image("poster.png")                               # is the text right?
inspect_image("poster.png", region=[400, 200, 1100, 400]) # zoom in to be sure

wait_for([video.id], download=True)
inspect_video_frame("clip.mp4")                           # did the motion happen?
```

Worth doing before you tell the user something worked, and before spending a video
render on a reference image you have not looked at. `inspect_image` costs roughly
1500-3100 visual tokens at its default size, and a `region` crop costs far less, so a
zoomed check is cheaper than a full-frame one. Trust the note that comes back over the
pixels when reporting dimensions: you are usually seeing a resized copy, and the note
says so.

Do **not** use `view_media` for this. That opens a player for the user and returns you
nothing to look at.

## Showing the user a file

Call `view_media` after generating anything. It renders a player inline and, on hosts
that support the MCP Apps `ui/download-file` request, shows a Download button that saves
the file directly from the viewer.

That button is the download path. Do **not** try to move the bytes yourself — no fetching
the file into a code sandbox, no asking for a token or a URL to curl. The viewer already
has the bytes and hands them to the host. If the button is absent the host does not
support saving, which is not something a tool call can work around.

## Resources are for the user, not for you

The server exposes `sanzaru://image/{filename}`, `sanzaru://video/{filename}` and
`sanzaru://audio/{filename}` so a **person** can attach their own renders from their
client's attachment menu. They deliver the whole file and are annotated for a user
audience.

Do not read media through them to look at it yourself — that ships a full-size file
where `inspect_image` / `inspect_video_frame` send a right-sized picture. Mention the
URI when someone asks how to reuse a file in a future conversation.

## Common Pitfalls

1. **Polling by hand** — `create_video` and `create_image` are async; call `wait_for(ids, download=True)` once instead of looping over `get_*_status`, and don't `download_*` before the job is done
2. **Using `create_image` when `generate_image` is simpler** — a single image needs no async job
3. **Passing `aspect_ratio` with a reference image** — framing follows the image; crop it with `prepare_reference_image(aspect_ratio=...)`
4. **Submitting an expensive render without a cap** — set `max_cost_usd` (or `dry_run=true` first); Seedance bills per second and edit/extend bill the source too
5. **Transparent output on gpt-image-2** — it raises; the 2.5 defaults support it
6. **Regex in `list_audio_files`** — `pattern` is substring or glob; regex syntax is refused with an error naming the syntax
7. **Recording a simulated podcast without a dry run** — it is the most expensive thing sanzaru does
8. **Fetching media bytes to deliver a file** — `view_media` shows a player and a Download button; the bytes never need to pass through you
9. **Reporting that an image or video is correct without looking** — `inspect_image` / `inspect_video_frame` exist; a "completed" status says nothing about what was rendered
10. **Reporting an image's dimensions from the pixels you were shown** — inspection returns a resized copy and states the source size in its note; quote the note

## Deep Reference

- [Workflows](reference/WORKFLOWS.md) — step-by-step tool sequences for common tasks
- `prompt-guidance` skill — how to write the prompts (video prompt anatomy, the reference-image golden rule, camera and lighting vocabulary)
