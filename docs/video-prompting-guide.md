# Video Prompting Guide (Higgsfield: Seedance 2.5, Kling 3.0)

sanzaru generates video through the Higgsfield API (0.13.0+). OpenAI removed the Sora Videos API
on 2026-09-24; the craft below carries over, but the controls, limits and costs are Higgsfield's.

## Brief a cinematographer, not a search engine

A good prompt names **one shot**: who or what is in frame, what happens, how the camera moves,
and the light. Detail buys control; leaving things open buys variation. Both are valid — and the
same prompt renders differently each time, so plan to iterate on a cheap draft before paying for
the final.

```
A lighthouse keeper climbs the spiral stairs at dusk, lantern swinging.
Slow handheld follow from below. Warm lamplight against blue windows; wind audible outside.
```

Useful levers, in rough order of impact:

- **Action, in beats.** "She stops, turns, and smiles" lands better than "she is happy". One or
  two clear beats per 4–5 seconds; longer clips can carry more.
- **Camera.** Name the move and the framing: *static wide*, *slow push-in*, *orbit left*,
  *handheld follow*, *overhead*, *rack focus to the door*.
- **Light and time of day.** *Golden hour backlight*, *overcast soft light*, *neon rim light*.
- **Style.** A film stock, a genre, or a reference period — sparingly; one strong cue beats five.
- **Sound** (Seedance and Kling generate native audio by default). Mention what should be heard:
  *rain on a tin roof*, *crowd murmur*, *no music*. Pass `audio=false` for a silent clip.

## Image-to-video: prompt only the motion

With `reference_image`, the start frame already fixes the subject, setting, framing, style and
lighting. **Describe only what changes.**

❌ `"A pilot in an orange suit sitting in a cockpit with glowing instruments…"` — re-describes the image
✅ `"The pilot glances up, takes a breath, then returns focus to the instruments. Slow push-in."`

- **Framing follows the image** — there is no `aspect_ratio` for image-to-video. Crop the reference
  first: `prepare_reference_image("pilot.png", aspect_ratio="9:16")`.
- **End frames** (`end_image`) pin where the shot lands; keep start and end the same shape and the
  change between them plausible for the duration.
- **Chaining:** a finished job's `hf_…` id works as the next job's `reference_image` or
  `source_video` — no download and re-upload.

## Edit and extend (Seedance 2.5)

- **`edit_video`** keeps the source's motion, length and framing and changes what you name:
  *"make it snowing"*, *"turn the car red"*, *"render it as a watercolor"*. Say what to keep if it
  matters: *"keep the actor's face and the camera move"*.
- **`extend_video`** continues the shot: describe what happens **next**, in the same style —
  *"the door swings open and she steps into the rain"*. 4–30 s per extension.
- Both **bill the source as well as the output** (at 0.6× the text-to-video rate), so a 5 s edit
  costs about as much as 10 s of new footage at that rate.

## Models

| id | Use it for |
|---|---|
| `seedance-2.5` (default) | Best quality; long clips (4–30 s); 6 aspect ratios; the only curated model with edit/extend |
| `kling-3.0-std` / `-pro` / `-4k` | 3–15 s at a lower price; start **and** end frame; multi-shot via `extra={"multi_shots": true, "multi_prompt": [...]}` |
| `kling-3.0-turbo` | Fastest, cheapest Kling |
| any catalog slug | e.g. `minimax/hailuo-2.3/standard/text-to-video`; pass model-specific parameters in `extra` |

`sanzaru video models --catalog` lists every video model the API offers.

## Cost — draft cheap, then commit

Prices measured 2026-09-28, before any account discount:

| | 480p | 720p |
|---|---|---|
| Seedance 2.5, per second of output | ~$0.21 | ~$0.46 |
| Seedance 2.5 edit/extend, per second of source + output | ~$0.12 | ~$0.28 |
| Kling 3.0 std / pro / 4k, per 5 s | ~$0.35 / $0.46 / $1.16 | |

- **Every job is priced before it is submitted** and the cost comes back with it. Use
  `dry_run=true` to see the price for free, and `max_cost_usd` to refuse anything over budget —
  a refused job uploads nothing and charges nothing.
- Draft at **480p and 4–5 s**, pick the take you like, then render the final at 720p.
- The Higgsfield **API** is prepaid and billed separately from a Higgsfield app subscription; it
  has no balance endpoint, so the per-job estimate is the guardrail.
- Content moderation (`nsfw`) and failed jobs are not charged; a job canceled while still queued
  is refunded.

## Checking the result

A job reporting `completed` says nothing about whether the motion happened. After
`wait_for([id], download=true)`, look: `inspect_video_frame("hf_….mp4", frames=3)` samples frames
across the clip; `timestamps=[0.0]` checks the opening frame. Present it with `view_media`.
