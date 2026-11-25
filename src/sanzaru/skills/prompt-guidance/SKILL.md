---
name: prompt-guidance
description: How to write prompts for video (Higgsfield — Seedance 2.5, Kling 3.0) and gpt-image models — prompt anatomy (style, scene, camera, action beats, lighting), the reference-image golden rule (describe motion only), duration and model selection, camera/lighting vocabulary. Surface-neutral — applies whether you call sanzaru's MCP tools or the CLI. Load before writing or revising a video or image prompt.
---

# Prompting video and image models

This skill is about **what to say**. How to call the tools lives in `sanzaru-mcp` (MCP) and
`sanzaru-cli` (shell). Examples below use the MCP tool names for brevity; the CLI takes the same
prompts.

## The Golden Rule: Reference Images

> **CRITICAL**: When animating from a reference image (`reference_image`), describe
> **motion/action ONLY**. Do NOT re-describe what's already in the image.

The reference image already contains: character, setting, framing, style, lighting.
Your prompt should only describe: what happens next, motion, camera movement.

**BAD** — re-describing the image:
```
create_video(
    prompt="A pilot in orange suit in cockpit with glowing instruments...",
    reference_image="pilot.png"
)
```

**GOOD** — motion only:
```
create_video(
    prompt="The pilot glances up, takes a breath, then returns focus to the instruments.",
    reference_image="pilot.png"
)
```

## Video Prompt Anatomy

Write prompts in this order for best results:

1. **Style** — "1970s film grain," "IMAX scale," "16mm black-and-white"
2. **Scene** — Characters, setting, framing
3. **Camera** — "wide establishing shot, eye level" or "medium close-up, tracking left"
4. **Action in beats** — Small, grounded steps: "takes four steps to window, pauses, pulls curtain"
5. **Lighting & color** — 3-5 concrete anchors: "warm lamp fill, cool rim from hallway, amber highlights"

| Weak | Strong |
|------|--------|
| "A beautiful street at night" | "Wet asphalt, neon signs reflecting in puddles, steam from grate" |
| "Person moves quickly" | "Cyclist pedals three times, brakes, stops at crosswalk" |
| "Cinematic look" | "Anamorphic 2.0x lens, shallow DOF, volumetric light" |

**Duration tips**: 4-5 s clips follow instructions best and cost least — draft there at 480p.
Longer clips (Seedance runs to 30 s) suit slow, ambient shots. With a reference image, framing
follows the image: crop it first with `prepare_reference_image(aspect_ratio=...)`.

## Model Selection

### Video (Higgsfield)
- **`seedance-2.5`** (default): best quality, 4-30 s, native audio, edit/extend.
  ~$0.46/s at 720p, ~$0.21/s at 480p
- **`kling-3.0-std` / `-pro` / `-4k`**: 3-15 s, start + end frame; ~$0.35 / $0.46 / $1.16 per 5 s
- **`kling-3.0-turbo`**: fastest, ~$0.31 per 5 s
- Every job is priced before submit — use `dry_run` to see the cost and `max_cost_usd` to cap it

### Image generation
- **gpt-image-2.5-flare** (generation default) / **gpt-image-2.5-sunburst** (editing default):
  state of the art, up to 4K, transparent backgrounds, `quality` up to `xhigh`/`max`
- **gpt-image-2**: previous flagship (~99% text accuracy); no transparent output
- **gpt-image-1.5**: older, fixed sizes; the only current model that honours `input_fidelity`
- **gpt-image-1-mini**: fast, cost-effective drafts

Image prompts: name the subject, medium and composition; state text to render in quotes; give
one lighting anchor. For edits, describe the *change*, not what is already in the picture.

### Audio (TTS)
- **gpt-4o-mini-tts**: recommended default; `instructions` steer delivery ("warm, unhurried")
- OpenAI voices: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
- ElevenLabs (`eleven_v4`, default): inline audio tags like `[whispers]` instead of `instructions`;
  v4 performs stacked tags in order (`[whispers] [nervously] Like this.`)

## Common Prompting Pitfalls

1. **Re-describing reference images** — describe motion only (Golden Rule above)
2. **Vague motion** — "walks around" is weak; use beats: "takes three steps, pauses, looks up"
3. **Complex long clips** — 4-5 s clips follow instructions better than long ones
4. **Abstract adjectives** — "cinematic", "beautiful" do nothing; name the lens, light and surface
5. **Describing the edit target twice** — on `edit_image`, say what changes, not what is there

## Deep Reference

- [Video Prompting Reference](reference/VIDEO-PROMPTING.md) — camera vocabulary, motion control, lighting, sound, edit/extend strategy, frame shapes
- Full guide in the repo: `docs/video-prompting-guide.md`
