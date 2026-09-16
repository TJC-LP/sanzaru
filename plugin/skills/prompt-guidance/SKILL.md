---
name: prompt-guidance
description: How to write prompts for Sora video and gpt-image models — prompt anatomy (style, scene, camera, action beats, lighting), the reference-image golden rule (describe motion only), duration and model selection, camera/lighting vocabulary. Surface-neutral — applies whether you call sanzaru's MCP tools or the CLI. Load before writing or revising a video or image prompt.
---

# Prompting Sora and image models

This skill is about **what to say**. How to call the tools lives in `sanzaru-mcp` (MCP) and
`sanzaru-cli` (shell). Examples below use the MCP tool names for brevity; the CLI takes the same
prompts.

## The Golden Rule: Reference Images

> **CRITICAL**: When animating from a reference image (`input_reference_filename`), describe
> **motion/action ONLY**. Do NOT re-describe what's already in the image.

The reference image already contains: character, setting, framing, style, lighting.
Your prompt should only describe: what happens next, motion, camera movement.

**BAD** — re-describing the image:
```
create_video(
    prompt="A pilot in orange suit in cockpit with glowing instruments...",
    input_reference_filename="pilot.png"
)
```

**GOOD** — motion only:
```
create_video(
    prompt="The pilot glances up, takes a breath, then returns focus to the instruments.",
    input_reference_filename="pilot.png"
)
```

## Sora Prompt Anatomy

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

**Duration tips**: 4s clips have the best instruction following. Use 8s for simple scenes. 12s only
for slow, ambient shots. Shorter clips follow instructions better than long ones.

## Model Selection

### Video (Sora)
- **`sora-2`** (default): Faster, cheaper, good for iteration
- **`sora-2-pro`**: Higher quality, supports the larger 1792x1024 / 1024x1792 frames

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
- ElevenLabs (`eleven_v3`): inline audio tags like `[whispers]` instead of `instructions`

## Common Prompting Pitfalls

1. **Re-describing reference images** — describe motion only (Golden Rule above)
2. **Vague motion** — "walks around" is weak; use beats: "takes three steps, pauses, looks up"
3. **Complex long clips** — 4s clips follow instructions better than 12s
4. **Abstract adjectives** — "cinematic", "beautiful" do nothing; name the lens, light and surface
5. **Describing the edit target twice** — on `edit_image`, say what changes, not what is there

## Deep Reference

- [Sora Prompting Guide](reference/SORA-PROMPTING.md) — camera vocabulary, motion control, lighting, dialogue, remix strategy, size reference
- Full guide in the repo: `docs/sora2-prompting-guide.md`
