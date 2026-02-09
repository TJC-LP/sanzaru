---
name: prompt-guidance
description: "Read the entirety of @docs/sora2-prompting-guide.md and await further instruction."
---

# Sanzaru MCP — Prompting & Workflow Guide

## Tool Quick Reference

| Category | Tool | Pattern | Description |
|----------|------|---------|-------------|
| **Video** | `create_video` | async | Create Sora video (returns job ID, poll for completion) |
| | `get_video_status` | poll | Check video generation progress (0-100%) |
| | `download_video` | sync | Download completed video/thumbnail/spritesheet |
| | `list_videos` | sync | List video jobs with pagination |
| | `list_local_videos` | sync | List downloaded video files |
| | `delete_video` | sync | Permanently delete a video from OpenAI |
| | `remix_video` | async | Create new video by remixing an existing one |
| **Image** | `generate_image` | **sync** | Images API — returns immediately, no polling (RECOMMENDED) |
| | `edit_image` | **sync** | Edit/compose images (up to 16 inputs) |
| | `create_image` | async | Responses API — for iterative refinement chains |
| | `get_image_status` | poll | Check image generation status |
| | `download_image` | sync | Download completed image |
| **Reference** | `list_reference_images` | sync | List available images for Sora |
| | `prepare_reference_image` | sync | Resize image to exact Sora dimensions |
| **Audio** | `create_audio` | sync | Text-to-speech (10 voices, any length) |
| | `transcribe_audio` | sync | Whisper transcription |
| | `chat_with_audio` | sync | GPT-4o audio analysis |
| | `list_audio_files` | sync | List and filter audio files |

## Model Selection

### Video (Sora)
- **`sora-2`** (default): Faster, cheaper, good for iteration
- **`sora-2-pro`**: Higher quality, supports larger resolutions (1024x1792, 1792x1024)

### Image Generation
| Tool | API | Best For |
|------|-----|----------|
| `generate_image` | Images API | New generation — **synchronous, no polling** (RECOMMENDED) |
| `edit_image` | Images API | Editing existing images, composition |
| `create_image` | Responses API | Iterative refinement with `previous_response_id` |

- **gpt-image-1.5**: STATE-OF-THE-ART (recommended default)
- **gpt-image-1-mini**: Fast, cost-effective for iteration

### Audio (TTS)
- **gpt-4o-mini-tts**: Recommended default
- Voices: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer

## The Golden Rule: Reference Images

> **CRITICAL**: When using `input_reference_filename` with Sora, describe **motion/action ONLY**. Do NOT re-describe what's already in the image.

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

**Duration tips**: 4s clips have best instruction following. Use 8s for simple scenes. 12s only for slow, ambient shots.

## Async Polling Pattern

```
# Video: create → poll → download
video = create_video(prompt="...", size="1280x720")
status = get_video_status(video.id)   # Poll until "completed"
download_video(video.id, filename="output.mp4")

# Image (Responses API): create → poll → download
resp = create_image(prompt="...")
status = get_image_status(resp.id)    # Poll until "completed"
download_image(resp.id, filename="output.png")

# Image (Images API): SYNCHRONOUS — no polling!
result = generate_image(prompt="...")  # Returns immediately
```

## Common Pitfalls

1. **Re-describing reference images** — Describe motion only (see Golden Rule above)
2. **Using `create_image` when `generate_image` is simpler** — Most cases don't need async polling
3. **Dimension mismatch** — Reference image MUST match target video size exactly. Use `prepare_reference_image` to resize.
4. **Vague motion** — "walks around" is weak. Use beats: "takes three steps, pauses, looks up"
5. **Integer seconds** — `seconds` must be a string: `"8"` not `8`
6. **Complex long clips** — Shorter (4s) clips follow instructions better than 12s
7. **Forgetting to poll** — `create_video` and `create_image` are async; always poll status before downloading

## Deep Reference

For detailed guidance:
- [Sora Prompting Guide](reference/SORA-PROMPTING.md) — Camera vocabulary, lighting, dialogue, remix strategy
- [Workflows](reference/WORKFLOWS.md) — Step-by-step patterns for common tasks
