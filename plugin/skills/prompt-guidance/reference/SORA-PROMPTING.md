# Sora Prompting Deep Reference

## Camera Direction Vocabulary

| Shot | Description | When to Use |
|------|-------------|-------------|
| Wide/establishing | Full environment visible | Opening shots, scene setting |
| Medium | Waist-up framing | Dialogue, character interaction |
| Close-up | Face/object fills frame | Emotion, detail, reaction |
| Tracking | Camera follows subject | Walking, driving, movement |
| Static/locked | No camera movement | Stability, focus on action |
| Crane/aerial | High angle, moving down | Reveals, scale, grandeur |
| Handheld | Slight natural shake | Documentary feel, urgency |
| Dutch angle | Tilted frame | Tension, disorientation |

**Tip**: One camera move per shot. "Slow dolly forward on medium close-up" works; "dolly forward while panning and tilting" doesn't.

## Motion Control

Break action into **beats** — discrete, sequential steps:

**Weak**: "The dancer performs gracefully across the stage"
**Strong**: "The dancer steps left, extends arms, pauses mid-turn, then completes the spin with arms overhead"

**Rules:**
- Each beat = one clear action
- 4s clip = 2-3 beats
- 8s clip = 4-6 beats
- Never combine conflicting motions ("walks forward while turning around")

## Lighting & Color

Don't say "well-lit" or "moody." Specify sources and colors:

**Template**: `[key light source], [fill light], [accent/rim], [color temperature]`

**Examples:**
- "Soft window light camera-left, warm practical lamp fill, cool blue rim from hallway"
- "High-contrast top light, deep shadows below chin, amber highlights on wet surfaces"
- "Overcast flat light, muted palette: slate, sage, cream"

**Color anchors** — Name 3-5 specific colors:
- Instead of "colorful": "teal, rust, gold, cream"
- Instead of "dark": "charcoal, deep navy, muted bronze"

## Dialogue

Place dialogue in a `<dialogue>` block below the prose description:

```
A coffee shop at golden hour. Two friends sit across from each other,
one leaning forward earnestly while the other stirs their drink slowly.

<dialogue>
Friend 1: "I think we should just go for it."
Friend 2: (pauses, looks down at coffee) "You really think so?"
</dialogue>
```

**Rules:**
- 4s clip = 1-2 short exchanges max
- 8s clip = 3-4 lines
- Keep lines concise and natural
- Label speakers consistently

## Remix Strategy

Use `remix_video` to **nudge, not gamble**. Make one change at a time:

**Good remix prompts:**
- "Same shot, switch from 35mm to 85mm lens"
- "Same lighting, new color palette: teal, sand, rust"
- "Keep framing, slow the camera movement by half"
- "Same scene, add light rain"

**Bad remix prompts:**
- "Completely different scene" (just create a new video)
- "Change everything but keep it similar" (too vague)

**When a shot keeps failing:**
1. Freeze the camera (static shot)
2. Simplify the action (one beat)
3. Clear the background (reduce visual complexity)
4. Layer complexity back in once the base works

## Style Keywords That Work

| Category | Effective Keywords |
|----------|--------------------|
| Film stock | "35mm film grain," "16mm," "IMAX 70mm," "Super 8" |
| Era | "1970s color grading," "90s home video," "2020s digital" |
| Genre | "Film noir," "sci-fi thriller," "nature documentary" |
| Lens | "Anamorphic 2.0x," "85mm portrait," "24mm wide angle" |
| Processing | "Cross-processed," "bleach bypass," "day-for-night" |

## Video Size Reference

| Size | Orientation | Models |
|------|-------------|--------|
| `1280x720` | Landscape 16:9 | sora-2, sora-2-pro |
| `720x1280` | Portrait 9:16 | sora-2, sora-2-pro |
| `1792x1024` | Wide landscape | sora-2-pro only |
| `1024x1792` | Tall portrait | sora-2-pro only |

**Important**: Reference images MUST match the target video size exactly. Use `prepare_reference_image` to resize.
