# Video Prompting Deep Reference (Higgsfield: Seedance 2.5, Kling 3.0)

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
- 4-5 s clip = 2-3 beats
- 8-10 s clip = 4-6 beats (Seedance runs to 30 s; Kling to 15 s)
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

## Sound and dialogue

Seedance 2.5 and Kling 3.0 generate native audio by default (`audio=false` for a silent clip). Name
what should be heard — "rain on a tin roof", "crowd murmur, no music" — and put spoken lines in
quotes with the speaker:

```
A coffee shop at golden hour. Two friends sit across from each other.
The first leans in: "I think we should just go for it." The second pauses, stirring: "You really think so?"
```

- 4-5 s clip = 1-2 short lines; keep them concise and natural
- Label speakers consistently

## Edit and extend strategy (Seedance 2.5)

Use `edit_video` to **nudge, not gamble**. Make one change at a time and say what to keep:

**Good edit prompts:**
- "Same shot, make it snowing"
- "Keep the camera move and the actor; change the palette to teal, sand, rust"
- "Render it as a watercolor, same framing"

**Bad edit prompts:**
- "Completely different scene" (just create a new video)
- "Change everything but keep it similar" (too vague)

`extend_video` continues the shot: describe what happens **next**, in the same style. Both bill the
source as well as the output — draft at 480p.

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

## Frame shape reference

| Setting | Values | Models |
|---------|--------|--------|
| `aspect_ratio` (text-to-video) | 16:9, 4:3, 1:1, 3:4, 9:16, 21:9 | seedance-2.5 |
| `aspect_ratio` (text-to-video) | 16:9, 9:16, 1:1 | kling-3.0-* |
| `resolution` | 480p, 720p | seedance-2.5 (Kling's tier sets it: std/pro/4k) |

**Image-to-video has no aspect ratio** — framing follows the start image. Crop it first with
`prepare_reference_image(input, aspect_ratio="9:16")` (or `size="WxH"`).
