# Sanzaru Workflow Patterns

## Workflow 1: Generate Image → Animate with Sora

The most common pattern — create a reference image and bring it to life.

```python
# Step 1: Generate reference image (synchronous, no polling!)
generate_image(
    prompt="A lone astronaut standing on a red desert planet, cinematic lighting",
    size="1536x1024",
    quality="high",
    filename="astronaut.png"
)

# Step 2: Resize to match Sora dimensions
prepare_reference_image("astronaut.png", "1280x720", resize_mode="crop")
# Creates: astronaut_1280x720.png

# Step 3: Create video with MOTION-ONLY prompt
video = create_video(
    prompt="The astronaut turns and walks toward the horizon",
    size="1280x720",
    input_reference_filename="astronaut_1280x720.png",
    seconds="8"
)

# Step 4: Poll until complete
status = get_video_status(video.id)
# Repeat until status == "completed"

# Step 5: Download
download_video(video.id, filename="astronaut_walk.mp4")
```

**Resize modes:**
- `crop` (default): Scale to cover, center crop — no distortion, may lose edges
- `pad`: Scale to fit, black bars — no distortion, full image preserved
- `rescale`: Stretch to exact dims — may distort

## Workflow 2: Iterative Image Refinement

Build up an image through conversational refinement using `previous_response_id`.

```python
# Round 1: Initial concept
resp1 = create_image(prompt="modern minimalist logo for AI company")
get_image_status(resp1.id)  # Wait for completion

# Round 2: Add details
resp2 = create_image(
    prompt="add blue and silver color scheme",
    previous_response_id=resp1.id
)
get_image_status(resp2.id)

# Round 3: Refine
resp3 = create_image(
    prompt="make it more geometric and abstract",
    previous_response_id=resp2.id
)
get_image_status(resp3.id)

# Download final version
download_image(resp3.id, filename="logo_final.png")
```

**When to use this vs generate_image:**
- `generate_image`: One-shot generation, no refinement needed
- `create_image` + `previous_response_id`: Multi-step refinement, building on previous results

## Workflow 3: Audio Generation + Playback

```python
# Generate speech
create_audio(
    text_prompt="Welcome to our product demo. Today we'll explore...",
    voice="onyx",
    instructions="Professional, confident tone. Slightly slower pace.",
    output_filename="demo_intro.mp3"
)

# Play it back
view_media(media_type="audio", filename="demo_intro.mp3")
```

**Available voices:**
| Voice | Character |
|-------|-----------|
| alloy | Neutral, balanced (default) |
| ash | Authoritative, confident |
| ballad | Warm, expressive |
| coral | Bright, energetic |
| echo | Deep, resonant |
| nova | Youthful, friendly |
| onyx | Professional, clear |
| sage | Calm, soothing |
| shimmer | Smooth, polished |

**Tips:**
- Use `instructions` for style guidance ("British accent", "whispered", "excited")
- Use `speed` (0.25-4.0) for pacing control
- Long text is auto-chunked (no length limit)

## Workflow 4: Image Editing + Composition

```python
# Edit a single image
edit_image(
    prompt="add a hat to the person",
    input_images=["portrait.png"]
)

# Multi-image composition (up to 16 images)
edit_image(
    prompt="create a gift basket containing all these items",
    input_images=["lotion.png", "soap.png", "candle.png"]
)

# Inpainting with mask
edit_image(
    prompt="add a flamingo standing in the water",
    input_images=["pool.png"],
    mask_filename="pool_mask.png"  # PNG with alpha channel
)
```

## Workflow 5: Reference Image Discovery

```python
# Find available images
images = list_reference_images(pattern="sunset*", file_type="png")

# Check dimensions
# (images include filename, size_bytes, file_type)

# Resize for Sora
prepare_reference_image(
    "sunset_original.jpg",
    "1280x720",
    resize_mode="crop"
)

# Animate
create_video(
    prompt="The sun rises slowly as clouds drift across the sky",
    input_reference_filename="sunset_original_1280x720.png",
    size="1280x720"
)
```

## Quality Optimization

**For best video quality:**
- Use `sora-2-pro` + `1280x720` or larger
- Keep clips to 4s for best instruction following
- Specify lighting explicitly (3-5 color anchors)
- Use concrete verbs and nouns, not vague adjectives

**For best image quality:**
- Use `generate_image` with `model="gpt-image-1.5"` + `quality="high"`
- Use `size="1536x1024"` for landscape detail
- Add style references: "photorealistic," "cinematic," "studio lighting"

**For speed/cost:**
- Use `sora-2` (not pro), `gpt-image-1-mini`, `generate_image` (synchronous)
