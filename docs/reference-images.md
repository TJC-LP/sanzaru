# Reference Images Guide

Complete guide to working with reference images for Sora video generation.

## Overview

Reference images allow you to start video generation from a specific visual starting point instead of generating from text alone. Sora will animate from your reference image, maintaining its style, composition, and subject matter.

## Supported Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **WEBP** (.webp)

All formats are supported for both input and output. Prepared images are saved as PNG for best quality.

## File Management

### Setting Up Your Reference Directory

1. Set `IMAGE_PATH` environment variable to your reference images directory
2. Place images in this directory
3. The server only has access to files in this directory (security sandbox)

```bash
export IMAGE_PATH=~/sanzaru/images
mkdir -p $IMAGE_PATH
```

### Discovering Available Images

Use `list_reference_images` to search your reference directory:

```python
# List all images
all_images = list_reference_images()

# Find specific images
dogs = list_reference_images(pattern="dog*", file_type="png")

# Get recent uploads
recent = list_reference_images(sort_by="modified", order="desc", limit=10)

# Find large files
large = list_reference_images(sort_by="size", order="desc", limit=5)
```

## Dimension Requirements

Reference images **must match** your target video dimensions exactly:

| Video Size | Required Image Size |
|------------|-------------------|
| `720x1280` | 720×1280 pixels |
| `1280x720` | 1280×720 pixels |
| `1024x1792` | 1024×1792 pixels (pro only) |
| `1792x1024` | 1792×1024 pixels (pro only) |

**If dimensions don't match**, use `prepare_reference_image` to resize automatically.

## Automatic Image Resizing

### Basic Workflow

1. **List** available images
2. **Prepare** image to target dimensions
3. **Create** video with prepared image

```python
# 1. Find your image
images = list_reference_images(pattern="sunset*")

# 2. Resize to match video dimensions
result = prepare_reference_image(
    "sunset.jpg",
    "1280x720",
    resize_mode="crop"
)

# 3. Generate video
video = create_video(
    prompt="The sun rises slowly, casting golden light across the landscape",
    size="1280x720",
    input_reference_filename="sunset_1280x720.png"
)
```

### Resize Modes Explained

#### `crop` (Default - Recommended)
**Best for:** Most use cases where you want to avoid distortion

- Scales image to **cover** the target dimensions
- Centers the image and crops excess
- **Preserves aspect ratio** (no distortion)
- May lose some edges

**Example:** 1920×1080 → 1280×720
- Image is scaled to 1280×720 coverage
- Excess on edges is cropped off
- Result: No distortion, but some content may be cut

#### `pad` (Letterbox)
**Best for:** When you need to preserve the entire image

- Scales image to **fit inside** the target dimensions
- Adds black bars (letterbox/pillarbox) to fill gaps
- **Preserves aspect ratio** (no distortion)
- Preserves full image content

**Example:** 1920×1080 → 1280×720
- Image is scaled to fit within 1280×720
- Black bars added on sides
- Result: Entire image visible, but with black bars

#### `rescale` (Stretch)
**Best for:** Rare cases where distortion is acceptable

- Stretches/squashes to **exact** dimensions
- Uses full canvas (no cropping or padding)
- **May distort** the image
- No content loss, but aspect ratio changes

**Example:** 1920×1080 → 1280×720
- Image is stretched horizontally and/or vertically
- Result: No bars or cropping, but may look distorted

### Choosing the Right Mode

```python
# Portrait photo → landscape video
# Use crop to avoid black bars
prepare_reference_image("portrait.jpg", "1280x720", resize_mode="crop")

# Logo or graphic that must stay visible
# Use pad to preserve everything
prepare_reference_image("logo.png", "1280x720", resize_mode="pad")

# Abstract art where distortion is ok
# Use rescale for full canvas usage
prepare_reference_image("abstract.jpg", "1280x720", resize_mode="rescale")
```

## Prompting with Reference Images

**CRITICAL:** When using reference images, your prompt should describe **motion and action only**, not what's already in the image.

### ❌ Bad Prompts (Re-describing the image)
```python
# Don't do this:
create_video(
    prompt="A pilot in an orange suit sitting in a cockpit with glowing instruments...",
    input_reference_filename="pilot.png"
)
```

The image already shows: character, setting, framing, style, lighting.

### ✅ Good Prompts (Describing the action)
```python
# Do this instead:
create_video(
    prompt="The pilot glances up, takes a breath, then returns focus to the instruments.",
    input_reference_filename="pilot.png"
)
```

**What to include in your prompt:**
- What happens next
- Motion and camera movement
- Actions and transformations
- Timing and pacing

**What NOT to include:**
- Descriptions of what's already visible
- Style, lighting, or composition (already in the image)
- Character or setting descriptions

See [docs/sora2-prompting-guide.md](sora2-prompting-guide.md) for comprehensive prompting guidance.

## Complete Example Workflows

### Workflow 1: Using Existing Images

```python
# 1. Search for available images
sunset_images = list_reference_images(pattern="sunset*", file_type="jpg")

# 2. Check if image needs resizing
# (if you know the dimensions don't match 1280x720)
result = prepare_reference_image(
    "sunset_original.jpg",
    "1280x720",
    resize_mode="crop"
)

# 3. Generate video with simple motion prompt
video = create_video(
    prompt="The sun rises slowly over the horizon as clouds drift by",
    model="sora-2",
    seconds="8",
    size="1280x720",
    input_reference_filename="sunset_original_1280x720.png"
)

# 4. Poll for completion
status = get_video_status(video.id)
# Keep checking until status.status == "completed"

# 5. Download when ready
download_video(video.id, filename="sunrise_animation.mp4")
```

### Workflow 2: Generate Reference Image First

```python
# 1. Generate a reference image
resp = create_image(
    prompt="futuristic pilot in mech cockpit, cinematic lighting",
    model="gpt-5"
)

# 2. Wait for completion
status = get_image_status(resp.id)

# 3. Download the image
download_image(resp.id, filename="pilot.png")

# 4. Resize for Sora (if needed)
prepare_reference_image("pilot.png", "1280x720", resize_mode="crop")

# 5. Animate it
video = create_video(
    prompt="The pilot looks up and smiles",
    size="1280x720",
    input_reference_filename="pilot_1280x720.png"
)
```

### Workflow 3: Iterative Refinement

```python
# 1. Generate base image
resp1 = create_image(prompt="a cyberpunk character")
get_image_status(resp1.id)  # wait
download_image(resp1.id, filename="character_v1.png")

# 2. Refine the image
resp2 = create_image(
    prompt="add more neon details and a cityscape background",
    previous_response_id=resp1.id
)
get_image_status(resp2.id)  # wait
download_image(resp2.id, filename="character_v2.png")

# 3. Prepare for video
prepare_reference_image("character_v2.png", "1280x720", resize_mode="crop")

# 4. Animate
video = create_video(
    prompt="Character turns and walks into the neon-lit street",
    size="1280x720",
    input_reference_filename="character_v2_1280x720.png"
)
```

## Security & Best Practices

### File Security
- All images must be in `IMAGE_PATH` directory
- Path traversal is blocked (`../` not allowed)
- Symlinks are rejected
- Only specify filenames, never full paths

### Performance Tips
- Prepare images before peak generation time
- Use `crop` mode for best quality/speed balance
- PNG output ensures no quality loss during resizing
- Original files are always preserved

### Naming Conventions
The tool automatically creates descriptive filenames:
```
original.jpg → original_1280x720.png (after prepare)
photo.png → photo_1920x1080.png (after prepare)
```

You can also specify custom names:
```python
prepare_reference_image(
    "sunset.jpg",
    "1280x720",
    output_filename="my_custom_name.png"
)
```

## Troubleshooting

### "File not found" Error
- Check that file is in `IMAGE_PATH` directory
- Use `list_reference_images()` to verify filename
- Filenames are case-sensitive

### "Invalid dimensions" Error
- Target size must be one of: `720x1280`, `1280x720`, `1024x1792`, `1792x1024`
- Use exact string format (e.g., `"1280x720"`)

### Video Doesn't Match Reference
- Check that you used the correctly sized image (`{name}_{width}x{height}.png`)
- Verify the prompt focuses on action, not re-describing the image
- Try a simpler prompt describing only motion

### Image Looks Distorted
- Try different resize mode (`crop` usually works best)
- Check original image aspect ratio
- Consider using `pad` to preserve full image without distortion
