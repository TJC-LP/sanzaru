# Image Generation Guide

Complete guide to generating reference images using OpenAI's Responses API with GPT-5 and GPT-4.1.

## Overview

Generate high-quality reference images for use with Sora video generation. Images are created using OpenAI's Responses API with GPT-5 (recommended) or GPT-4.1 models, then automatically saved to `IMAGE_PATH` for use as video reference images.

## Basic Workflow

```python
# 1. Generate image
resp = create_image(prompt="sunset over mountains")

# 2. Poll for completion
status = get_image_status(resp.id)
# Check until status.status == "completed"

# 3. Download to IMAGE_PATH
download_image(resp.id, filename="sunset.png")

# 4. Use with video generation
video = create_video(
    prompt="The sun rises slowly over the peaks",
    input_reference_filename="sunset.png",
    size="1280x720"
)
```

## Models

### GPT-5 (Recommended)
```python
create_image(prompt="...", model="gpt-5")  # Default
```
**Best for:**
- High-quality reference images
- Complex scenes
- Photorealistic output
- Production use

**Characteristics:**
- Excellent prompt following
- High detail and quality
- Fast generation
- Cost-effective

### GPT-4.1
```python
create_image(prompt="...", model="gpt-4.1")
```
**Best for:**
- Alternative style
- Different artistic interpretation
- Experimentation

## Iterative Refinement

Use `previous_response_id` to refine images conversationally without starting over:

```python
# 1. Generate initial concept
resp1 = create_image(prompt="a futuristic cityscape")
get_image_status(resp1.id)  # wait for completion

# 2. Refine iteratively
resp2 = create_image(
    prompt="add more neon lights and flying cars",
    previous_response_id=resp1.id
)
get_image_status(resp2.id)  # wait

# 3. Further refinement
resp3 = create_image(
    prompt="make it nighttime with rain",
    previous_response_id=resp2.id
)
get_image_status(resp3.id)  # wait

# 4. Download final version
download_image(resp3.id, filename="cityscape_final.png")
```

**Benefits of iterative refinement:**
- Faster than regenerating from scratch
- Maintains consistent style and composition
- Fine-tune specific elements
- Conversational workflow

## Parameters

### Basic Parameters

```python
create_image(
    prompt="your description here",  # Required
    model="gpt-5",                    # Optional: "gpt-5" (default) or "gpt-4.1"
    previous_response_id="resp_123"  # Optional: for refinement
)
```

### Advanced Configuration

For advanced control, use `tool_config` parameter:

```python
from openai.types.responses.tool_param import ImageGeneration

config = ImageGeneration(
    type="image_generation",
    model="gpt-image-1",              # Image model
    size="1536x1024",                 # Image dimensions
    quality="high",                   # Quality level
    output_format="png",              # Format
    background="transparent",         # Background
    moderation="auto"                 # Content moderation
)

resp = create_image(
    prompt="logo for tech startup",
    model="gpt-5",                    # LLM model
    tool_config=config
)
```

**tool_config fields:**
- **model**: `"gpt-image-1"` (high quality) or `"gpt-image-1-mini"` (fast)
- **size**: `"auto"`, `"1024x1024"`, `"1024x1536"`, `"1536x1024"`
- **quality**: `"auto"`, `"low"`, `"medium"`, `"high"`
- **output_format**: `"png"`, `"jpeg"`, `"webp"`
- **background**: `"auto"`, `"transparent"`, `"opaque"`
- **moderation**: `"auto"` (default), `"low"` (more permissive)

## Image Editing

Edit existing images by providing input images:

### Single Image Editing

```python
# Place base image in IMAGE_PATH first
resp = create_image(
    prompt="add a flamingo to the pool",
    input_images=["pool.png"]
)
```

The tool will:
1. Read `pool.png` from `IMAGE_PATH`
2. Apply your edits (add flamingo)
3. Generate new image

### Multiple Image Composition

```python
# Combine multiple images
resp = create_image(
    prompt="create a gift basket with all these items",
    input_images=["lotion.png", "soap.png", "candle.png"]
)
```

**Image order matters:**
- First image receives highest fidelity
- Use most important image first
- Up to multiple images supported

### Masked Inpainting

```python
# Edit specific region using mask
resp = create_image(
    prompt="add logo to woman's shirt",
    input_images=["woman.jpg", "logo.png"],
    mask_filename="shirt_mask.png"
)
```

**Mask requirements:**
- PNG format with alpha channel
- Transparent = edit this area
- Black = keep original
- Mask must match first input image dimensions

## Complete Workflows

### Workflow 1: Generate Reference for Video

```python
# Step 1: Generate base image
print("Generating reference image...")
resp = create_image(
    prompt="A lone astronaut standing on a red desert planet, cinematic lighting",
    model="gpt-5"
)

# Step 2: Wait for completion
import time
while True:
    status = get_image_status(resp.id)
    if status.status == "completed":
        break
    elif status.status == "failed":
        raise Exception("Image generation failed")
    print(f"Status: {status.status}")
    time.sleep(2)

# Step 3: Download
result = download_image(resp.id, filename="astronaut.png")
print(f"Downloaded: {result.path}")

# Step 4: Prepare for video (if dimensions don't match)
prep = prepare_reference_image(
    "astronaut.png",
    "1280x720",
    resize_mode="crop"
)

# Step 5: Generate video
video = create_video(
    prompt="The astronaut turns and walks toward the horizon",
    size="1280x720",
    input_reference_filename=prep.output_filename
)
```

### Workflow 2: Iterative Design Process

```python
# Start with concept
print("Phase 1: Initial concept")
resp1 = create_image(prompt="modern minimalist logo for AI company")
get_image_status(resp1.id)  # wait

# Add details
print("Phase 2: Adding details")
resp2 = create_image(
    prompt="add blue and silver color scheme",
    previous_response_id=resp1.id
)
get_image_status(resp2.id)  # wait

# Refine style
print("Phase 3: Style refinement")
resp3 = create_image(
    prompt="make it more geometric and abstract",
    previous_response_id=resp2.id
)
get_image_status(resp3.id)  # wait

# Final touches
print("Phase 4: Final touches")
resp4 = create_image(
    prompt="add subtle gradient and increase contrast",
    previous_response_id=resp3.id
)
get_image_status(resp4.id)  # wait

# Download final
download_image(resp4.id, filename="logo_final.png")
```

### Workflow 3: Product Visualization

```python
# Generate base product image
resp = create_image(
    prompt="luxury perfume bottle on marble surface, studio lighting",
    model="gpt-5"
)
get_image_status(resp.id)  # wait
download_image(resp.id, filename="perfume_base.png")

# Create variant 1: Different angle
resp2 = create_image(
    prompt="rotate 45 degrees, show side profile",
    previous_response_id=resp.id
)
get_image_status(resp2.id)  # wait
download_image(resp2.id, filename="perfume_side.png")

# Create variant 2: Different setting
resp3 = create_image(
    prompt="place on dark velvet fabric with soft rim lighting",
    previous_response_id=resp.id
)
get_image_status(resp3.id)  # wait
download_image(resp3.id, filename="perfume_luxury.png")
```

### Workflow 4: Logo Placement on Images

```python
# You have: company_shirt.jpg, company_logo.png in IMAGE_PATH

# Add logo to shirt
resp = create_image(
    prompt="place the logo on the chest area of the shirt",
    input_images=["company_shirt.jpg", "company_logo.png"],
    tool_config=ImageGeneration(
        type="image_generation",
        input_fidelity="high"  # Preserve quality
    )
)

get_image_status(resp.id)  # wait
download_image(resp.id, filename="shirt_with_logo.png")

# Now animate it
video = create_video(
    prompt="The person wearing the shirt turns and smiles at the camera",
    input_reference_filename="shirt_with_logo.png",
    size="1280x720"
)
```

## Best Practices

### Prompting Tips

**Be specific:**
```python
# ❌ Vague
create_image(prompt="a nice landscape")

# ✅ Specific
create_image(prompt="mountain landscape at golden hour, pine trees in foreground, snow-capped peaks, volumetric fog")
```

**Use artistic references:**
```python
create_image(prompt="portrait in the style of Rembrandt, dramatic lighting, oil painting texture")
```

**Specify composition:**
```python
create_image(prompt="wide shot of city skyline, rule of thirds composition, sunset backlighting")
```

### Quality Optimization

**For highest quality:**
```python
config = ImageGeneration(
    type="image_generation",
    model="gpt-image-1",     # Not mini
    quality="high",
    output_format="png"      # Lossless
)
create_image(prompt="...", tool_config=config)
```

**For speed:**
```python
config = ImageGeneration(
    type="image_generation",
    model="gpt-image-1-mini",  # Faster model
    quality="medium"
)
create_image(prompt="...", tool_config=config)
```

### File Management

**Use descriptive filenames:**
```python
download_image(resp.id, filename="project_hero_v3.png")
# Better than: image_123.png
```

**Organize by project:**
```bash
IMAGE_PATH/
├── project_alpha/
│   ├── hero_image.png
│   ├── logo_variants/
│   └── backgrounds/
└── project_beta/
    └── references/
```

### Error Handling

```python
resp = create_image(prompt="...")

# Check status
status = get_image_status(resp.id)

if status.status == "failed":
    print(f"Generation failed for: {resp.id}")
    # Try with different prompt or settings
elif status.status == "completed":
    download_image(resp.id)
else:
    print(f"Still generating: {status.status}")
```

## Common Issues

### Image Not Completing
**Problem:** Status stuck at "in_progress"
**Solutions:**
- Wait longer (complex images take time)
- Check OpenAI status page for service issues
- Try simpler prompt
- Use `model="gpt-4.1"` as alternative

### Low Quality Output
**Problem:** Image doesn't match expectations
**Solutions:**
- Add more descriptive details to prompt
- Use `quality="high"` in tool_config
- Try iterative refinement
- Add style references ("photorealistic", "cinematic")

### Dimensions Don't Match Sora
**Problem:** Generated image wrong size for video
**Solutions:**
- Use `prepare_reference_image` after download
- Or specify size in tool_config:
  ```python
  tool_config=ImageGeneration(
      type="image_generation",
      size="1536x1024"  # Matches Sora 1792x1024 better
  )
  ```

### Content Filtered
**Problem:** "Moderation" error
**Solutions:**
- Revise prompt to avoid policy violations
- Try `moderation="low"` for borderline content
- Check OpenAI usage policies

## API Limits

- **Rate limits:** Varies by account tier
- **Size limits:** Input images must be reasonable size
- **Timeout:** Complex generations may take 30-60 seconds
- **File formats:** JPEG, PNG, WEBP supported

## Next Steps

1. **Test basic generation:** Start with simple prompts
2. **Experiment with refinement:** Try iterative workflows
3. **Combine with video:** Generate → Download → Prepare → Animate
4. **Explore advanced features:** Image editing, masking, composition

See also:
- [API Reference](api-reference.md) - Complete parameter documentation
- [Reference Images](reference-images.md) - Using images with Sora
- [Sora Prompting Guide](sora2-prompting-guide.md) - Video generation tips
