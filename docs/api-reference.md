# API Reference

Complete documentation for all MCP tools exposed by the sanzaru server.

## Video Generation Tools

Video runs on **Higgsfield** (`HF_KEY`, 0.13.0+). OpenAI removed the Sora Videos API on
2026-09-24; `list_videos`, `delete_video` and `remix_video` are gone (use `edit_video` /
`extend_video`). These tools register only when `HF_KEY` is set. Jobs are asynchronous and
return an `hf_<uuid>` id; wait with `wait_for([id], download=True)`.

**Every job is priced before it is submitted.** The result carries `cost` = `{usd, credits,
basis: "api" | "local_table" | "unpriced" | "unavailable", usd_after_discount,
pricing_description, note}`. `max_cost_usd` refuses an over-budget job (nothing uploaded or
charged); `dry_run=True` validates and prices without submitting. The Higgsfield API is prepaid
and billed separately from a Higgsfield app subscription.

### `create_video`
Text-to-video, or image-to-video when `reference_image` is given.

**Parameters:**
- `prompt` (string, required): What happens. With a reference image, describe only the motion
- `model` (string, optional): `"seedance-2.5"` (default), `"kling-3.0-std"`, `"kling-3.0-pro"`,
  `"kling-3.0-4k"`, `"kling-3.0-turbo"`, or any Higgsfield catalog id (`"vendor/model/operation"`)
- `duration` (integer, optional): Seconds — Seedance 4-30, Kling 3-15 (model default 5)
- `aspect_ratio` (string, optional): Text-to-video only — `"16:9"`, `"4:3"`, `"1:1"`, `"3:4"`,
  `"9:16"`, `"21:9"` (Kling: 16:9 / 9:16 / 1:1)
- `resolution` (string, optional): `"480p"` or `"720p"` (Seedance; Kling's tier fixes it)
- `audio` (boolean, optional): Generate a soundtrack (model default: on)
- `reference_image` (string, optional): Start frame — a filename in the reference directory, or
  the `hf_…` id of a completed job
- `end_image` (string, optional): Last frame (needs `reference_image`)
- `extra` (object, optional): Model-specific parameters, e.g. `{"bitrate_mode": "standard"}`
- `max_cost_usd` (number, optional): Refuse if the estimate is higher
- `dry_run` (boolean, optional): Price and validate only

**Returns:** `VideoJob` — `id` (`hf_…`, null for a dry run), `status`, `model`, `slug`,
`operation`, `cost`, `arguments`

**Example:**
```python
job = create_video(
    prompt="A serene mountain landscape at sunrise, slow aerial push-in",
    duration=8, aspect_ratio="16:9", resolution="720p",
    max_cost_usd=5,
)
wait_for([job.id], download=True)
```

---

### `edit_video` / `extend_video`
Re-render (edit) or continue (extend) an existing clip with Seedance 2.5.

**Parameters:**
- `prompt` (string, required): The change (edit) or what happens next (extend)
- `source_video` (string, required): An `.mp4` in the video directory, or the `hf_…` id of a
  completed job (its output is reused directly — no download)
- `duration` (integer, extend only): Seconds to add, 4-30 (default 5). Edit keeps the source length
- `model`, `resolution`, `audio`, `extra`, `max_cost_usd`, `dry_run`: as in `create_video`
- `reference_images` (array of strings, optional): Guide images from the reference directory

**Billing:** the source is billed as well as the output, at Seedance's 0.6× video-input rate
(~$0.28 per billed second at 720p).

---

### `get_video_status`
One status check. To wait, use `wait_for` instead of calling this in a loop.

**Returns:** `VideoStatus` — `id`, `status` (`queued` / `in_progress` / `completed` / `failed` /
`nsfw` / `canceled`), `done`, `error`, `video_url` (kept ~7 days)

---

### `download_video`
Save a completed job's output into the video directory (usually done by `wait_for(download=True)`).

**Parameters:**
- `video_id` (string, required): The `hf_…` id
- `filename` (string, optional): Defaults to `{video_id}.mp4`

**Returns:** `DownloadResult` — `filename`, `format`

---

### `cancel_video`
Cancel a job that is still queued (refunded). A started job cannot be canceled.

**Returns:** `{"id": ..., "canceled": true}`

---

### `list_local_videos`
List locally downloaded video files in `VIDEO_PATH`.

**Parameters:**
- `pattern` (string, optional): Glob pattern to filter filenames (e.g., `"*.mp4"`, `"hf_*"`)
- `file_type` (string, optional): Filter by type - `"mp4"`, `"webm"`, `"mov"`, or `"all"` (default)
- `sort_by` (string, optional): Sort by `"name"`, `"size"`, or `"modified"` (default)
- `order` (string, optional): `"desc"` (default) or `"asc"`
- `limit` (integer, optional): Max results (default: 50)

**Returns:** Object with `data` (array of VideoFile objects with `filename`, `size_bytes`, `modified_timestamp`, `file_type`)

**Example:**
```python
# List all local videos
videos = list_local_videos()

# Find MP4 files matching a pattern
videos = list_local_videos(pattern="hf_*", file_type="mp4")

# Get recently modified
recent = list_local_videos(sort_by="modified", order="desc", limit=10)
```

---

## Image Generation Tools

Two APIs are available for image generation:

| Tool | API | Best For |
|------|-----|----------|
| `generate_image` | Images API | New generation with gpt-image-2.5 (RECOMMENDED) |
| `edit_image` | Images API | Editing existing images |
| `create_image` | Responses API | Iterative refinement with `previous_response_id` |

**Images API** (gpt-image-2.5 default): Synchronous, returns immediately, no polling required, up to 4K output
**Responses API** (gpt-6-astra by default): Async polling pattern, supports iterative refinement chains + `action` field, gpt-image-2.5 (or gpt-image-2) via tool_config

---

### `generate_image`
Generate images using OpenAI's Images API with gpt-image-2.5-flare (default). **RECOMMENDED** for new image generation.

**Key advantages:**
- Synchronous - returns immediately (no polling)
- gpt-image-2.5-flare / gpt-image-2.5-sunburst - state-of-the-art, transparent backgrounds, `xhigh`/`max` quality, up to 4K output; gpt-image-2 - previous flagship (~99% text accuracy)
- Token usage tracking for cost monitoring
- Accepts thousands of valid resolutions (not just the documented presets)

**Parameters:**
- `prompt` (string, required): Text description of the image (max 32k chars)
- `model` (string, optional): Model - `"gpt-image-2.5-flare"` (default, recommended), `"gpt-image-2.5-sunburst"`, `"gpt-image-2"`, `"gpt-image-1.5"`, `"gpt-image-1"`, `"gpt-image-1-mini"`, `"dall-e-3"`, `"dall-e-2"`
- `size` (string, optional): Dimensions - `"auto"` (default), `"1024x1024"`, `"1536x1024"`, `"1024x1536"`, plus gpt-image-2.5/gpt-image-2 sizes `"2048x2048"`, `"2048x1152"`, `"3840x2160"`, `"2160x3840"`
- `quality` (string, optional): Quality - `"auto"` (default), `"low"`, `"medium"`, `"high"`
- `background` (string, optional): Background - `"auto"` (default), `"transparent"` (NOT supported on gpt-image-2; fine on gpt-image-2.5 — use gpt-image-1.5), `"opaque"`
- `output_format` (string, optional): Format - `"png"` (default), `"jpeg"`, `"webp"`
- `moderation` (string, optional): Content moderation - `"auto"` (default), `"low"`
- `filename` (string, optional): Custom output filename (auto-generated if omitted)

**Returns:** ImageGenerateResult with `filename`, `size`, `format`, `model`, `usage`

**Usage tracking:** Returns token counts for cost monitoring:
```python
result.usage.input_tokens   # Text tokens
result.usage.output_tokens  # Image tokens
result.usage.total_tokens   # Combined total
```

**Examples:**
```python
# Basic generation (recommended path)
result = generate_image(prompt="a sunset over mountains")
# File immediately available at result.path

# High quality portrait
result = generate_image(
    prompt="professional headshot, studio lighting",
    size="1024x1536",
    quality="high"
)

# Transparent background for icons (the default gpt-image-2.5 model supports it)
result = generate_image(
    prompt="product icon, clean design",
    model="gpt-image-1.5",
    background="transparent",
    output_format="png"
)

# Fast generation with mini model
result = generate_image(
    prompt="quick sketch of a cat",
    model="gpt-image-1-mini"
)
```

---

### `edit_image`
Edit existing images using OpenAI's Images API with gpt-image-2.5-sunburst (default).

**Key features:**
- Synchronous - returns immediately (no polling)
- Supports up to 16 input images for composition
- Mask-based inpainting
- Multi-image composition and blending

**Parameters:**
- `prompt` (string, required): Description of desired edits (max 32k chars)
- `input_images` (array, required): List of image filenames from `IMAGE_PATH` (1-16 images)
- `model` (string, optional): Model - `"gpt-image-2.5-sunburst"` (default), `"gpt-image-2.5-flare"`, `"gpt-image-2"`, `"gpt-image-1.5"`, `"gpt-image-1"`, `"gpt-image-1-mini"`
- `mask_filename` (string, optional): PNG mask with alpha channel for inpainting (transparent = edit, opaque = keep)
- `size` (string, optional): Output dimensions - `"auto"` (default), `"1024x1024"`, `"1536x1024"`, `"1024x1536"`, plus gpt-image-2.5/gpt-image-2 sizes `"2048x2048"`, `"2048x1152"`, `"3840x2160"`, `"2160x3840"`
- `quality` (string, optional): Quality - `"auto"` (default), `"low"`, `"medium"`, `"high"`
- `background` (string, optional): Background - `"auto"` (default), `"transparent"` (NOT supported on gpt-image-2), `"opaque"`
- `output_format` (string, optional): Format - `"png"` (default), `"jpeg"`, `"webp"`
- `input_fidelity` (string, optional): Fidelity to input - `"high"` (preserve faces/style) or `"low"` (more creative freedom). gpt-image-1 / gpt-image-1.5 only. Silently stripped for gpt-image-2 and gpt-image-2.5 (always high fidelity; the API rejects the flag).
- `filename` (string, optional): Custom output filename

**Returns:** ImageGenerateResult with `filename`, `size`, `format`, `model`, `usage`

**Examples:**
```python
# Simple edit
result = edit_image(
    prompt="add a hat to the person",
    input_images=["portrait.png"]
)

# Multi-image composition
result = edit_image(
    prompt="create a gift basket containing all these items",
    input_images=["lotion.png", "soap.png", "candle.png"]
)

# Inpainting with mask
result = edit_image(
    prompt="add a flamingo standing in the water",
    input_images=["pool.png"],
    mask_filename="pool_mask.png"
)

# High-fidelity face preservation on gpt-image-1.5
result = edit_image(
    prompt="change hair color to red",
    input_images=["portrait.jpg"],
    model="gpt-image-1.5",
    input_fidelity="high",
)
```

---

### `create_image`
Generate images using OpenAI's Responses API. Use for iterative refinement with `previous_response_id`.

**Tip:** The image model defaults to gpt-image-2.5-flare; pin `"gpt-image-2.5-sunburst"` for precision edits. Use `"gpt-image-1.5"` when you need transparent backgrounds. You can also pass `action: "generate"` / `"edit"` to force a mode when an image is in context (default `"auto"`).

**Parameters:**
- `prompt` (string, required): Text description of image to generate
- `model` (string, optional): Mainline model that drives the image tool - `"gpt-6-astra"` (default, flagship), `"gpt-5.6-sol"`, `"gpt-5.6-terra"` (balanced cost), `"gpt-5.6-luna"` (cheapest). Its tokens bill on top of the image; an image turn is a few hundred tokens.
- `tool_config` (object, optional): Advanced configuration (ImageGeneration type)
- `previous_response_id` (string, optional): Previous response ID for iterative refinement
- `input_images` (array, optional): Array of filenames from `IMAGE_PATH` for image editing
- `mask_filename` (string, optional): PNG mask file for inpainting

**Returns:** ImageResponse with `id`, `status`, `created_at`

**Example:**
```python
# Generate from text
resp = create_image(prompt="sunset over mountains")

# Iterative refinement
resp2 = create_image(
    prompt="add more dramatic clouds",
    previous_response_id=resp.id
)

# Image editing
resp3 = create_image(
    prompt="add a flamingo to the pool",
    input_images=["pool.png"]
)
```

---

### `get_image_status`
Check status of image generation job.

**Parameters:**
- `response_id` (string, required): ID returned from `create_image`

**Returns:** ImageResponse with updated status

---

### `download_image`
Download completed image to `IMAGE_PATH`.

**Parameters:**
- `response_id` (string, required): ID of completed image
- `filename` (string, optional): Custom filename (auto-generated if omitted)

**Returns:** ImageDownloadResult with `filename`, `size`, `format`

---

## Reference Image Management Tools

### `list_reference_images`
Search and list available reference images in `IMAGE_PATH`.

**Parameters:**
- `pattern` (string, optional): Glob pattern to filter filenames (e.g., `"cat*.png"`, `"*.jpg"`)
- `file_type` (string, optional): Filter by type - `"jpeg"`, `"png"`, `"webp"`, or `"all"` (default)
- `sort_by` (string, optional): Sort by `"name"`, `"size"`, or `"modified"` (default)
- `order` (string, optional): `"desc"` (default) or `"asc"`
- `limit` (integer, optional): Max results (default: 50)

**Returns:** Array of ReferenceImage objects with `filename`, `size_bytes`, `modified_timestamp`, `file_type`

**Example:**
```python
# Find all dog images
images = list_reference_images(pattern="dog*", file_type="png")

# Get recently modified
recent = list_reference_images(sort_by="modified", order="desc", limit=10)
```

---

### `prepare_reference_image`
Crop, pad or rescale an image to a video frame shape. Image-to-video has no `aspect_ratio`
(framing follows the start image), so this is how you choose the shape of an animated reference.

**Parameters:**
- `input_filename` (string, required): Source image filename in the reference directory
- `aspect_ratio` (string, optional): `"16:9"`, `"4:3"`, `"1:1"`, `"3:4"`, `"9:16"`, `"21:9"` —
  target frame from Seedance's 720p sizes (e.g. 16:9 → 1280x720)
- `size` (string, optional): Exact `"WxH"` target (64-4096 per edge). Pass exactly one of
  `aspect_ratio` / `size`
- `output_filename` (string, optional): Custom output name (defaults to `{original}_{width}x{height}.png`)
- `resize_mode` (string, optional): `"crop"` (default), `"pad"`, or `"rescale"`

**Resize modes:**
- **crop**: Scale to cover target, center crop excess (no distortion, may lose edges)
- **pad**: Scale to fit inside target, add black bars (no distortion, preserves full image)
- **rescale**: Stretch/squash to exact dimensions (may distort, no cropping/padding)

**Returns:** PrepareResult with `output_filename`, `original_size`, `target_size`, `resize_mode`

**Example:**
```python
result = prepare_reference_image("photo.jpg", aspect_ratio="16:9", resize_mode="crop")
# Creates: photo_1280x720.png
```

---

## Audio Tools

For detailed audio tool documentation, see [docs/audio/README.md](audio/README.md).

**Available tools:**
- `list_audio_files` - List and filter audio files
- `get_latest_audio` - Get most recent audio file
- `convert_audio` - Convert to mp3/wav
- `compress_audio` - Compress for API limits
- `transcribe_audio` - Whisper transcription
- `chat_with_audio` - audio analysis with OpenAI's audio chat models (default `gpt-audio-1.5`)
- `transcribe_with_enhancement` - Enhanced transcription
- `create_audio` - Text-to-speech generation

---

## Best Practices

### Waiting for completion
Don't poll in a loop — hand the ids to `wait_for`, which blocks server-side, reports progress to
the client on every poll, and returns on its deadline (call again to keep waiting):
```python
job = create_video(prompt="...", max_cost_usd=3)
result = wait_for([job.id], download=True)   # file saved on completion
```

### File Security
- All file operations are sandboxed to configured paths
- Reference images must be in `IMAGE_PATH` (no path traversal)
- Symlinks are rejected for security
- Downloaded content goes to `VIDEO_PATH` or `IMAGE_PATH`

### Error Handling
All tools return structured error messages. Common errors:
- File not found in reference path
- Over the cost cap (nothing submitted)
- Video not completed yet
- Account concurrency limit (4 Higgsfield jobs in flight — resubmit after one finishes)
- API rate limits
