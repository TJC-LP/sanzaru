# API Reference

Complete documentation for all MCP tools exposed by the sanzaru server.

## Video Generation Tools

### `create_video`
Generate videos using OpenAI's Sora API.

**Parameters:**
- `prompt` (string, required): Text description of the video to generate
- `model` (string, optional): Model to use - `"sora-2"` (default) or `"sora-2-pro"`
- `seconds` (string, optional): Duration as string - `"4"`, `"8"`, or `"12"` (**NOTE:** Must be string, not integer)
- `size` (string, optional): Resolution - `"720x1280"`, `"1280x720"`, `"1024x1792"`, or `"1792x1024"`
- `input_reference_filename` (string, optional): Filename (not path) of reference image from `IMAGE_PATH`

**Returns:** Video object with `id`, `status`, `progress`, `model`, `seconds`, `size`

**Example:**
```python
video = create_video(
    prompt="A serene mountain landscape at sunrise",
    model="sora-2",
    seconds="8",
    size="1280x720"
)
```

---

### `get_video_status`
Check the status of a video generation job.

**Parameters:**
- `video_id` (string, required): ID returned from `create_video`

**Returns:** Video object with updated `status` and `progress`

**Status values:**
- `"queued"`: Job is queued
- `"in_progress"`: Currently generating (check `progress` field for 0-100%)
- `"completed"`: Ready to download
- `"failed"`: Generation failed

**Example:**
```python
status = get_video_status(video.id)
# Poll until status.status == "completed"
```

---

### `download_video`
Download a completed video to `VIDEO_PATH`.

**Parameters:**
- `video_id` (string, required): ID of completed video
- `filename` (string, optional): Custom filename (defaults to `{video_id}.{extension}`)
- `variant` (string, optional): What to download - `"video"` (default), `"thumbnail"`, or `"spritesheet"`

**Variant formats:**
- `"video"` → MP4 file
- `"thumbnail"` → WEBP image
- `"spritesheet"` → JPG image

**Returns:** DownloadResult with `filename`, `variant`

**Example:**
```python
result = download_video(video.id, filename="my_video.mp4")
# File saved to: {VIDEO_PATH}/my_video.mp4
```

---

### `list_videos`
List all video generation jobs with pagination.

**Parameters:**
- `limit` (integer, optional): Max results to return (default: 20, max: 100)
- `after` (string, optional): Cursor for pagination (use `last` from previous response)
- `order` (string, optional): Sort order - `"desc"` (default, newest first) or `"asc"`

**Returns:** Object with `data` (array of video summaries), `has_more` (boolean), `last` (cursor)

**Example:**
```python
page1 = list_videos(limit=20)
if page1.has_more:
    page2 = list_videos(limit=20, after=page1.last)
```

---

### `delete_video`
Permanently delete a video from OpenAI's storage.

**Parameters:**
- `video_id` (string, required): ID of video to delete

**Returns:** Confirmation with deleted video ID

**Warning:** This is permanent and cannot be undone!

---

### `remix_video`
Create a new video by remixing an existing completed video.

**Parameters:**
- `previous_video_id` (string, required): ID of completed video to remix
- `prompt` (string, required): New prompt to guide the remix

**Returns:** NEW Video object with different video_id

**Note:** This creates a brand new job. Poll the NEW video_id for completion.

---

### `list_local_videos`
List locally downloaded video files in `VIDEO_PATH`.

**Parameters:**
- `pattern` (string, optional): Glob pattern to filter filenames (e.g., `"*.mp4"`, `"sora*"`)
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
videos = list_local_videos(pattern="sora*", file_type="mp4")

# Get recently modified
recent = list_local_videos(sort_by="modified", order="desc", limit=10)
```

---

## Image Generation Tools

Two APIs are available for image generation:

| Tool | API | Best For |
|------|-----|----------|
| `generate_image` | Images API | New generation with gpt-image-1.5 (RECOMMENDED) |
| `edit_image` | Images API | Editing existing images |
| `create_image` | Responses API | Iterative refinement with `previous_response_id` |

**Images API** (gpt-image-1.5): Synchronous, returns immediately, no polling required, 20% cheaper
**Responses API** (GPT-5.2): Async polling pattern, supports iterative refinement chains, gpt-image-1.5 via tool_config

---

### `generate_image`
Generate images using OpenAI's Images API with gpt-image-1.5. **RECOMMENDED** for new image generation.

**Key advantages:**
- Synchronous - returns immediately (no polling)
- gpt-image-1.5 - state-of-the-art quality, better instruction following, improved text rendering
- Token usage tracking for cost monitoring
- 20% cheaper than Responses API

**Parameters:**
- `prompt` (string, required): Text description of the image (max 32k chars)
- `model` (string, optional): Model - `"gpt-image-1.5"` (default, recommended), `"gpt-image-1"`, `"gpt-image-1-mini"`, `"dall-e-3"`, `"dall-e-2"`
- `size` (string, optional): Dimensions - `"auto"` (default), `"1024x1024"`, `"1536x1024"` (landscape), `"1024x1536"` (portrait)
- `quality` (string, optional): Quality - `"auto"` (default), `"low"`, `"medium"`, `"high"`
- `background` (string, optional): Background - `"auto"` (default), `"transparent"`, `"opaque"`
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

# Transparent background for icons
result = generate_image(
    prompt="product icon, clean design",
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
Edit existing images using OpenAI's Images API with gpt-image-1.5.

**Key features:**
- Synchronous - returns immediately (no polling)
- Supports up to 16 input images for composition
- Mask-based inpainting
- Multi-image composition and blending

**Parameters:**
- `prompt` (string, required): Description of desired edits (max 32k chars)
- `input_images` (array, required): List of image filenames from `IMAGE_PATH` (1-16 images)
- `model` (string, optional): Model - `"gpt-image-1.5"` (default), `"gpt-image-1"`, `"gpt-image-1-mini"`
- `mask_filename` (string, optional): PNG mask with alpha channel for inpainting (transparent = edit, opaque = keep)
- `size` (string, optional): Output dimensions - `"auto"` (default), `"1024x1024"`, `"1536x1024"`, `"1024x1536"`
- `quality` (string, optional): Quality - `"auto"` (default), `"low"`, `"medium"`, `"high"`
- `background` (string, optional): Background - `"auto"` (default), `"transparent"`, `"opaque"`
- `output_format` (string, optional): Format - `"png"` (default), `"jpeg"`, `"webp"`
- `input_fidelity` (string, optional): Fidelity to input - `"high"` (preserve faces/style) or `"low"` (more creative freedom). gpt-image-1 only.
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

# High-fidelity face preservation
result = edit_image(
    prompt="change hair color to red",
    input_images=["portrait.jpg"],
    input_fidelity="high"
)
```

---

### `create_image`
Generate images using OpenAI's Responses API. Use for iterative refinement with `previous_response_id`.

**Tip:** Use `tool_config={"type": "image_generation", "model": "gpt-image-1.5"}` for best quality.

**Parameters:**
- `prompt` (string, required): Text description of image to generate
- `model` (string, optional): Model to use - `"gpt-5.2"` (default, OpenAI's latest), `"gpt-5.1"`, `"gpt-5"`, `"gpt-4.1"`
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
Resize images to match Sora's required dimensions.

**Parameters:**
- `input_filename` (string, required): Source image filename in `IMAGE_PATH`
- `target_size` (string, required): Target size - `"720x1280"`, `"1280x720"`, `"1024x1792"`, or `"1792x1024"`
- `output_filename` (string, optional): Custom output name (defaults to `{original}_{width}x{height}.png`)
- `resize_mode` (string, optional): How to handle aspect ratio - `"crop"` (default), `"pad"`, or `"rescale"`

**Resize modes:**
- **crop**: Scale to cover target, center crop excess (no distortion, may lose edges)
- **pad**: Scale to fit inside target, add black bars (no distortion, preserves full image)
- **rescale**: Stretch/squash to exact dimensions (may distort, no cropping/padding)

**Returns:** PrepareResult with `output_filename`, `original_size`, `target_size`, `resize_mode`

**Example:**
```python
result = prepare_reference_image(
    "photo.jpg",
    "1280x720",
    resize_mode="crop"
)
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
- `chat_with_audio` - GPT-4o audio analysis
- `transcribe_with_enhancement` - Enhanced transcription
- `create_audio` - Text-to-speech generation

---

## Best Practices

### Polling for Completion
Don't block - poll status periodically:
```python
# ❌ Don't block
video = create_video(...)
while get_video_status(video.id).status != "completed":
    # blocks LLM session

# ✅ Do poll with messaging
video = create_video(...)
status = get_video_status(video.id)
if status.status != "completed":
    return f"Video generating... {status.progress}% complete. Check back in a moment."
```

### File Security
- All file operations are sandboxed to configured paths
- Reference images must be in `IMAGE_PATH` (no path traversal)
- Symlinks are rejected for security
- Downloaded content goes to `VIDEO_PATH` or `IMAGE_PATH`

### Error Handling
All tools return structured error messages. Common errors:
- File not found in reference path
- Invalid dimensions for target size
- Video not completed yet
- API rate limits
