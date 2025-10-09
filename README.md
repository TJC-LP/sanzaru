# sora-mcp-server

A **stateless**, lightweight **FastMCP** server that wraps the **OpenAI Sora Video API** via the OpenAI Python SDK.

## Features
- **Video Generation**: Create Sora jobs (`sora-2` / `sora-2-pro`), optional image reference, optional remix
- **Image Generation**: Create reference images using GPT-5/GPT-4.1 with iterative refinement
- Get status, wait until completion (polling), download assets
- List and delete videos/images
- Stateless by default; no DB, no in-memory job tracking

> **Note:** Content guardrails are enforced by OpenAI. This server does not run local moderation.

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` environment variable
- `SORA_VIDEO_PATH` environment variable (directory for downloaded videos)
- `SORA_REFERENCE_PATH` environment variable (directory for reference images)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TJC-LP/sora-mcp-server.git
   cd sora-mcp-server
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   The script will:
   - Prompt for your OpenAI API key with hidden input (or use `$OPENAI_API_KEY` if set)
   - Create default directories with absolute paths in project root
   - Generate `.env` configuration file
   - Install dependencies with `uv sync`

3. **Start generating videos:**
   ```bash
   claude
   ```

That's it! Claude Code will automatically connect to the Sora MCP server and you can start generating videos.

## Manual Installation (without Claude Code)

If you want to run the server manually or integrate it with other MCP clients:

```bash
uv venv
uv sync

# Set environment variables and run
export OPENAI_API_KEY=sk-...
export SORA_VIDEO_PATH=~/sora-videos
export SORA_REFERENCE_PATH=~/sora-references
uv run sora-mcp-server
```

**Defaults:**
- `SORA_VIDEO_PATH` defaults to `./sora-videos`
- `SORA_REFERENCE_PATH` defaults to `./sora-references`

Both directories must exist before starting the server.

## MCP Tools

This runs an MCP server over stdio that exposes these tools:

### Video Generation
- `sora_create_video(prompt, model="sora-2", seconds?, size?, input_reference_filename?)`
  - Note: `seconds` must be a string: `"4"`, `"8"`, or `"12"` (not an integer)
  - Note: `size` must be one of: `"720x1280"`, `"1280x720"`, `"1024x1792"`, `"1792x1024"`
  - Note: `model` must be one of: `"sora-2"`, `"sora-2-pro"`
  - Note: `input_reference_filename` is a filename (not path) from `SORA_REFERENCE_PATH`
- `sora_get_status(video_id)` - Returns Video object with status/progress
- `sora_download(video_id, filename?, variant="video")` - Downloads to `SORA_VIDEO_PATH`
  - `filename` is optional - defaults to `{video_id}.{extension}` if not provided
  - Example: `sora_download(video_id, filename="my_video.mp4")`
- `sora_list(limit=20, after?, order="desc")` - Returns paginated list of videos
- `sora_delete(video_id)` - Deletes video from OpenAI storage
- `sora_remix(previous_video_id, prompt)` - Creates a remix

### Image Generation
- `image_create(prompt, model="gpt-5", size?, quality?, output_format?, background?, previous_response_id?)` - Generate images with GPT-5/GPT-4.1
  - Supported sizes: `1024x1024`, `1024x1536`, `1536x1024`, or `auto`
- `image_get_status(response_id)` - Check image generation status
- `image_download(response_id, filename?)` - Download completed image to `SORA_REFERENCE_PATH`

### Reference Image Management
- `sora_list_references(pattern?, file_type="all", sort_by="modified", order="desc", limit=50)` - Search reference images
- `sora_prepare_reference(input_filename, target_size, output_filename?, resize_mode="crop")` - Resize images to match Sora dimensions

> **Note:** To wait for completion, poll `sora_get_status` or `image_get_status` periodically rather than blocking. This keeps the LLM session responsive.

## Download variants
- `variant="video"` → `mp4`
- `variant="thumbnail"` → `webp`
- `variant="spritesheet"` → `jpg`

## Reference Images
- Supported formats: JPEG, PNG, WEBP
- Place reference images in `SORA_REFERENCE_PATH` directory
- Use `sora_list_references` to discover available images
- Reference image dimensions must match target video `size` parameter
- LLMs can only access images in the configured reference directory (security sandbox)

### Automatic Image Resizing
Use `sora_prepare_reference` to automatically resize any image to match Sora's required dimensions:

**Workflow:**
1. List available images: `sora_list_references()`
2. Prepare image: `sora_prepare_reference("photo.jpg", "1280x720", resize_mode="crop")`
3. Create video: `sora_create_video(prompt="...", size="1280x720", input_reference_filename="photo_1280x720.png")`

**Resize modes:**
- `crop` (default): Scale to cover target dimensions, center crop excess. No distortion, but may lose edges.
- `pad`: Scale to fit inside target, add black letterbox bars. No distortion, preserves full image.
- `rescale`: Stretch/squash to exact dimensions. May distort image, but uses full canvas with no cropping or padding.

The original image is preserved; a new resized PNG is created with dimensions in the filename.

## Image Generation

Generate reference images using OpenAI's Responses API with GPT-5 or GPT-4.1 models. Images are automatically saved to `SORA_REFERENCE_PATH` for use with Sora video generation.

### Basic Workflow
```
1. image_create(prompt="sunset over mountains") -> response_id
2. image_get_status(response_id) -> poll until status='completed'
3. image_download(response_id, filename="sunset.png")
4. sora_create_video(prompt="...", input_reference_filename="sunset.png")
```

### Iterative Refinement
Use `previous_response_id` to refine images conversationally:
```
1. resp1 = image_create(prompt="a cat")
2. Wait for completion with image_get_status(resp1.id)
3. resp2 = image_create(prompt="make it more realistic", previous_response_id=resp1.id)
4. Wait for completion with image_get_status(resp2.id)
5. image_download(resp2.id, filename="realistic_cat.png")
```

### Parameters
- **model**: `gpt-5` (default), `gpt-4.1`, or other models with image generation support
- **size**: `auto` (default), `1024x1024`, `1024x1536`, `1536x1024`
- **quality**: `high` (default), `low`, `medium`, `auto`
- **output_format**: `png` (default), `jpeg`, `webp`
- **background**: `auto` (default), `transparent`, `opaque`

### Combined Workflow
Generate a reference image and create a video from it:
```
1. image_create(prompt="futuristic cityscape at night", size="1280x720")
2. Poll with image_get_status until completed
3. image_download with custom filename
4. sora_create_video using the generated reference image
```

## Notes
- Download URLs from OpenAI are time-limited
- Videos are automatically saved to `SORA_VIDEO_PATH` when downloaded

## License
MIT
