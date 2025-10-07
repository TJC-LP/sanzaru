# sora-mcp

A **stateless**, lightweight **FastMCP** server that wraps the **OpenAI Sora Video API** via the OpenAI Python SDK.

## Features
- Create Sora jobs (`sora-2` / `sora-2-pro`), optional image reference, optional remix
- Get status, wait until completion (polling), download assets (`video|thumbnail|spritesheet`)
- List and delete videos
- Stateless by default; no DB, no in-memory job tracking

> **Note:** Content guardrails are enforced by OpenAI. This server does not run local moderation.

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` environment variable
- `SORA_VIDEO_PATH` environment variable (directory for downloaded videos)
- `SORA_REFERENCE_PATH` environment variable (directory for reference images)

## Install with `uv`
```bash
uv venv
uv sync
```

## Run
```bash
# Create directories for videos and reference images
mkdir -p ~/sora-videos ~/sora-references

# Set environment variables and run
export OPENAI_API_KEY=sk-...
export SORA_VIDEO_PATH=~/sora-videos
export SORA_REFERENCE_PATH=~/sora-references
uv run sora-mcp
```

**Defaults:**
- `SORA_VIDEO_PATH` defaults to `./sora-videos`
- `SORA_REFERENCE_PATH` defaults to `./sora-references`

Both directories must exist before starting the server.

## MCP Tools

This runs an MCP server over stdio that exposes these tools:

- `sora_create_video(prompt, model="sora-2", seconds?, size?, input_reference_filename?)`
  - Note: `seconds` must be a string: `"4"`, `"8"`, or `"12"` (not an integer)
  - Note: `size` must be one of: `"720x1280"`, `"1280x720"`, `"1024x1792"`, `"1792x1024"`
  - Note: `model` must be one of: `"sora-2"`, `"sora-2-pro"`
  - Note: `input_reference_filename` is a filename (not path) from `SORA_REFERENCE_PATH`
- `sora_get_status(video_id)` - Returns Video object with status/progress
- `sora_download(video_id, variant="video")` - Downloads to `SORA_VIDEO_PATH`
- `sora_list(limit=20, after?, order="desc")` - Returns paginated list of videos
- `sora_list_references(pattern?, file_type="all", sort_by="modified", order="desc", limit=50)` - Search reference images
- `sora_prepare_reference(input_filename, target_size, output_filename?, resize_mode="crop")` - Resize images to match Sora dimensions
- `sora_delete(video_id)` - Deletes video from OpenAI storage
- `sora_remix(previous_video_id, prompt)` - Creates a remix

> **Note:** To wait for video completion, poll `sora_get_status` periodically rather than blocking. This keeps the LLM session responsive.

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

The original image is preserved; a new resized PNG is created with dimensions in the filename.

## Notes
- Download URLs from OpenAI are time-limited
- Videos are automatically saved to `SORA_VIDEO_PATH` when downloaded

## License
MIT
