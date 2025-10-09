# SPDX-License-Identifier: MIT
"""Tool descriptions for MCP server.

These descriptions are LLM-facing and optimized for Claude and other AI assistants
to understand how to use each tool effectively.
"""

# ==================== VIDEO TOOL DESCRIPTIONS ====================

CREATE_VIDEO = """Create a new Sora video generation job. This starts an async job and returns immediately with a video_id.

The video is NOT ready immediately - use get_video_status(video_id) to poll for completion.
Status will be 'queued' -> 'in_progress' -> 'completed' or 'failed'.
Once status='completed', use download_video(video_id) to save the video to disk.

Parameters:
- prompt: Text description of the video to generate (required)
- model: "sora-2" (faster, cheaper) or "sora-2-pro" (higher quality). Default: "sora-2"
- seconds: Duration as string "4", "8", or "12" (NOT an integer). Default: varies by model
- size: Resolution as "720x1280" (portrait), "1280x720" (landscape), "1024x1792", or "1792x1024". Default: "720x1280"
- input_reference_filename: Filename of reference image in SORA_REFERENCE_PATH (e.g., "cat.png"). Use list_reference_images to find available images. Image must match target size. Supported: JPEG, PNG, WEBP. Optional.

Returns Video object with fields: id, status, progress, model, seconds, size."""

GET_VIDEO_STATUS = """Check the status and progress of a video generation job.

Use this to poll for completion after calling create_video or remix_video.
Call this repeatedly (e.g. every 5-10 seconds) until status changes from 'queued'/'in_progress' to 'completed' or 'failed'.

The returned Video object contains:
- status: "queued" | "in_progress" | "completed" | "failed"
- progress: Integer 0-100 showing completion percentage
- id: The video_id for use with other tools
- Other metadata: model, seconds, size, created_at, etc.

Typical workflow:
1. Create video with create_video() -> get video_id
2. Poll with get_video_status(video_id) until status='completed'
3. Download with download_video(video_id)"""

DOWNLOAD_VIDEO = """Download a completed video to disk.

IMPORTANT: Only call this AFTER get_video_status shows status='completed'.
If the video is not completed, this will fail.

The video is automatically saved to the directory configured in SORA_VIDEO_PATH.
Returns the absolute path to the downloaded file.

Parameters:
- video_id: The ID from create_video or remix_video (required)
- filename: Custom filename (optional, defaults to video_id with appropriate extension)
- variant: What to download (default: "video")
  * "video" -> MP4 video file
  * "thumbnail" -> WEBP thumbnail image
  * "spritesheet" -> JPG spritesheet of frames

Typical workflow:
1. Create: create_video() -> video_id
2. Poll: get_video_status(video_id) until status='completed'
3. Download: download_video(video_id, filename="my_video.mp4") -> returns local file path

Returns DownloadResult with: filename, path, variant"""

LIST_VIDEOS = """List all video jobs in your OpenAI account with pagination support.

Returns a paginated list of all videos (completed, in-progress, failed, etc.).
Each video summary includes: id, status, progress, created_at, model, seconds, size.

Parameters:
- limit: Max number of videos to return (default: 20, max: 100)
- after: For pagination, pass the 'last' id from previous response (optional)
- order: "desc" for newest first (default) or "asc" for oldest first

Returns:
- data: Array of video summaries
- has_more: Boolean indicating if more results exist
- last: The ID of the last video (use this as 'after' for next page)

Pagination example:
1. page1 = list_videos(limit=20) -> get page1.last
2. page2 = list_videos(limit=20, after=page1.last)
3. Continue until has_more=false"""

DELETE_VIDEO = """Permanently delete a video from OpenAI's cloud storage.

WARNING: This is permanent and cannot be undone! The video will be deleted from OpenAI's servers.
This does NOT delete any local files you may have downloaded with download_video.

Use this to:
- Clean up test videos
- Remove unwanted content
- Free up storage quota

Parameters:
- video_id: The ID of the video to delete (required)

Returns confirmation with the deleted video_id and deleted=true."""

REMIX_VIDEO = """Create a NEW video by remixing an existing completed video with a different prompt.

This creates a brand new video generation job (with a new video_id) based on an existing video.
The original video must have status='completed' for remix to work.

Like create_video, this returns immediately with a new video_id - the remix is NOT instant.
You must poll the NEW video_id with get_video_status until it completes.

Parameters:
- previous_video_id: ID of the completed video to use as a base (required)
- prompt: New text prompt to guide the remix (required)

Returns a NEW Video object with a different video_id, status='queued', progress=0.

Typical workflow:
1. Create original: create_video("a cat") -> video_id_1
2. Wait: Poll get_video_status(video_id_1) until completed
3. Remix: remix_video(video_id_1, "a dog") -> video_id_2 (NEW ID!)
4. Wait: Poll get_video_status(video_id_2) until completed
5. Download: download_video(video_id_2)"""


# ==================== REFERENCE IMAGE TOOL DESCRIPTIONS ====================

LIST_REFERENCE_IMAGES = """Search and list reference images available for video generation.

Use this to discover what reference images are available in the SORA_REFERENCE_PATH directory.
These images can be used with create_video's input_reference_filename parameter.

The reference image must match your target video size:
- "720x1280" or "1280x720" videos -> use 720x1280 or 1280x720 images
- "1024x1792" or "1792x1024" videos -> use 1024x1792 or 1792x1024 images

Parameters:
- pattern: Glob pattern to filter filenames (e.g., "cat*.png", "*.jpg"). Default: all files
- file_type: Filter by type: "jpeg", "png", "webp", or "all". Default: "all"
- sort_by: Sort results by "name", "size", or "modified". Default: "modified"
- order: "desc" for newest/largest/Z-A first, "asc" for oldest/smallest/A-Z. Default: "desc"
- limit: Max results to return. Default: 50

Returns list of ReferenceImage objects with: filename, size_bytes, modified_timestamp, file_type.

Example workflow:
1. list_reference_images(pattern="dog*", file_type="png") -> find dog images
2. Choose "dog_1280x720.png" from results
3. create_video(prompt="...", size="1280x720", input_reference_filename="dog_1280x720.png")"""

PREPARE_REFERENCE_IMAGE = """Automatically resize a reference image to match Sora's required dimensions.

This tool prepares images for use with create_video by resizing them to exact Sora dimensions.
The original image is preserved; a new resized copy is created.

Parameters:
- input_filename: Source image filename in SORA_REFERENCE_PATH (required)
- target_size: Target video size: "720x1280", "1280x720", "1024x1792", or "1792x1024" (required)
- output_filename: Custom output filename (optional, defaults to "{original_name}_{width}x{height}.png")
- resize_mode: How to handle aspect ratio (default: "crop")
  * "crop": Scale to cover target, center crop excess (no distortion, may lose edges)
  * "pad": Scale to fit inside target, add black bars (no distortion, preserves full image)
  * "rescale": Stretch/squash to exact dimensions (may distort, no cropping/padding)

Returns PrepareResult with: output_filename, original_size, target_size, resize_mode, path

Example workflow:
1. list_reference_images() -> find "photo.jpg"
2. prepare_reference_image("photo.jpg", "1280x720", resize_mode="crop") -> "photo_1280x720.png"
3. create_video(prompt="...", size="1280x720", input_reference_filename="photo_1280x720.png")"""


# ==================== IMAGE GENERATION TOOL DESCRIPTIONS ====================

CREATE_IMAGE = """Generate an image using OpenAI's Responses API and save to reference path.

Creates an image from a text prompt using GPT-5 or newer models with image generation capability.
Returns immediately with a response_id - use get_image_status() to poll for completion.

The image is saved to SORA_REFERENCE_PATH and can be used with create_video.

Parameters:
- prompt: Text description of the image to generate (required)
- model: Model to use - "gpt-5", "gpt-4.1", etc. Default: "gpt-5"
- size: Resolution - "1024x1024", "1024x1536", "1536x1024", or "auto". Default: "auto"
- quality: "low", "medium", "high", "auto". Default: "high"
- output_format: "png", "jpeg", "webp". Default: "png"
- background: "transparent", "opaque", "auto". Default: "auto"
- previous_response_id: Optional response ID to refine previous image

Returns ImageResponse with: id (response_id), status, created_at

Typical workflow:
1. create_image(prompt="sunset over mountains") -> response_id
2. get_image_status(response_id) -> poll until status='completed'
3. download_image(response_id, filename="sunset.png") -> saves to reference path
4. create_video(..., input_reference_filename="sunset.png")

Iterative refinement:
1. resp1 = create_image(prompt="a cat") -> response_id_1
2. Wait for completion
3. resp2 = create_image(prompt="make it more realistic", previous_response_id=response_id_1) -> response_id_2
4. Download response_id_2"""

GET_IMAGE_STATUS = """Check status and progress of image generation.

Use this to poll for completion after calling create_image.
Call repeatedly until status changes from 'queued'/'in_progress' to 'completed' or 'failed'.

Returns ImageResponse with: id, status, created_at"""

DOWNLOAD_IMAGE = """Download a completed generated image to reference path.

IMPORTANT: Only call AFTER get_image_status shows status='completed'.

The image is saved to SORA_REFERENCE_PATH and can immediately be used with create_video.

Parameters:
- response_id: The response ID from create_image (required)
- filename: Custom filename (optional, auto-generates if not provided)

Returns ImageDownloadResult with: filename, path, size, format"""
