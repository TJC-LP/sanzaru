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
- input_reference_filename: Filename of reference image in IMAGE_PATH (e.g., "cat.png"). Use list_reference_images to find available images. Image must match target size. Supported: JPEG, PNG, WEBP. Optional.

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

The video is automatically saved to the directory configured in VIDEO_PATH.
Returns the filename of the downloaded file.

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
3. Download: download_video(video_id, filename="my_video.mp4") -> returns filename

Returns DownloadResult with: filename, variant"""

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

LIST_LOCAL_VIDEOS = """List locally downloaded video files with filtering and sorting.

Use this to discover what video files have been downloaded to the VIDEO_PATH directory.
These are videos previously downloaded with download_video.

Parameters:
- pattern: Glob pattern to filter filenames (e.g., "sora*.mp4", "*.webm"). Default: all files
- file_type: Filter by type: "mp4", "webm", "mov", or "all". Default: "all"
- sort_by: Sort results by "name", "size", or "modified". Default: "modified"
- order: "desc" for newest/largest/Z-A first, "asc" for oldest/smallest/A-Z. Default: "desc"
- limit: Max results to return. Default: 50

Returns list of VideoFile objects with: filename, size_bytes, modified_timestamp, file_type.

Example workflow:
1. list_local_videos(file_type="mp4") -> find downloaded MP4 videos
2. view_media(media_type="video", filename="my_video.mp4") -> watch it"""

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

Use this to discover what reference images are available in the IMAGE_PATH directory.
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
- input_filename: Source image filename in IMAGE_PATH (required)
- target_size: Target video size: "720x1280", "1280x720", "1024x1792", or "1792x1024" (required)
- output_filename: Custom output filename (optional, defaults to "{original_name}_{width}x{height}.png")
- resize_mode: How to handle aspect ratio (default: "crop")
  * "crop": Scale to cover target, center crop excess (no distortion, may lose edges)
  * "pad": Scale to fit inside target, add black bars (no distortion, preserves full image)
  * "rescale": Stretch/squash to exact dimensions (may distort, no cropping/padding)

Returns PrepareResult with: output_filename, original_size, target_size, resize_mode

Example workflow:
1. list_reference_images() -> find "photo.jpg"
2. prepare_reference_image("photo.jpg", "1280x720", resize_mode="crop") -> "photo_1280x720.png"
3. create_video(prompt="...", size="1280x720", input_reference_filename="photo_1280x720.png")"""


# ==================== IMAGE GENERATION TOOL DESCRIPTIONS ====================

# Shared fragments referenced by CREATE_IMAGE, GENERATE_IMAGE, and EDIT_IMAGE
# below to avoid repeating the same model list and size guidance three times.

_IMAGE_MODELS_SUMMARY = """**Image generation models:**
- gpt-image-2: STATE-OF-THE-ART (default) — best quality, ~99% text accuracy.
  Always processes inputs at high fidelity (no input_fidelity knob).
  Does not support transparent backgrounds.
- gpt-image-1.5: Previous gen — use when you need transparent backgrounds or
  explicit input_fidelity control.
- gpt-image-1: High quality (legacy).
- gpt-image-1-mini: Fast, cost-effective drafts."""

_IMAGE_SIZE_GUIDANCE = """**Sizes (gpt-image-2):**
Recommended (reliable results): "1024x1024", "1024x1536" / "1536x1024"
(portrait / landscape), "2048x2048", "2560x1440" / "1440x2560" (2K — OpenAI's
upper reliability ceiling).
Experimental: "3840x2160" / "2160x3840" (4K). Results more variable and the
client memory / decode cost climbs fast. Prefer 2K or smaller unless the user
explicitly asks for 4K.

gpt-image-2 also accepts any resolution that satisfies all of: max edge
≤ 3840px, both edges multiples of 16, long:short ratio ≤ 3:1, and
655,360 ≤ total pixels ≤ 8,294,400. Pass "auto" to let the model pick."""


CREATE_IMAGE = (
    """Non-blocking async image generation (gpt-image-2 by default).

Creates images from text prompts OR edits existing images by providing reference images.
Returns immediately with a response_id - use get_image_status() to poll for completion.
Supports iterative refinement via previous_response_id.

**Best for:** parallel generation (multiple images at once), iterative refinement chains,
and workflows where you need to do other work while images generate.
For simple one-shot generation, generate_image is simpler (no polling needed).

**Text-only generation (no input_images):**
- Generates image from scratch based on prompt

**Image editing (with input_images):**
- Modifies existing images based on prompt
- Combines multiple images into new composition
- First image receives highest detail preservation
- Prompt describes desired changes, not what's already in images

Parameters:
- prompt: Text description (required)
  * Without input_images: Describe what to generate
  * With input_images: Describe what changes to make
- model: Mainline model - "gpt-5.2" (default), "gpt-5.1", "gpt-5", etc.
- tool_config: Optional ImageGeneration configuration object (optional)
  * Image model defaults to gpt-image-2 when "model" is omitted
  * Supports all fields: model, size, quality, moderation, input_fidelity, action, etc.
  * MCP library handles serialization automatically
  * See examples below for common configurations
- previous_response_id: Refine previous image iteratively (optional)
- input_images: List of filenames from IMAGE_PATH (optional)
  * Example: ["cat.png"] or ["lotion.jpg", "soap.png", "bomb.jpg"]
  * Use list_reference_images() to discover available images
  * Supported formats: JPEG, PNG, WEBP
- mask_filename: PNG with alpha channel for inpainting (optional)
  * Defines which region of first input image to edit
  * Transparent = edit this area, black = keep original
  * Requires input_images parameter

"""
    + _IMAGE_MODELS_SUMMARY
    + "\n\n"
    + _IMAGE_SIZE_GUIDANCE
    + """

**Other tool_config fields (all optional):**
- quality: "low", "medium", "high", or "auto"
- moderation: "auto" (default) or "low"
- background: "auto", "opaque", or "transparent" (transparent NOT supported on gpt-image-2 — raises; use gpt-image-1.5)
- input_fidelity: "high" or "low" (gpt-image-1/1.5 only — ignored by gpt-image-2)
- output_format: "png", "jpeg", or "webp"
- action: "auto" (default), "generate", or "edit" — force a mode when an image is in context
- partial_images: 0-3 — stream partial images during generation

Common tool_config examples:

Best quality with gpt-image-2:
  tool_config={"type": "image_generation", "model": "gpt-image-2", "quality": "high"}

2K landscape:
  tool_config={"type": "image_generation", "model": "gpt-image-2", "size": "2560x1440"}

Transparent PNG (falls back to gpt-image-1.5):
  tool_config={"type": "image_generation", "model": "gpt-image-1.5", "background": "transparent"}

Fast draft with mini model:
  tool_config={"type": "image_generation", "model": "gpt-image-1-mini", "quality": "low"}

Lower content moderation:
  tool_config={"type": "image_generation", "moderation": "low"}

Force a fresh image even when one is in context:
  tool_config={"type": "image_generation", "action": "generate"}

Workflows:

1. Text-only generation (recommended):
   create_image("sunset over mountains", tool_config={"type": "image_generation", "model": "gpt-image-2"})

2. Single image editing:
   create_image("add a flamingo to the pool", input_images=["lounge.png"])

3. Multi-image composition:
   create_image("gift basket with all these items", input_images=["lotion.png", "soap.png", "bomb.jpg"])

4. High-fidelity logo placement:
   create_image(
       "add logo to woman's shirt",
       input_images=["woman.jpg", "logo.png"],
       tool_config={"type": "image_generation", "input_fidelity": "high"}
   )

5. Masked inpainting:
   create_image("add flamingo", input_images=["pool.png"], mask_filename="pool_mask.png")

6. Fast generation with mini model:
   create_image("quick sketch of a cat", tool_config={"type": "image_generation", "model": "gpt-image-1-mini"})

7. Iterative refinement:
   resp1 = create_image("a cyberpunk character")
   resp2 = create_image("add neon details", previous_response_id=resp1.id)

8. Force a fresh image even with a prior response in context:
   create_image("a new cat", previous_response_id=prev.id,
                tool_config={"type": "image_generation", "action": "generate"})

Returns ImageResponse with: id, status, created_at"""
)

GET_IMAGE_STATUS = """Check status and progress of image generation.

Use this to poll for completion after calling create_image.
Call repeatedly until status changes from 'queued'/'in_progress' to 'completed' or 'failed'.

Returns ImageResponse with: id, status, created_at"""

DOWNLOAD_IMAGE = """Download a completed generated image to reference path.

IMPORTANT: Only call AFTER get_image_status shows status='completed'.

The image is saved to IMAGE_PATH and can immediately be used with create_video.

Parameters:
- response_id: The response ID from create_image (required)
- filename: Custom filename (optional, auto-generates if not provided)

Returns ImageDownloadResult with: filename, size, format"""


# ==================== IMAGES API TOOL DESCRIPTIONS ====================

GENERATE_IMAGE = (
    """RECOMMENDED default image generation tool. Synchronous — returns the finished image directly.

No polling needed. Blocks until the image is ready and saves it to disk in one step.
Provides token usage for cost tracking.

For parallel generation (multiple images at once) or iterative refinement chains,
use create_image instead (async with previous_response_id support).

Parameters:
- prompt: Text description of the image (required, max 32k chars)
- model: Image model to use. Default: "gpt-image-2". Also accepts legacy
  "dall-e-3" and "dall-e-2". See model summary below.
- size: Image dimensions. Default: "auto". See size guidance below.
- quality: Generation quality. Default: "auto"
  * "auto", "low", "medium", "high"
- background: Background type. Default: "auto"
  * "auto", "transparent", "opaque"
  * NOTE: gpt-image-2 does NOT support transparent — use gpt-image-1.5 for that
- output_format: Output format. Default: "png"
  * "png", "jpeg", "webp"
- moderation: Content moderation. Default: "auto"
  * "auto", "low"
- filename: Custom output filename (optional)

"""
    + _IMAGE_MODELS_SUMMARY
    + "\n\n"
    + _IMAGE_SIZE_GUIDANCE
    + """

Returns ImageGenerateResult with: filename, size, format, model, usage

Example workflows:

1. Basic generation (gpt-image-2):
   generate_image("a sunset over mountains")

2. High quality 2K landscape:
   generate_image("mountain vista at golden hour", size="2560x1440", quality="high")

3. Transparent background (falls back to gpt-image-1.5):
   generate_image("product icon", model="gpt-image-1.5", background="transparent")

4. Fast draft:
   generate_image("quick sketch", quality="low")"""
)

EDIT_IMAGE = (
    """Edit images using OpenAI's Images API with gpt-image-2 (default).

Modify existing images based on a prompt. Supports up to 16 input images.
Returns immediately with the edited image (no polling required).

Parameters:
- prompt: Text description of desired edits (required, max 32k chars)
- input_images: List of image filenames from IMAGE_PATH (required, max 16 images)
- model: Image model. Default: "gpt-image-2". See model summary below.
- mask_filename: PNG mask with alpha channel for inpainting (optional)
  * Transparent areas = edit these regions
  * Opaque areas = preserve original
- size: Output dimensions. Default: "auto". See size guidance below.
- quality: Generation quality. Default: "auto"
- background: Background type. Default: "auto" (transparent unsupported on gpt-image-2)
- output_format: Output format. Default: "png"
- input_fidelity: Control fidelity to input (gpt-image-1/gpt-image-1.5 only).
  Silently ignored for gpt-image-2 (always high).
  * "high" - better face/style preservation
  * "low" - more creative freedom
- filename: Custom output filename (optional)

"""
    + _IMAGE_MODELS_SUMMARY
    + "\n\n"
    + _IMAGE_SIZE_GUIDANCE
    + """

Returns ImageGenerateResult with: filename, size, format, model, usage

Example workflows:

1. Simple edit (gpt-image-2):
   edit_image("add a hat", input_images=["person.png"])

2. Multi-image composition:
   edit_image("combine into collage", input_images=["photo1.jpg", "photo2.jpg", "photo3.jpg"])

3. Inpainting with mask:
   edit_image("add flamingo", input_images=["pool.png"], mask_filename="pool_mask.png")

4. High-fidelity face edit on gpt-image-1.5:
   edit_image("change hair color to red", input_images=["portrait.jpg"],
              model="gpt-image-1.5", input_fidelity="high")"""
)


# ==================== AUDIO TOOL DESCRIPTIONS ====================

LIST_AUDIO_FILES = """List, filter, and sort audio files with comprehensive filtering options.

Use this to discover available audio files for transcription, chat, or TTS workflows.
Supports advanced filtering by metadata, pattern matching, and flexible sorting.

Supported formats:
- Transcription (Whisper/GPT-4o): flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
- Chat (GPT-4o audio): mp3, wav

Parameters:
- pattern: Optional regex pattern to filter files by name (e.g., "meeting.*\\.mp3")
- min_size_bytes: Minimum file size in bytes (optional)
- max_size_bytes: Maximum file size in bytes (useful for API limits like 25MB)
- min_duration_seconds: Minimum audio duration in seconds (optional)
- max_duration_seconds: Maximum audio duration in seconds (optional)
- min_modified_time: Minimum file modification time as Unix timestamp (optional)
- max_modified_time: Maximum file modification time as Unix timestamp (optional)
- format: Filter by specific audio format (e.g., 'mp3', 'wav') (optional)
- sort_by: Sort by 'name', 'size', 'duration', 'modified_time', or 'format'. Default: 'name'
- reverse: Set to true for descending order, false for ascending. Default: false

Returns list of FilePathSupportParams with full metadata:
- file_name: Name of the file
- size_bytes: File size in bytes
- format: Audio format (mp3, wav, etc.)
- duration_seconds: Audio duration (if available)
- modified_time: Last modified timestamp
- transcription_support: List of supported transcription models (if applicable)
- chat_support: List of supported chat models (if applicable)

Example workflows:
1. Find all mp3 files: list_audio_files(format="mp3")
2. Find large files needing compression: list_audio_files(min_size_bytes=26214400)
3. Find recent recordings: list_audio_files(sort_by="modified_time", reverse=true, limit=10)
4. Find meetings: list_audio_files(pattern="meeting.*")"""

GET_LATEST_AUDIO = """Get the most recently modified audio file with model support info.

Quick way to access the latest audio recording without filtering.
Returns full metadata including supported models for transcription and chat.

Returns FilePathSupportParams with metadata for the latest audio file.

Example workflow:
1. get_latest_audio() -> latest file metadata
2. transcribe_audio(file_name=metadata.file_name)"""

CONVERT_AUDIO = """Convert audio files to GPT-4o compatible formats (mp3 or wav).

Use this when you have audio in unsupported formats (like flac, m4a, ogg) and need
to convert for GPT-4o audio chat, which only supports mp3 and wav.

Whisper transcription supports many formats, but GPT-4o audio chat is more limited.
This tool ensures compatibility.

Parameters:
- input_file_name: Name of the input audio file to convert (required)
- target_format: Target format - "mp3" (smaller, lossy) or "wav" (larger, lossless). Default: "mp3"
- output_filename: Optional custom name for output file (defaults to input name with new extension)

Returns AudioProcessingResult with:
- output_file: Name of the converted file

Example workflow:
1. list_audio_files(format="flac") -> find "recording.flac"
2. convert_audio("recording.flac", target_format="mp3") -> "recording.mp3"
3. chat_with_audio("recording.mp3")"""

COMPRESS_AUDIO = """Compress audio files that exceed API size limits.

OpenAI APIs have a 25MB file size limit. Use this tool to automatically compress
large audio files by adjusting bitrate while maintaining acceptable quality.

ONLY USE THIS IF:
- User explicitly requests compression
- Other tools fail with "file too large" errors
- File size exceeds 25MB (26,214,400 bytes)

The tool intelligently compresses only if necessary - if file is already under
the limit, it returns the original unchanged.

Parameters:
- input_file_name: Name of the input audio file to compress (required)
- max_mb: Maximum target size in MB. Default: 25 (API limit)
- output_filename: Optional custom name for compressed file (defaults to input name with _compressed suffix)

Returns AudioProcessingResult with:
- output_file: Name of compressed file (or original if no compression needed)

Example workflow:
1. list_audio_files(min_size_bytes=26214400) -> find large files
2. compress_audio("huge_recording.wav", max_mb=25) -> "huge_recording_compressed.wav"
3. transcribe_audio("huge_recording_compressed.wav")"""

TRANSCRIBE_AUDIO = """Transcribe audio using OpenAI Whisper or GPT-4o transcription models.

Standard transcription tool for converting speech to text with high accuracy.
Supports multiple models, response formats, and advanced features like timestamps.

Model recommendations:
- gpt-4o-mini-transcribe: RECOMMENDED - Fast, accurate, cost-effective (DEFAULT)
- gpt-4o-transcribe: Maximum accuracy, slower, more expensive
- whisper-1: Original Whisper model, rarely needed but available

Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm

Parameters:
- input_file_name: Name of the audio file to transcribe (required)
- model: Transcription model. Default: "gpt-4o-mini-transcribe"
- response_format: Output format. Default: "text"
  * "text": Plain text transcription
  * "json": JSON with text
  * "verbose_json": JSON with text, language, duration, timestamps
  * "srt": SubRip subtitle format
  * "vtt": WebVTT subtitle format
- prompt: Optional hint to guide transcription (speaker names, context, jargon, etc.)
- timestamp_granularities: Optional list - ["word"] for word-level, ["segment"] for sentence-level, or both

Returns TranscriptionResult with:
- text: Transcribed text
- format: Response format used
- model: Model used
- duration_seconds: Audio duration (if available)

Example workflows:
1. Basic transcription:
   transcribe_audio("meeting.mp3")

2. With speaker context:
   transcribe_audio("interview.wav", prompt="Speakers: Alice (interviewer), Bob (CEO)")

3. Generate subtitles:
   transcribe_audio("video.mp3", response_format="srt")

4. Word-level timestamps:
   transcribe_audio("speech.wav", response_format="verbose_json", timestamp_granularities=["word"])"""

CHAT_WITH_AUDIO = """Interactive audio analysis using GPT-4o audio understanding models.

Have a conversation about audio content. Unlike transcription which just converts
speech to text, this tool lets you ask questions, analyze tone, extract insights,
summarize, or discuss the audio content.

Supported formats: mp3, wav only (use convert_audio for other formats)

Model recommendations:
- gpt-4o-audio-preview: RECOMMENDED - Best audio understanding (DEFAULT)
- gpt-4o-mini-audio-preview: Faster but limited audio processing capabilities

Parameters:
- input_file_name: Name of the audio file to analyze (required)
- model: Audio chat model. Default: "gpt-4o-audio-preview"
- system_prompt: Optional system context (e.g., "You are analyzing a medical interview")
- user_prompt: Optional question or instruction (e.g., "What are the main topics discussed?")

Returns ChatResult with:
- response_text: GPT-4o's analysis or response
- model: Model used

Example workflows:
1. Summarize audio:
   chat_with_audio("lecture.mp3", user_prompt="Summarize the main points in 3 bullet points")

2. Analyze tone:
   chat_with_audio("call.wav", user_prompt="Analyze the emotional tone and sentiment")

3. Extract information:
   chat_with_audio("interview.mp3", user_prompt="List all action items mentioned")

4. Question answering:
   chat_with_audio("meeting.wav", user_prompt="What decisions were made about the product launch?")"""

TRANSCRIBE_WITH_ENHANCEMENT = """Enhanced transcription with specialized AI-powered templates.

Provides transcription with additional AI enhancements beyond basic speech-to-text.
Uses specialized prompts to add context, analysis, or formatting to the transcription.

Enhancement types:
- detailed: Includes tone, emotion, background sounds, pauses, emphasis
  Example: "Said with frustration. [Background: traffic noise]. Long pause."

- storytelling: Transforms transcript into narrative form with descriptive language
  Example: "The speaker passionately described..."

- professional: Formal, business-appropriate formatting with proper grammar
  Example: Converts casual speech to polished business writing

- analytical: Adds analysis of speech patterns, key points, structure, themes
  Example: Includes notes like "[Key Point]", "[Repetition for emphasis]"

Parameters:
- input_file_name: Name of the audio file to transcribe (required)
- enhancement_type: Type of enhancement. Default: "detailed"
- model: Transcription model. Default: "gpt-4o-mini-transcribe"
- response_format: Output format. Default: "text"
- timestamp_granularities: Optional timestamp granularities (optional)

Returns TranscriptionResult with enhanced transcription.

Example workflows:
1. Detailed meeting notes:
   transcribe_with_enhancement("meeting.mp3", enhancement_type="detailed")

2. Convert presentation to narrative:
   transcribe_with_enhancement("talk.wav", enhancement_type="storytelling")

3. Professional documentation:
   transcribe_with_enhancement("interview.mp3", enhancement_type="professional")

4. Speech analysis:
   transcribe_with_enhancement("speech.wav", enhancement_type="analytical")"""

CREATE_AUDIO = """Generate text-to-speech audio using OpenAI or ElevenLabs.

Convert text to natural-sounding speech with multiple voice options and customization.
Handles texts of any length by automatically splitting into chunks at natural boundaries
and concatenating the audio seamlessly (the per-request limit is provider-specific).

Providers:
- openai (DEFAULT): named voices, `instructions` style control, speed 0.25-4.0
- elevenlabs: voice library + `voice_settings` tuning, speed 0.7-1.2, needs ELEVENLABS_API_KEY
  and the [elevenlabs] extra

Model recommendation:
- gpt-4o-mini-tts: RECOMMENDED for openai - High quality, fast, cost-effective (DEFAULT)
- eleven_v3: default for elevenlabs - most expressive, supports inline audio tags,
  but does NOT support speed adjustment
- eleven_multilingual_v2: use when you need ElevenLabs speed control (0.7-1.2)
- eleven_flash_v2_5 / eleven_turbo_v2_5: lower latency and cost

Available voices (OpenAI — each with distinct characteristics):
- alloy: Neutral, balanced (good default)
- ash: Authoritative, confident
- ballad: Warm, expressive
- coral: Bright, energetic
- echo: Deep, resonant
- fable: Storytelling, engaging
- nova: Youthful, friendly
- onyx: Professional, clear
- sage: Calm, soothing
- shimmer: Smooth, polished

ElevenLabs voices are opaque voice ids from your own voice library (e.g. "21m00Tcm4TlvDq8ikWAM"),
not names from the list above.

Parameters:
- text_prompt: Text to convert to speech (required, any length)
- model: TTS model. Default: "gpt-4o-mini-tts" (openai) / "eleven_v3" (elevenlabs)
- voice: OpenAI voice name, or an ElevenLabs voice id. Default: "alloy" (openai); elevenlabs has
  no default and requires an explicit voice id
- instructions: Optional style guidance (e.g., "Speak slowly and clearly", "Use British accent", "Sound excited").
  OPENAI ONLY — ElevenLabs ignores it. For expressive ElevenLabs delivery, put inline audio tags
  in the text itself with eleven_v3: "[whispers] this is the secret. [laughs] Got you."
- speed: Speech speed. openai: 0.25 (very slow) to 4.0 (very fast). elevenlabs: 0.7-1.2, and
  eleven_v3 rejects any value other than 1.0. Default: 1.0
- output_filename: Optional custom filename (defaults to "speech_<timestamp>.mp3")
- provider: "openai" (default) or "elevenlabs"
- voice_settings: ELEVENLABS ONLY. Object with any of: stability (0-1), similarity_boost (0-1),
  style (0-1), use_speaker_boost (bool), speed (0.7-1.2, overrides the `speed` param).
  Rejected when provider is "openai".

Returns TTSResult with:
- output_file: Name of the generated audio file (use this exact filename in follow-up calls like view_media)

Example workflows:
1. Basic TTS:
   create_audio("Hello, welcome to our service!")

2. Custom voice and style:
   create_audio("Important announcement", voice="onyx", instructions="Speak slowly and authoritatively")

3. Fast narration:
   create_audio("Quick update", voice="nova", speed=1.3)

4. Long-form content:
   create_audio("An entire blog post with thousands of words...")  # Automatically chunks

5. Custom filename:
   create_audio("Welcome message", voice="shimmer", output_filename="welcome.mp3")

6. ElevenLabs with voice tuning:
   create_audio("[excited] You are not going to believe this.",
                provider="elevenlabs", voice="21m00Tcm4TlvDq8ikWAM",
                voice_settings={"stability": 0.4, "similarity_boost": 0.85})"""


# ==================== PODCAST GENERATION TOOL DESCRIPTIONS ====================

GENERATE_PODCAST = """Generate a multi-voice podcast from a structured PodcastScript.

Takes a fully-specified script with speaker definitions and segment content, generates
every segment via TTS in parallel (bounded per provider), and stitches them into a single
audio file with configurable silence gaps and optional loudness normalization.

Speakers choose their provider independently, so one episode can mix OpenAI and ElevenLabs
voices. Provider precedence: speaker.provider > config.provider > the `provider` argument.

All state is in-memory — no temp files are created. The final output is written directly
to the audio storage path.

**Schema:**

script must be a JSON object matching this structure:

{
  "title": string,                   // Used for output filename (required)
  "description": string,             // Optional show notes (optional)
  "speakers": [                      // 1-4 speaker definitions (required)
    {
      "id": string,                  // Unique identifier, referenced in segments (e.g. "host")
      "name": string,                // Display name (e.g. "Alex")
      "voice": string,               // openai: alloy|ash|ballad|coral|echo|fable|nova|onyx|sage|shimmer
                                     // elevenlabs: a voice id from your library (required, no default)
      "speed": number,               // openai 0.25-4.0; elevenlabs 0.7-1.2; eleven_v3 must be 1.0
      "instructions": string,        // Style directives (e.g. "Speak confidently and clearly").
                                     // OPENAI ONLY — ignored by elevenlabs speakers; use inline
                                     // audio tags like [whispers] in segment text with eleven_v3
      "role": string,                // Optional: "host"|"cohost"|"narrator"|"interviewer"|"guest"
      "provider": string,            // Optional: "openai" (default) | "elevenlabs"
      "model": string,               // Optional per-speaker model override; must belong to the
                                     // speaker's provider
      "voice_settings": {            // Optional, ELEVENLABS ONLY
        "stability": number,         //   0-1
        "similarity_boost": number,  //   0-1
        "style": number,             //   0-1
        "use_speaker_boost": boolean,
        "speed": number              //   0.7-1.2, overrides "speed" above
      }
    }
  ],
  "segments": [                      // Ordered list of spoken segments (required)
    {
      "speaker": string,             // Must match a speaker id (required)
      "text": string,                // Spoken content (required, max ~40000 chars)
      "pause_after": number,         // Silence in ms after this segment (optional, overrides default)
      "speed_override": number,      // Override speaker speed for this segment (optional; also
                                     // wins over the speaker's voice_settings "speed")
      "instruction_override": string // Override speaker instructions for this segment (optional)
    }
  ],
  "config": {                        // Global podcast settings (required)
    "default_pause_ms": number,      // Default silence between segments (required; 400-800ms recommended)
    "intro_silence_ms": number,      // Silence before first segment (optional; 500ms recommended)
    "outro_silence_ms": number,      // Silence after last segment (optional; 1000ms recommended)
    "normalize_loudness": boolean,   // Peak-normalize each segment for consistent volume (required)
    "output_format": "mp3"|"wav",    // Output format (required)
    "output_bitrate": string,        // MP3 bitrate (optional; default "192k")
    "provider": string,              // Optional episode default: "openai" | "elevenlabs"
    "max_concurrency": number,       // Optional cap on parallel TTS requests (positive int).
                                     // ElevenLabs enforces a per-tier cap (2-15, or 4-30 on
                                     // Flash/Turbo); lower this if you see HTTP 429.
    "render_mode": string,           // Optional: "segments" (default) | "dialogue"
    "dialogue_stability": number     // Optional 0-1, dialogue mode only
  }
}

**Render modes:**
- "segments" (default): one TTS request per segment, joined with your configured
  silence gaps. Full control; every segment is independent, so a single bad
  render can be retried on its own.
- "dialogue": consecutive turns that share a dialogue-capable provider and model
  (currently ElevenLabs eleven_v3) are sent as ONE request, so the model paces
  the conversation itself — turn-taking, reactions landing on the previous line.
  Noticeably more natural. Turns that cannot join a run still render per segment,
  so mixed episodes keep working: OpenAI speakers, other models, a lone turn, a
  stretch in one voice (no turn-taking to pace), a turn that alone fills the
  2000-character request budget.

  Inside a dialogue run: `pause_after` is ignored (the model owns the pacing) and
  per-speaker `voice_settings`/`speed` do not apply — the endpoint takes a single
  `config.dialogue_stability` for the whole request. Runs are split at turn
  boundaries to stay under 2000 characters per request.

  Prefer "segments" when you need exact gap control or per-speaker tuning;
  prefer "dialogue" for natural back-and-forth conversation.

**Voice guide (OpenAI):**
- ash: Authoritative, confident — good for hosts and anchors
- nova: Youthful, friendly — good for co-hosts and guests
- onyx: Deep, professional — good for narrators
- alloy: Neutral, balanced — good all-rounder
- coral: Bright, energetic — good for enthusiastic co-hosts
- fable: Warm, storytelling — good for narrative segments
- shimmer: Smooth, polished — good for intros and sign-offs

**Recommended voice pairings:**
- News/analysis: ash (host) + nova (co-host)
- Technical deep-dive: onyx (host) + coral (co-host)
- Narrative/documentary: fable (narrator) + alloy (interview subject)
- Debate: ash (moderator) + shimmer (guest A) + nova (guest B)

**Duration estimation:**
Before running, estimate duration: ~150 words/minute at 1.0x speed.
A 10-minute podcast needs ~1500 words of content.

**Parameters:**
- script: PodcastScript object (required)
- model: Default model for OpenAI speakers. Default: "gpt-4o-mini-tts". ElevenLabs speakers
  fall back to "eleven_v3" unless they set their own "model".
- provider: Episode-wide default provider, overridden by config.provider and speaker.provider.
  Default: "openai"

**Returns** PodcastResult with:
- output_file: Filename of the generated audio (use with view_media or list_audio_files)
- title: Podcast title
- segment_count: Number of segments generated
- estimated_duration_seconds: Estimated total duration
- speakers: List of speaker display names
- transcript: Full formatted transcript

**Example (minimal two-speaker podcast):**
{
  "title": "tech_talk_ep1",
  "speakers": [
    {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "Confident host"},
    {"id": "cohost", "name": "Sam", "voice": "nova", "speed": 1.05, "instructions": "Curious co-host"}
  ],
  "segments": [
    {"speaker": "host", "text": "Welcome to Tech Talk. Today we are discussing functional programming."},
    {"speaker": "cohost", "text": "I am really excited about this one. Haskell in production — is it real?", "pause_after": 1000},
    {"speaker": "host", "text": "It absolutely is. Let us break down why."}
  ],
  "config": {
    "default_pause_ms": 600,
    "intro_silence_ms": 500,
    "outro_silence_ms": 1000,
    "normalize_loudness": true,
    "output_format": "mp3",
    "output_bitrate": "192k"
  }
}

**Example (mixed-provider episode):**
{
  "title": "mixed_ep1",
  "speakers": [
    {"id": "host", "name": "Alex", "voice": "ash", "speed": 1.0, "instructions": "Confident host"},
    {"id": "guest", "name": "Sam", "voice": "21m00Tcm4TlvDq8ikWAM", "speed": 1.0,
     "instructions": "", "provider": "elevenlabs",
     "voice_settings": {"stability": 0.45, "similarity_boost": 0.85}}
  ],
  "segments": [
    {"speaker": "host", "text": "Welcome back. Today we have a special guest."},
    {"speaker": "guest", "text": "[warmly] Thanks for having me. I have been looking forward to this."}
  ],
  "config": {
    "default_pause_ms": 600,
    "normalize_loudness": true,
    "output_format": "mp3",
    "max_concurrency": 3
  }
}"""


SIMULATE_PODCAST = """Record a podcast by having N realtime voice agents actually talk to each other.

Nothing here is scripted. Each host is a `gpt-realtime` session with a persona; a producer
gives one host the floor at a time and plays that audio into the other hosts' ears, so they
respond to what was actually said, including delivery and timing. The conversation is an
OUTPUT of this tool, not an input — the transcript comes back in the result.

**Which podcast tool to use:**
- `simulate_podcast` (this one) — you have a TOPIC and want a real conversation. Highest
  quality by a wide margin, and the only option that produces genuine disagreement,
  interruption, and reaction. Costs real money (see below) and takes 1-3 minutes.
- `generate_podcast` with render_mode="dialogue" — you have a SCRIPT and want it performed
  naturally. ElevenLabs paces consecutive turns itself.
- `generate_podcast` with render_mode="segments" — you have a SCRIPT and need exact control
  over gaps, per-speaker settings, and per-segment retry.

**COST WARNING.** This is the most expensive tool in sanzaru. A 12-minute episode runs
roughly $0.40 on gpt-realtime-2.1-mini and ~$3 on gpt-realtime-2.1. ALWAYS call once with
`dry_run: true` first — it plans the episode and projects turns, duration, tokens and
dollars without recording anything or spending audio tokens. Set `max_cost_usd` on the real
call. Prefer `model: "gpt-realtime-2.1-mini"` unless the user asked for maximum quality.

**Two ways in:**
1. `premise` — a sentence about the topic. Pre-production plans the acts and invents hosts.
2. `rundown` — a full plan (see below), typically produced by a `dry_run` call or by
   `sanzaru podcast rundown`, then hand-edited.

**How an episode is recorded.** The episode is split into acts, each recorded in its own
sessions, in parallel. This is not an optimization: a Realtime WebSocket closes at 60
minutes, so a long episode cannot be one session. It also keeps cost linear and puts a
30-minute episode at 1-2 minutes of wall clock.

Because acts record in parallel, an act CANNOT hear another act. Coherence comes entirely
from what the rundown says: `prior_context` (what earlier acts covered), `upcoming` (what
later acts own, filled in automatically), and `handoff` (where to leave off).

**Schema (all fields optional unless noted):**

{
  "premise": string,               // Topic to plan from. Required unless `rundown` is given.
  "rundown": {                     // A full plan; skips pre-production
    "title": string,
    "premise": string,
    "style": string,               // One line of tone/format direction
    "hosts": [
      {"id": string, "name": string,
       "voice": string,            // realtime voice: marin|cedar|alloy|ash|ballad|coral|
                                   // echo|sage|shimmer|verse (NOT a TTS or ElevenLabs voice)
       "persona": string,          // 2-3 sentences, second person ("You are ...")
       "model": string}            // Optional per-host model override
    ],
    "acts": [
      {"id": string, "title": string, "topic": string,
       "talking_points": [string], // 3-4 concrete claims/examples, one sentence each
       "target_seconds": number,   // Soft duration budget for the act
       "max_turns": number,        // Hard turn cap; roughly target_seconds/turn_seconds + 1
       "prior_context": string,    // What earlier acts covered. Empty for act 1.
       "upcoming": string,         // What later acts own. Derived when empty.
       "handoff": string,          // Where to leave the conversation. Empty for the last act.

       // --- direct it yourself (all optional; see "You are the producer" below)
       "direction": string,        // How to play this act, in your own words
       "turn_notes": {"0": string},// Producer note per turn index, replacing the default.
                                   // "" suppresses the note for that turn entirely.
       "speaking_order": [string]} // Host ids per turn, cycled. Default: round-robin.
    ]
  },
  "title": string, "style": string,
  "hosts": [HostSpec],             // Fix the cast when planning from a premise
  "acts": number,                  // Act count when planning from a premise (default 4)
  "target_minutes": number,        // Episode length (default 12)

  "model": string,                 // Realtime model (default "gpt-realtime-2.1")
  "planner_model": string,         // Text model for pre-production (default "gpt-5.5")
  "turn_seconds": number,          // Target upper bound per turn (default 15)
  "turn_tokens": number,           // Hard per-turn output cap; 0 = derive from turn_seconds

  "max_cost_usd": number,          // Abort once spend crosses this. STRONGLY RECOMMENDED.
  "max_concurrent_sessions": number, // Realtime sessions across all acts (default 6)

  "dry_run": boolean,              // Plan and project only. Records nothing. START HERE.
  "run_id": string,                // Identify a run for resume
  "resume": boolean,               // Reuse checkpointed acts, record only what is missing
  "stems": boolean,                // Also write one time-aligned track per host
  "qc": boolean,                   // Verify the result (default true)
  "qc_retry": boolean,             // Re-record acts QC flags, once

  "output_format": "mp3"|"wav", "output_bitrate": string, "filename": string,
  "act_gap_ms": number,            // Silence between acts (default 700)
  "intro_silence_ms": number, "outro_silence_ms": number,
  "normalize_loudness": boolean    // Default false — see below
}

**You are the producer.** A producer inside the tool gives one host the floor at a time, pushes
talking points across the act, and steers the last turns to a landing. Those are *defaults* —
they exist because unsteered agents drift to 30-second turns and mutual agreement — but you will
usually write better direction than they do, and three fields on each act hand you the chair:

- `direction` — how to play this act, added to every host's instructions. "Let this one get
  heated." "No jokes, the subject is grim." "Rory should refuse to concede the point."
- `turn_notes` — `{"0": "...", "4": "..."}`, keyed by zero-based turn index, replacing the
  default note for that turn. This is the strongest lever in the tool: it is how you make turn 4
  a genuine objection rather than a topic change. Set `""` to say nothing that turn. A note on
  the last turn takes over the closing, so it becomes your job to land the act.
- `speaking_order` — `["avery", "rory", "rory", "avery"]`, cycled. Lets a host follow their own
  point before handing back, instead of strict alternation.

You cannot steer live — the tool blocks while acts record in parallel, and a round-trip per turn
would break that — so put your direction in the rundown before recording. A good loop is:
`dry_run` → read the returned rundown → rewrite `direction`/`turn_notes` → record.

**Checkpointing and resume.** Every act is written to the audio directory the moment it
finishes, alongside a JSON sidecar holding its turns and usage. If the run is interrupted or
hits `max_cost_usd`, that audio is safe. Call again with `resume: true` and the same
`run_id` (it is in every result, and in the error payload on a cost abort) to record only
the missing acts.

**QC.** Enabled by default and cheap (~$0.005/minute). Each act's rendered audio is
transcribed with `gpt-transcribe` and compared two ways: a similarity score against what the
models reported saying (catches dropped audio), and a judge model reading the transcript
against the brief (catches skipped talking points, drift, and acts repeating each other —
the characteristic failure of parallel recording). `result.qc.flagged_acts` names anything
worth a listen; `qc_retry: true` re-records just those.

**Validation.** The brief is schema-checked before anything runs, so a bad episode fails in
milliseconds rather than mid-recording. Rejected outright: duplicate act or host ids (they would
silently overwrite a checkpoint or collapse two speakers), a `speaking_order` naming a host that
is not in the episode, `turn_notes` on turns the act will never reach, ids/filenames containing
a path separator, and out-of-range budgets (`acts` 1-24, `target_minutes` <= 240,
`target_seconds` <= 2400 per act, positive `max_cost_usd`). Warned about but allowed: an unknown
voice, a model with no known price (the cost ceiling cannot fire), and an act with too few turns
to fill its duration. Read stderr — the warnings are the ones that cost you a re-record.

**Notes:**
- `normalize_loudness` defaults to FALSE here, unlike generate_podcast: acts recorded by the
  same models are already level, and per-act peak normalization can introduce jumps.
- Truncated turns are reported per act. If `truncated_turns` is high, raise `turn_tokens`.
- Memory scales with episode length (~48KB per second of audio, more with `stems`).

**Example (always do this first):**

{"premise": "why the plumbing under generative media is harder than the models",
 "acts": 3, "target_minutes": 6, "model": "gpt-realtime-2.1-mini", "dry_run": true}

**Then record, reusing the rundown the dry run returned:**

{"rundown": <rundown from the dry run>, "model": "gpt-realtime-2.1-mini",
 "max_cost_usd": 1.00, "stems": true}"""


# ==================== MEDIA VIEWER TOOL DESCRIPTIONS ====================

VIEW_MEDIA = """Open a media file in the interactive media viewer.

Opens a rich media player UI for viewing videos, listening to audio, or displaying images.
The viewer loads the file and renders it with native playback controls.

Parameters:
- media_type: Type of media to view — "video", "audio", or "image" (required)
- filename: Name of the file to open (required)
  * video files are in the videos directory
  * image files are in the images directory
  * audio files are in the audio directory

Use list_videos, list_reference_images, or list_audio_files to discover available files.

Returns metadata: filename, media_type, size_bytes, mime_type"""
