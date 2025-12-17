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

Returns PrepareResult with: output_filename, original_size, target_size, resize_mode, path

Example workflow:
1. list_reference_images() -> find "photo.jpg"
2. prepare_reference_image("photo.jpg", "1280x720", resize_mode="crop") -> "photo_1280x720.png"
3. create_video(prompt="...", size="1280x720", input_reference_filename="photo_1280x720.png")"""


# ==================== IMAGE GENERATION TOOL DESCRIPTIONS ====================

CREATE_IMAGE = """Generate or edit images using OpenAI's Responses API.

Creates images from text prompts OR edits existing images by providing reference images.
Returns immediately with a response_id - use get_image_status() to poll for completion.

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
  * Supports all fields: model, size, quality, moderation, input_fidelity, etc.
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

**Image generation models (tool_config.model):**
- gpt-image-1.5: STATE-OF-THE-ART - Best quality, better instruction following, improved text rendering (RECOMMENDED)
- gpt-image-1: High quality image generation
- gpt-image-1-mini: Fast, cost-effective generation

Common tool_config examples:

Best quality with GPT Image 1.5:
  tool_config={"type": "image_generation", "model": "gpt-image-1.5"}

Fast generation with mini model:
  tool_config={"type": "image_generation", "model": "gpt-image-1-mini"}

Lower content moderation:
  tool_config={"type": "image_generation", "moderation": "low"}

High-fidelity with custom settings:
  tool_config={
      "type": "image_generation",
      "model": "gpt-image-1.5",
      "quality": "high",
      "input_fidelity": "high",
      "size": "1536x1024"
  }

Workflows:

1. Text-only generation:
   create_image("sunset over mountains")

2. Best quality generation with GPT Image 1.5:
   create_image("sunset over mountains", tool_config={"type": "image_generation", "model": "gpt-image-1.5"})

3. Single image editing:
   create_image("add a flamingo to the pool", input_images=["lounge.png"])

4. Multi-image composition:
   create_image("gift basket with all these items", input_images=["lotion.png", "soap.png", "bomb.jpg"])

5. High-fidelity logo placement:
   create_image(
       "add logo to woman's shirt",
       input_images=["woman.jpg", "logo.png"],
       tool_config={"type": "image_generation", "input_fidelity": "high"}
   )

6. Masked inpainting:
   create_image("add flamingo", input_images=["pool.png"], mask_filename="pool_mask.png")

7. Fast generation with mini model:
   create_image("quick sketch of a cat", tool_config={"type": "image_generation", "model": "gpt-image-1-mini"})

Returns ImageResponse with: id, status, created_at"""

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

Returns ImageDownloadResult with: filename, path, size, format"""


# ==================== IMAGES API TOOL DESCRIPTIONS ====================

GENERATE_IMAGE = """Generate images using OpenAI's Images API with gpt-image-1.5.

RECOMMENDED for new image generation - uses the latest gpt-image-1.5 model.
Returns immediately with the generated image (no polling required).

Unlike create_image (which uses Responses API), this tool:
- Supports gpt-image-1.5 (state-of-the-art, 20% cheaper)
- Returns synchronously (no polling needed)
- Provides token usage for cost tracking
- Does NOT support iterative refinement (use create_image for that)

Parameters:
- prompt: Text description of the image (required, max 32k chars)
- model: Image model to use. Default: "gpt-image-1.5"
  * gpt-image-1.5: STATE-OF-THE-ART (recommended) - best quality, improved text rendering
  * gpt-image-1: High quality
  * gpt-image-1-mini: Fast, cost-effective
  * dall-e-3: Legacy DALL-E 3
  * dall-e-2: Legacy DALL-E 2
- size: Image dimensions. Default: "auto"
  * "auto", "1024x1024", "1536x1024" (landscape), "1024x1536" (portrait)
- quality: Generation quality. Default: "auto"
  * "auto", "low", "medium", "high"
- background: Background type. Default: "auto"
  * "auto", "transparent", "opaque"
- output_format: Output format. Default: "png"
  * "png", "jpeg", "webp"
- moderation: Content moderation. Default: "auto"
  * "auto", "low"
- filename: Custom output filename (optional)

Returns ImageGenerateResult with: filename, path, size, format, model, usage

Example workflows:

1. Basic generation with gpt-image-1.5:
   generate_image("a sunset over mountains")

2. High quality portrait:
   generate_image("professional headshot", size="1024x1536", quality="high")

3. Transparent background:
   generate_image("product icon", background="transparent", output_format="png")

4. Fast generation with mini model:
   generate_image("quick sketch", model="gpt-image-1-mini")"""

EDIT_IMAGE = """Edit images using OpenAI's Images API with gpt-image-1.5.

Modify existing images based on a prompt. Supports up to 16 input images.
Returns immediately with the edited image (no polling required).

Parameters:
- prompt: Text description of desired edits (required, max 32k chars)
- input_images: List of image filenames from IMAGE_PATH (required, max 16 images)
- model: Image model. Default: "gpt-image-1.5"
- mask_filename: PNG mask with alpha channel for inpainting (optional)
  * Transparent areas = edit these regions
  * Opaque areas = preserve original
- size: Output dimensions. Default: "auto"
- quality: Generation quality. Default: "auto"
- background: Background type. Default: "auto"
- output_format: Output format. Default: "png"
- input_fidelity: Control fidelity to input (gpt-image-1 only). Default: None
  * "high" - better face/style preservation
  * "low" - more creative freedom
- filename: Custom output filename (optional)

Returns ImageGenerateResult with: filename, path, size, format, model, usage

Example workflows:

1. Simple edit:
   edit_image("add a hat", input_images=["person.png"])

2. Multi-image composition:
   edit_image("combine into collage", input_images=["photo1.jpg", "photo2.jpg", "photo3.jpg"])

3. Inpainting with mask:
   edit_image("add flamingo", input_images=["pool.png"], mask_filename="pool_mask.png")

4. High-fidelity face edit:
   edit_image("change hair color to red", input_images=["portrait.jpg"], input_fidelity="high")"""


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
- output_file_name: Optional custom name for output file (defaults to input name with new extension)

Returns AudioProcessingResult with:
- output_file_name: Name of the converted file
- success: Boolean indicating if conversion succeeded
- message: Description of the operation

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
- output_file_name: Optional custom name for compressed file (defaults to input name with _compressed suffix)

Returns AudioProcessingResult with:
- output_file_name: Name of compressed file (or original if no compression needed)
- success: Boolean indicating if compression succeeded
- message: Description of operation (e.g., "File already under 25MB" or "Compressed from 45MB to 24MB")

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

CREATE_AUDIO = """Generate text-to-speech audio using OpenAI TTS API.

Convert text to natural-sounding speech with multiple voice options and customization.
Handles texts of any length by automatically splitting into 4096-character chunks
and concatenating the audio seamlessly.

Model recommendation:
- gpt-4o-mini-tts: RECOMMENDED - High quality, fast, cost-effective (DEFAULT)

Available voices (each with distinct characteristics):
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

Parameters:
- text_prompt: Text to convert to speech (required, any length)
- model: TTS model. Default: "gpt-4o-mini-tts"
- voice: Voice to use. Default: "alloy"
- instructions: Optional style guidance (e.g., "Speak slowly and clearly", "Use British accent", "Sound excited")
- speed: Speech speed from 0.25 (very slow) to 4.0 (very fast). Default: 1.0
- output_file_name: Optional custom filename (defaults to "speech_<timestamp>.mp3")

Returns TTSResult with:
- output_file_name: Name of the generated audio file
- success: Boolean indicating if generation succeeded
- duration_seconds: Length of generated audio (if available)

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
   create_audio("Welcome message", voice="shimmer", output_file_name="welcome.mp3")"""
