# SPDX-License-Identifier: MIT
"""Tool descriptions for MCP server. Optimized for token efficiency."""

# ==================== VIDEO TOOL DESCRIPTIONS ====================

CREATE_VIDEO = """Create Sora video. Returns video_id (async). Poll get_video_status() until completed, then download_video().

Params: prompt, model (sora-2 faster|sora-2-pro quality), seconds ("4"|"8"|"12"), size (720x1280|1280x720|1024x1792|1792x1024), input_reference_filename (must match size)

Example: create_video("cat walking", model="sora-2", seconds="8", size="1280x720")"""

GET_VIDEO_STATUS = """Poll video generation status. Call repeatedly until status='completed' or 'failed'.

Returns: id, status (queued|in_progress|completed|failed), progress (0-100), model, seconds, size"""

DOWNLOAD_VIDEO = """Download completed video to VIDEO_PATH. Only call after status='completed'.

Params: video_id, filename (optional), variant (video|thumbnail|spritesheet)

Example: download_video(video_id, filename="my_video.mp4")"""

LIST_VIDEOS = """List all video jobs with pagination.

Params: limit (max 100), after (pagination cursor), order (desc|asc)

Returns: data (array), has_more, last (use as 'after' for next page)"""

DELETE_VIDEO = """Permanently delete video from OpenAI cloud. Cannot be undone. Does not delete local downloads.

Params: video_id"""

REMIX_VIDEO = """Create new video by remixing completed video with different prompt. Returns new video_id (async).

Params: previous_video_id (must be completed), prompt

Example: remix_video(video_id_1, "change to a dog") -> new video_id_2"""


# ==================== REFERENCE IMAGE TOOL DESCRIPTIONS ====================

LIST_REFERENCE_IMAGES = """List reference images in IMAGE_PATH for use with create_video.

Note: Reference image must match target video size (720x1280, 1280x720, 1024x1792, 1792x1024).

Params: pattern (glob), file_type (jpeg|png|webp|all), sort_by (name|size|modified), order, limit

Example: list_reference_images(pattern="dog*", file_type="png")"""

PREPARE_REFERENCE_IMAGE = """Resize image to exact Sora dimensions. Original preserved, creates new file.

Params: input_filename, target_size (720x1280|1280x720|1024x1792|1792x1024), output_filename (optional), resize_mode (crop|pad|rescale)

Resize modes: crop (cover+center crop), pad (fit+black bars), rescale (stretch to fit)

Example: prepare_reference_image("photo.jpg", "1280x720", resize_mode="crop")"""


# ==================== IMAGE GENERATION TOOL DESCRIPTIONS ====================

CREATE_IMAGE = """Generate/edit images via Responses API. Async - poll get_image_status() until completed.

For sync generation with gpt-image-1.5, use generate_image instead.

Params: prompt, tool_config (ImageGeneration config), previous_response_id (iterative), input_images (editing), mask_filename (inpainting)

tool_config example: {"type": "image_generation", "model": "gpt-image-1.5", "quality": "high"}

Example: create_image("sunset", tool_config={"type": "image_generation", "model": "gpt-image-1.5"})"""

GET_IMAGE_STATUS = """Poll image generation status. Call until status='completed' or 'failed'.

Returns: id, status (queued|in_progress|completed|failed), created_at"""

DOWNLOAD_IMAGE = """Download completed image to IMAGE_PATH. Only call after status='completed'.

Params: response_id, filename (optional)

Example: download_image(response_id, filename="result.png")"""


# ==================== IMAGES API TOOL DESCRIPTIONS ====================

GENERATE_IMAGE = """Generate images via Images API. Sync - returns immediately with saved file. Recommended for new generation.

Models: gpt-image-1.5 (best), gpt-image-1, gpt-image-1-mini, dall-e-3, dall-e-2

Params: prompt, model, size (1024x1024|1536x1024|1024x1536|auto), quality (low|medium|high|auto), background (transparent|opaque|auto), output_format (png|jpeg|webp), moderation, filename

Example: generate_image("mountain sunset", model="gpt-image-1.5", size="1536x1024", quality="high")"""

EDIT_IMAGE = """Edit existing images via Images API. Sync - returns immediately. Supports up to 16 input images.

Models: gpt-image-1.5 (best), gpt-image-1, gpt-image-1-mini

Params: prompt, input_images (list of filenames), model, mask_filename (inpainting), size, quality, background, output_format, input_fidelity (high|low for face/style), filename

Example: edit_image("add a hat", input_images=["person.png"])"""


# ==================== AUDIO TOOL DESCRIPTIONS ====================

LIST_AUDIO_FILES = """List audio files in AUDIO_PATH with filtering and sorting.

Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm (transcription); mp3, wav (chat)

Params: pattern (regex), min/max_size_bytes, min/max_duration_seconds, min/max_modified_time, format, sort_by (name|size|duration|modified_time|format), reverse

Example: list_audio_files(format="mp3", sort_by="modified_time", reverse=true)"""

GET_LATEST_AUDIO = """Get most recently modified audio file with metadata and model support info.

Example: get_latest_audio() -> file metadata"""

CONVERT_AUDIO = """Convert audio to GPT-4o chat compatible formats. Chat only supports mp3/wav; transcription supports more.

Params: input_path, output_format (mp3|wav)

Example: convert_audio("recording.flac", output_format="mp3")"""

COMPRESS_AUDIO = """Compress audio exceeding 25MB API limit. Only use if file too large or user requests.

Params: input_path, max_mb (default 25)

Example: compress_audio("large_file.wav", max_mb=25)"""

TRANSCRIBE_AUDIO = """Transcribe audio to text via Whisper/GPT-4o.

Models: gpt-4o-mini-transcribe (fast, recommended), gpt-4o-transcribe (accurate), whisper-1
Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm

Params: file_path, model, response_format (text|json|verbose_json|srt|vtt), prompt (context hint), timestamp_granularities (word|segment)

Example: transcribe_audio("meeting.mp3", response_format="srt")"""

CHAT_WITH_AUDIO = """Analyze audio with GPT-4o - ask questions, summarize, analyze tone. Unlike transcribe, enables conversation about content.

Formats: mp3, wav only (use convert_audio for others)
Models: gpt-4o-audio-preview (best), gpt-4o-mini-audio-preview (faster)

Params: file_path, model, system_prompt, user_prompt

Example: chat_with_audio("meeting.mp3", user_prompt="Summarize key decisions")"""

TRANSCRIBE_WITH_ENHANCEMENT = """Enhanced transcription with AI templates for context/analysis.

Enhancements: detailed (tone/emotion/sounds), storytelling (narrative), professional (formal), analytical (key points/structure)

Params: file_path, enhancement_type, model, response_format

Example: transcribe_with_enhancement("meeting.mp3", enhancement_type="professional")"""

CREATE_AUDIO = """Text-to-speech via OpenAI TTS. Auto-chunks long text.

Voices: alloy (neutral), ash (authoritative), coral (energetic), nova (friendly), onyx (professional), sage (calm), shimmer (polished), echo (deep), ballad (warm), fable (storytelling)

Params: text_prompt, voice, model (gpt-4o-mini-tts), instructions (style), speed (0.25-4.0), output_filename

Example: create_audio("Hello world", voice="nova", speed=1.2)"""
