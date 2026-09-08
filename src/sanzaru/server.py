# SPDX-License-Identifier: MIT
"""sanzaru MCP Server - Unified MCP server for OpenAI multimodal APIs.

This module initializes the FastMCP server and conditionally registers tools
based on installed optional dependencies (video, audio, image).

Business logic is organized into submodules under tools/.
"""

import argparse
import importlib.resources
import os
import secrets
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from mcp.types import ToolAnnotations
from openai.types import VideoModel, VideoSeconds, VideoSize
from openai.types.responses.tool_param import ImageGeneration
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from .config import DEFAULT_IMAGE_MODEL, logger
from .dotenv_loader import load_local_dotenv
from .features import check_audio_available, check_image_available, check_video_available
from .storage.factory import get_storage
from .tools.media_viewer import MEDIA_TYPE_TO_PATH_TYPE
from .user_context import UserContext, reset_user_context, set_user_context

# Initialize FastMCP server (stateless configuration set at runtime)
mcp = FastMCP("sanzaru")

# Tool annotation presets (MCP 2025-03-26+)
READ_ONLY_OPEN = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
READ_ONLY_CLOSED = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE_OPEN = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
WRITE_OPEN_IDEMPOTENT = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
)
WRITE_CLOSED = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
# delete_video: second call 404s but final state is identical (video absent) — idempotent per MCP spec
DESTRUCTIVE_OPEN = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=True)


# ==================== VIDEO TOOLS (CONDITIONAL) ====================
if check_video_available():
    from .descriptions import (
        CREATE_VIDEO,
        DELETE_VIDEO,
        DOWNLOAD_VIDEO,
        GET_VIDEO_STATUS,
        LIST_LOCAL_VIDEOS,
        LIST_VIDEOS,
        REMIX_VIDEO,
    )
    from .tools import video

    @mcp.tool(description=CREATE_VIDEO, annotations=WRITE_OPEN)
    async def create_video(
        prompt: str,
        model: VideoModel = "sora-2",
        seconds: VideoSeconds | None = None,
        size: VideoSize | None = None,
        input_reference_filename: str | None = None,
    ):
        return await video.create_video(prompt, model, seconds, size, input_reference_filename)

    @mcp.tool(description=GET_VIDEO_STATUS, annotations=READ_ONLY_OPEN)
    async def get_video_status(video_id: str):
        return await video.get_video_status(video_id)

    @mcp.tool(description=DOWNLOAD_VIDEO, annotations=WRITE_OPEN_IDEMPOTENT)
    async def download_video(
        video_id: str,
        filename: str | None = None,
        variant: Literal["video", "thumbnail", "spritesheet"] = "video",
    ):
        return await video.download_video(video_id, filename, variant)

    @mcp.tool(description=LIST_VIDEOS, annotations=READ_ONLY_OPEN)
    async def list_videos(limit: int = 20, after: str | None = None, order: Literal["asc", "desc"] = "desc"):
        return await video.list_videos(limit, after, order)

    @mcp.tool(description=DELETE_VIDEO, annotations=DESTRUCTIVE_OPEN)
    async def delete_video(video_id: str):
        return await video.delete_video(video_id)

    @mcp.tool(description=REMIX_VIDEO, annotations=WRITE_OPEN)
    async def remix_video(previous_video_id: str, prompt: str):
        return await video.remix_video(previous_video_id, prompt)

    @mcp.tool(description=LIST_LOCAL_VIDEOS, annotations=READ_ONLY_CLOSED)
    async def list_local_videos(
        pattern: str | None = None,
        file_type: Literal["mp4", "webm", "mov", "all"] = "all",
        sort_by: Literal["name", "size", "modified"] = "modified",
        order: Literal["asc", "desc"] = "desc",
        limit: int = 50,
    ):
        return await video.list_local_videos(pattern, file_type, sort_by, order, limit)

    logger.info("Video tools registered (7 tools)")


# ==================== IMAGE TOOLS (CONDITIONAL) ====================
if check_image_available():
    from .descriptions import (
        CREATE_IMAGE,
        DOWNLOAD_IMAGE,
        EDIT_IMAGE,
        GENERATE_IMAGE,
        GET_IMAGE_STATUS,
        LIST_REFERENCE_IMAGES,
        PREPARE_REFERENCE_IMAGE,
    )
    from .tools import image, images_api, reference
    from .tools.images_api import ImageSize

    @mcp.tool(description=LIST_REFERENCE_IMAGES, annotations=READ_ONLY_CLOSED)
    async def list_reference_images(
        pattern: str | None = None,
        file_type: Literal["jpeg", "png", "webp", "all"] = "all",
        sort_by: Literal["name", "size", "modified"] = "modified",
        order: Literal["asc", "desc"] = "desc",
        limit: int = 50,
    ):
        return await reference.list_reference_images(pattern, file_type, sort_by, order, limit)

    @mcp.tool(description=PREPARE_REFERENCE_IMAGE, annotations=WRITE_CLOSED)
    async def prepare_reference_image(
        input_filename: str,
        target_size: VideoSize,
        output_filename: str | None = None,
        resize_mode: Literal["crop", "pad", "rescale"] = "crop",
    ):
        return await reference.prepare_reference_image(input_filename, target_size, output_filename, resize_mode)

    @mcp.tool(description=CREATE_IMAGE, annotations=WRITE_OPEN)
    async def create_image(
        prompt: str,
        model: str = "gpt-5.2",
        tool_config: ImageGeneration | None = None,
        previous_response_id: str | None = None,
        input_images: list[str] | None = None,
        mask_filename: str | None = None,
    ):
        return await image.create_image(prompt, model, tool_config, previous_response_id, input_images, mask_filename)

    @mcp.tool(description=GET_IMAGE_STATUS, annotations=READ_ONLY_OPEN)
    async def get_image_status(response_id: str):
        return await image.get_image_status(response_id)

    @mcp.tool(description=DOWNLOAD_IMAGE, annotations=WRITE_OPEN_IDEMPOTENT)
    async def download_image(response_id: str, filename: str | None = None):
        return await image.download_image(response_id, filename)

    # Images API tools (synchronous, blocks until done). `model` is typed as
    # str so callers can pass new models (like gpt-image-2) ahead of SDK type
    # updates. Size options live in `ImageSize` (imported above) — single
    # source of truth in `tools/images_api.py`.

    @mcp.tool(description=GENERATE_IMAGE, annotations=WRITE_OPEN)
    async def generate_image(
        prompt: str,
        model: str = DEFAULT_IMAGE_MODEL,
        size: ImageSize = "auto",
        quality: Literal["auto", "low", "medium", "high"] = "auto",
        background: Literal["auto", "transparent", "opaque"] = "auto",
        output_format: Literal["png", "jpeg", "webp"] = "png",
        moderation: Literal["auto", "low"] = "auto",
        filename: str | None = None,
    ):
        return await images_api.generate_image(
            prompt, model, size, quality, background, output_format, moderation, filename
        )

    @mcp.tool(description=EDIT_IMAGE, annotations=WRITE_OPEN)
    async def edit_image(
        prompt: str,
        input_images: list[str],
        model: str = DEFAULT_IMAGE_MODEL,
        mask_filename: str | None = None,
        size: ImageSize = "auto",
        quality: Literal["auto", "low", "medium", "high"] = "auto",
        background: Literal["auto", "transparent", "opaque"] = "auto",
        output_format: Literal["png", "jpeg", "webp"] = "png",
        input_fidelity: Literal["high", "low"] | None = None,
        filename: str | None = None,
    ):
        return await images_api.edit_image(
            prompt,
            input_images,
            model,
            mask_filename,
            size,
            quality,
            background,
            output_format,
            input_fidelity,
            filename,
        )

    logger.info("Image tools registered (7 tools)")


# ==================== AUDIO TOOLS (CONDITIONAL) ====================
if check_audio_available():
    from openai.types import AudioModel, AudioResponseFormat
    from openai.types.audio.speech_model import SpeechModel

    from .audio.constants import (
        AudioChatModel,
        ElevenLabsModel,
        EnhancementType,
        SortBy,
        TTSProviderName,
        TTSVoice,
    )
    from .audio.providers import VoiceSettingsDict
    from .audio.realtime.types import Filename
    from .descriptions import (
        CHAT_WITH_AUDIO,
        COMPRESS_AUDIO,
        CONVERT_AUDIO,
        CREATE_AUDIO,
        GENERATE_PODCAST,
        GET_LATEST_AUDIO,
        LIST_AUDIO_FILES,
        SIMULATE_PODCAST,
        TRANSCRIBE_AUDIO,
        TRANSCRIBE_WITH_ENHANCEMENT,
    )
    from .tools import audio, podcast
    from .tools import simulate_podcast as simulate

    @mcp.tool(description=LIST_AUDIO_FILES, annotations=READ_ONLY_CLOSED)
    async def list_audio_files(
        pattern: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        min_duration_seconds: float | None = None,
        max_duration_seconds: float | None = None,
        min_modified_time: float | None = None,
        max_modified_time: float | None = None,
        format: str | None = None,
        sort_by: Literal["name", "size", "duration", "modified_time", "format"] = "name",
        reverse: bool = False,
    ):
        return await audio.list_audio_files(
            pattern=pattern,
            min_size_bytes=min_size_bytes,
            max_size_bytes=max_size_bytes,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            min_modified_time=min_modified_time,
            max_modified_time=max_modified_time,
            format=format,
            sort_by=SortBy(sort_by),
            reverse=reverse,
        )

    @mcp.tool(description=GET_LATEST_AUDIO, annotations=READ_ONLY_CLOSED)
    async def get_latest_audio():
        return await audio.get_latest_audio()

    @mcp.tool(description=CONVERT_AUDIO, annotations=WRITE_CLOSED)
    async def convert_audio(input_path: str, output_format: Literal["mp3", "wav"]):
        return await audio.convert_audio(input_path, output_format)

    @mcp.tool(description=COMPRESS_AUDIO, annotations=WRITE_CLOSED)
    async def compress_audio(input_path: str, max_mb: int = 25, output_filename: str | None = None):
        return await audio.compress_audio(input_path, max_mb, output_filename)

    @mcp.tool(description=TRANSCRIBE_AUDIO, annotations=READ_ONLY_OPEN)
    async def transcribe_audio(
        file_path: str,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ):
        return await audio.transcribe_audio(file_path, model, response_format, prompt, timestamp_granularities)

    @mcp.tool(description=CHAT_WITH_AUDIO, annotations=READ_ONLY_OPEN)
    async def chat_with_audio(
        file_path: str,
        model: AudioChatModel = "gpt-4o-audio-preview",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        return await audio.chat_with_audio(file_path, model, system_prompt, user_prompt)

    @mcp.tool(description=TRANSCRIBE_WITH_ENHANCEMENT, annotations=READ_ONLY_OPEN)
    async def transcribe_with_enhancement(
        file_path: str,
        enhancement_type: EnhancementType = "detailed",
        model: AudioModel = "gpt-4o-mini-transcribe",
    ):
        return await audio.transcribe_with_enhancement(file_path, enhancement_type, model)

    @mcp.tool(description=CREATE_AUDIO, annotations=WRITE_OPEN)
    async def create_audio(
        text_prompt: str,
        # None so each provider resolves its own default — hardcoding the OpenAI
        # pair here made provider="elevenlabs" unreachable from MCP.
        model: SpeechModel | ElevenLabsModel | None = None,
        voice: TTSVoice | str | None = None,
        instructions: str | None = None,
        speed: float = 1.0,
        output_filename: str | None = None,
        provider: TTSProviderName = "openai",
        voice_settings: VoiceSettingsDict | None = None,
    ):
        return await audio.create_audio(
            text_prompt=text_prompt,
            model=model,
            voice=voice,
            instructions=instructions,
            speed=speed,
            output_file_name=output_filename,
            provider=provider,
            voice_settings=voice_settings,
        )

    @mcp.tool(description=GENERATE_PODCAST, annotations=WRITE_OPEN)
    async def generate_podcast(
        script: podcast.PodcastScript,
        model: SpeechModel | ElevenLabsModel = "gpt-4o-mini-tts",
        provider: TTSProviderName = "openai",
        # `Filename`, not `str`: the same constraint SimulationBrief.filename
        # carries, so a path is refused by the schema before an episode is
        # synthesized rather than by storage after it is paid for.
        output_filename: Filename | None = None,
        verify: bool = False,
    ):
        return await podcast.generate_podcast(
            script, model=model, provider=provider, filename=output_filename, verify=verify
        )

    @mcp.tool(description=SIMULATE_PODCAST, annotations=WRITE_OPEN)
    async def simulate_podcast(brief: simulate.SimulationBrief):
        # No on_progress: MCP has no stderr channel to stream it to, and the
        # result already carries per-act summaries.
        return await simulate.simulate_podcast(brief)

    logger.info("Audio tools registered (10 tools)")


# ==================== MEDIA VIEWER (CONDITIONAL) ====================
# Resource is always registered (HTML is bundled); tools require at least one media path.
@mcp.resource(
    "ui://sanzaru/media-viewer.html",
    mime_type="text/html;profile=mcp-app",
    meta={
        "ui": {
            "prefersBorder": True,
            # Declare the iframe's resource requirements via the MCP Apps
            # protocol so hosts that honor `meta.ui.csp` allow the Blob URL
            # rendered by the viewer's <video>, <audio>, and <img> elements.
            "csp": {
                "resourceDomains": ["blob:"],
            },
        }
    },
)
def media_viewer_html() -> str:
    """Serve the bundled media viewer MCP App HTML."""
    return (
        importlib.resources.files("sanzaru").joinpath("app/media-viewer/dist/mcp-app.html").read_text(encoding="utf-8")
    )


if check_video_available() or check_audio_available() or check_image_available():
    from .descriptions import VIEW_MEDIA
    from .tools import media_viewer

    @mcp.tool(
        description=VIEW_MEDIA,
        annotations=READ_ONLY_CLOSED,
        meta={"ui": {"resourceUri": "ui://sanzaru/media-viewer.html"}},
    )
    async def view_media(
        media_type: Literal["video", "audio", "image"],
        filename: str,
    ):
        return await media_viewer.view_media(media_type, filename)

    @mcp.tool(
        description="Internal tool used by the MCP App media viewer to fetch base64-encoded chunks of media data. Do not call directly — use view_media instead.",  # noqa: E501
        annotations=READ_ONLY_CLOSED,
        meta={"ui": {"visibility": ["app"]}},
    )
    async def _get_media_data(
        media_type: Literal["video", "audio", "image"],
        filename: str,
        offset: int = 0,
        chunk_size: int = 2097152,
    ):
        return await media_viewer.get_media_data(media_type, filename, offset, chunk_size)

    logger.info("Media viewer tools registered (2 tools)")


# ==================== HTTP TRANSPORT SECURITY ====================
# Everything below applies to HTTP mode only. stdio is a single trusted client
# on a pipe and is unaffected.

#: Bearer token required on /mcp and /media when set.
HTTP_TOKEN_ENV = "SANZARU_HTTP_TOKEN"
#: Escape hatch for operators who terminate authentication in front of us
#: (a reverse proxy, a service mesh). Without it a non-loopback bind refuses
#: to start unauthenticated rather than silently serving the whole toolset.
ALLOW_UNAUTH_ENV = "SANZARU_ALLOW_UNAUTHENTICATED_HTTP"
#: Header carrying the proxy-verified caller identity for multi-tenant
#: deployments (Databricks Apps injects `x-forwarded-email`). Unset, no header
#: is trusted at all — see `identity_header_name`.
IDENTITY_HEADER_ENV = "SANZARU_IDENTITY_HEADER"

#: Content types /media is willing to emit. Anything else is served as an
#: opaque download: the route used to hand back `mimetypes.guess_type()` of a
#: caller-chosen name, so any stored file named `x.html` became an executable
#: document in the server's own origin — the one origin the SDK's rebinding
#: allowlist trusts, which put same-origin POSTs to /mcp within reach of a
#: stored payload (CWE-79).
_MEDIA_CONTENT_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

#: Sent on every /media response. `nosniff` stops content sniffing from
#: promoting an octet-stream back to HTML, `attachment` keeps the browser from
#: rendering it inline at all, and the sandbox CSP neuters script even if some
#: future change reintroduces an executable content type.
_MEDIA_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Disposition": "attachment",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}


def http_auth_token() -> str | None:
    """The configured bearer token, or None when HTTP auth is disabled."""
    token = os.environ.get(HTTP_TOKEN_ENV, "").strip()
    return token or None


def identity_header_name() -> str | None:
    """The header trusted for caller identity, or None when none is.

    Explicitly opt-in — there is no default header. The value is only
    trustworthy behind a proxy that both injects it and strips any
    client-supplied copy, and whether such a proxy exists is a fact about the
    deployment that only the operator knows. Defaulting to `x-forwarded-email`
    trusted that header everywhere: on a direct-exposed Databricks-backed
    server, any client holding the bearer token could choose an arbitrary
    tenant namespace by writing the header themselves, and a corporate proxy
    that injects it unasked silently re-pathed a single-tenant deployment's
    files into per-user prefixes.
    """
    return os.environ.get(IDENTITY_HEADER_ENV, "").strip().lower() or None


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


def _authorized(headers: Headers, token: str) -> bool:
    """Constant-time check of an `Authorization: Bearer <token>` header.

    Compared as bytes, not str. `secrets.compare_digest` refuses str operands
    that are not pure ASCII, and Starlette decodes raw header bytes as latin-1 —
    so a header of `Bearer \\xff` raised TypeError out of the auth path and came
    back as a 500 (the error middleware sits outside this one) instead of a 401.
    A non-ASCII `SANZARU_HTTP_TOKEN` was worse still: every single request 500ed.
    """
    supplied = headers.get("authorization", "")
    scheme, _, value = supplied.partition(" ")
    if scheme.lower() != "bearer":
        return False
    # latin-1 to encode, because that is the inverse of Starlette's decode: it
    # recovers the exact bytes the client put on the wire, which is what a UTF-8
    # token has to be compared against. Encoding as UTF-8 here would re-encode
    # already-mangled text and never match a non-ASCII token.
    return secrets.compare_digest(value.strip().encode("latin-1", "replace"), token.encode("utf-8"))


class BearerTokenMiddleware:
    """Require a bearer token on every HTTP request.

    The SDK's DNS-rebinding allowlist is not authentication: a direct network
    attacker sets `Host: 127.0.0.1:8000` themselves and matches it. Once the
    port is reachable, nothing else distinguished the operator from anyone else
    — every tool, including paid generation and `delete_video`, was callable by
    a single unauthenticated POST (CWE-306).
    """

    def __init__(self, app: object, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)  # type: ignore[operator]
            return
        if not _authorized(Headers(scope=scope), self._token):
            response = Response(
                content="Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)  # type: ignore[operator]


class UserContextMiddleware:
    """Bind the proxy-supplied identity to the request's context variable.

    The Databricks backend prefixes every path with the caller's slug, but
    nothing ever populated the contextvar it reads, so every tenant of a shared
    deployment resolved to the one shared namespace while the README promised
    isolation (CWE-862). This is the missing half.

    The header is only trustworthy because a proxy injects it *and* strips any
    client-supplied copy — the same assumption Databricks Apps documents. That
    is why this middleware is opt-in: it is only installed when
    `SANZARU_IDENTITY_HEADER` is explicitly set, so trusting a header is an
    operator's statement that such a proxy exists. A header nobody strips is a
    header anyone can set.
    """

    def __init__(self, app: object, header: str) -> None:
        self._app = app
        self._header = header

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)  # type: ignore[operator]
            return

        email = Headers(scope=scope).get(self._header, "").strip()
        ctx: UserContext | None = None
        if email:
            try:
                ctx = UserContext(email=email)
            except ValueError:
                logger.warning("Ignoring malformed identity header %s", self._header)

        token = set_user_context(ctx)
        try:
            await self._app(scope, receive, send)  # type: ignore[operator]
        finally:
            reset_user_context(token)


def _media_guard() -> TransportSecurityMiddleware:
    """Host/Origin policy for /media — deliberately the same object /mcp uses.

    FastMCP appends `custom_route` handlers straight onto the Starlette app,
    outside the middleware that guards the streamable-HTTP endpoint, so this
    route answered requests with any Host header at all. A rebound browser page
    could read the whole media library through it while /mcp rejected the same
    request (CWE-346).
    """
    return TransportSecurityMiddleware(mcp.settings.transport_security)


# Custom HTTP route for direct media serving (functional in HTTP mode)
@mcp.custom_route("/media/{media_type}/{filename:path}", methods=["GET"])
async def serve_media(request: Request) -> Response:
    """Serve media files directly over HTTP — no base64 overhead."""
    # Same rebinding policy as /mcp, applied by hand because custom routes sit
    # outside the SDK's middleware stack.
    rejected = await _media_guard().validate_request(request)
    if rejected is not None:
        return rejected

    # And the same credential. The route is registered on the app unconditionally,
    # so it cannot rely on the token middleware that only wraps HTTP mode.
    token = http_auth_token()
    if token is not None and not _authorized(request.headers, token):
        return Response(content="Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"})

    media_type = request.path_params["media_type"]
    filename = request.path_params["filename"]  # Path traversal protection handled by storage backend

    path_type = MEDIA_TYPE_TO_PATH_TYPE.get(media_type)
    if path_type is None:
        return Response(content="Invalid media type", status_code=400)

    storage = get_storage()
    try:
        data = await storage.read(path_type, filename)
    except (FileNotFoundError, ValueError):
        return Response(content="Not found", status_code=404)
    except PermissionError:
        # SANZARU_REQUIRE_USER_CONTEXT with no identity on the request. A refusal
        # to resolve a namespace is a 403, not a crash — uncaught it reached the
        # error middleware as a 500 with a traceback.
        return Response(content="Forbidden", status_code=403)

    suffix = os.path.splitext(filename)[1].lower()
    content_type = _MEDIA_CONTENT_TYPES.get(suffix, "application/octet-stream")

    return Response(content=data, media_type=content_type, headers=dict(_MEDIA_SECURITY_HEADERS))


# ==================== SERVER ENTRYPOINT ====================
def run_server(transport: Literal["stdio", "http"] = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the MCP server on the given transport.

    Tools are registered conditionally based on installed optional dependencies:
    - video: Sora video generation (no extra deps, always available)
    - audio: Whisper transcription, GPT-4o audio, TTS (requires pydub, ffmpeg-python)
    - image: GPT Vision image generation and reference management (requires pillow)

    Install with: uv add "sanzaru[video,audio,image]" or any combination.

    Paths are validated lazily at runtime when tools are called.

    Transport options:
    - stdio (default): Standard I/O for Claude Desktop and MCP clients
    - http: Stateless HTTP streaming for web clients and remote access
    """
    # Log enabled features
    enabled = []
    if check_video_available():
        enabled.append("video")
    if check_audio_available():
        enabled.append("audio")
    if check_image_available():
        enabled.append("image")

    if enabled:
        logger.info(f"Enabled features: {', '.join(enabled)}")
    else:
        logger.warning("No features enabled - install optional dependencies with: uv add 'sanzaru[all]'")

    # Run server with selected transport
    if transport == "http":
        _run_http(host=host, port=port)
    else:
        logger.info("Starting sanzaru MCP server over stdio")
        mcp.run()


def _run_http(*, host: str, port: int) -> None:
    """Serve the streamable-HTTP transport with sanzaru's own middleware stack.

    Built here rather than via `mcp.run(transport="streamable-http")` because
    that helper hands the bare app straight to uvicorn, leaving no seam to
    require a credential on. Authentication is the whole point: this transport
    exposes `delete_video`, paid generation, and every media file to whoever
    reaches the port.
    """
    import uvicorn

    token = http_auth_token()
    loopback = _is_loopback(host)

    if token is None and not loopback:
        if os.environ.get(ALLOW_UNAUTH_ENV, "").strip().lower() not in ("1", "true", "yes"):
            raise SystemExit(
                f"Refusing to serve {host}:{port} without authentication.\n"
                f"Set {HTTP_TOKEN_ENV} to a secret value and send it as "
                f"'Authorization: Bearer <token>', or set {ALLOW_UNAUTH_ENV}=1 if "
                f"a proxy in front of sanzaru already authenticates every request."
            )
        logger.warning(
            "Serving %s:%d with NO authentication because %s is set — every MCP tool "
            "and every media file is exposed to anyone who can reach this port.",
            host,
            port,
            ALLOW_UNAUTH_ENV,
        )

    # Configure for stateless HTTP (no session IDs needed - all state in OpenAI cloud)
    mcp.settings.stateless_http = True
    mcp.settings.host = host
    mcp.settings.port = port

    if not loopback:
        # The auto-configured allowlist only names localhost, and it was
        # computed from the *constructor's* host before this override. Against a
        # network peer who writes the Host header themselves it proves nothing,
        # and left in place it would reject the deployment's real hostname. The
        # bearer token is the control that actually holds here.
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    app = mcp.streamable_http_app()

    identity_header = identity_header_name()
    if identity_header is not None:
        app.add_middleware(UserContextMiddleware, header=identity_header)
        logger.info(
            "Trusting the %r header for caller identity (%s is set) — a proxy in front of "
            "sanzaru must inject it and strip any client-supplied copy",
            identity_header,
            IDENTITY_HEADER_ENV,
        )
    if token is not None:
        # Added last so it wraps outermost: identity is only trusted once the
        # request has proven it is allowed to be here at all.
        app.add_middleware(BearerTokenMiddleware, token=token)

    auth_state = "bearer token required" if token else "UNAUTHENTICATED"
    logger.info("Starting sanzaru MCP server over HTTP at http://%s:%d/mcp (%s)", host, port, auth_state)

    uvicorn.run(app, host=host, port=port, log_level=mcp.settings.log_level.lower())


def main():
    """Run the MCP server (argparse entrypoint).

    Kept for direct importers and `python -m sanzaru.server`; the installed
    `sanzaru` console script routes through sanzaru.cli:main, which delegates
    here-equivalent behavior to run_server().

    Environment variables should be set explicitly in .mcp.json or passed via the calling environment.
    For local development with .env files, install python-dotenv: uv add --dev python-dotenv
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Sanzaru MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transport (default: 8000)",
    )
    args = parser.parse_args()

    # Optional: load ./.env for local development. Scoped and filtered — see
    # sanzaru.dotenv_loader; the default ancestor walk let a file in an
    # untrusted workspace redirect the operator's API credentials.
    load_local_dotenv()

    run_server(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
