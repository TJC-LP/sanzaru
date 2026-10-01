# SPDX-License-Identifier: MIT
"""Configuration management for sanzaru MCP server.

This module handles:
- OpenAI client initialization
- Environment variable validation
- Path configuration with security checks
- Logging setup
"""

import logging
import os
import pathlib
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from openai.types import ImageModel

if TYPE_CHECKING:
    from elevenlabs.client import AsyncElevenLabs

    from .higgsfield.client import HiggsfieldClient

# ---------- Logging configuration ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # Log to stderr to avoid interfering with stdio MCP transport
)
logger = logging.getLogger("sanzaru")


# ---------- Image generation defaults ----------
# Single source of truth for the default image models. create_image injects the
# generation default into the Responses API image_generation tool config and
# generate_image uses it as its `model` default; edit_image defaults to the edit
# model. Typed with the SDK's ImageModel so the values are validated against the
# models the installed openai SDK knows about.
#
# gpt-image-2.5 ships as two variants at the same price as gpt-image-2: Flare is
# OpenAI's pick for fast, high-quality everyday generation, Sunburst for
# workflows where editing precision matters most. Per-model rules (transparent
# backgrounds, `xhigh`/`max` quality, `input_fidelity`) live in image_models.py.
DEFAULT_IMAGE_MODEL: ImageModel = "gpt-image-2.5-flare"
DEFAULT_IMAGE_EDIT_MODEL: ImageModel = "gpt-image-2.5-sunburst"

# The mainline model behind create_image lives in mainline_models.py (import-
# light, because the CLI lists the choices at startup).


# ---------- OpenAI client (stateless) ----------
_client_override: AsyncOpenAI | None = None


def set_client(client: AsyncOpenAI | None) -> None:
    """Install a process-wide AsyncOpenAI override; None restores default resolution.

    Used by the CLI runtime so one client (and its connection pool) is reused
    across every API call in an invocation — e.g. a poll loop — instead of
    re-instantiating per call. The MCP server never sets this.
    """
    global _client_override
    _client_override = client


def get_client() -> AsyncOpenAI:
    """Get an OpenAI async client instance.

    Returns:
        The installed override (see set_client), or a new client per call

    Raises:
        RuntimeError: If OPENAI_API_KEY environment variable is not set
    """
    if _client_override is not None:
        return _client_override
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)


# ---------- Higgsfield client (video generation) ----------
# Same lazy, cached shape as the ElevenLabs seam below, for the same reason:
# nothing pays for httpx client construction until a video tool actually runs.
# `HIGGSFIELD_BASE_URL` overrides the API endpoint under the rules in
# higgsfield/client.py: process environment only, https or loopback http.
HIGGSFIELD_BASE_URL_ENV = "HIGGSFIELD_BASE_URL"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_higgsfield_override: "HiggsfieldClient | None" = None
_higgsfield_cached: "HiggsfieldClient | None" = None


def set_higgsfield_client(client: "HiggsfieldClient | None") -> None:
    """Install a process-wide Higgsfield client override; None restores default resolution.

    Used by tests (a client over httpx.MockTransport). Also drops the cache.
    """
    global _higgsfield_override, _higgsfield_cached
    _higgsfield_override = client
    _higgsfield_cached = None


def higgsfield_base_url() -> str:
    """The Higgsfield API base URL: `HIGGSFIELD_BASE_URL` when set, else the production API.

    The credential is sent to this URL, so the override must be https, or plain
    http to a loopback host (a sandbox's credential-proxy forwarder), with no
    userinfo, query or fragment. A trailing slash is dropped so request paths
    join cleanly.

    Raises:
        RuntimeError: If HIGGSFIELD_BASE_URL is set to anything else
    """
    from .higgsfield.client import BASE_URL

    raw = (os.getenv(HIGGSFIELD_BASE_URL_ENV) or "").strip()
    if not raw:
        return BASE_URL
    parts = urlsplit(raw)
    try:
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        host = None
    clean = host is not None and parts.username is None and parts.password is None
    clean = clean and not parts.query and not parts.fragment
    if clean and (parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK_HOSTS)):
        return raw.rstrip("/")
    raise RuntimeError(
        f"{HIGGSFIELD_BASE_URL_ENV} must be an https URL, or http to 127.0.0.1, ::1 or localhost, "
        "with no credentials, query or fragment"
    )


def get_higgsfield_client() -> "HiggsfieldClient":
    """Get the Higgsfield client, building it from `HF_KEY` on first use.

    Returns:
        The installed override (see set_higgsfield_client), else a cached client

    Raises:
        RuntimeError: If HF_KEY is unset or not `key_id:key_secret`, or
            HIGGSFIELD_BASE_URL is not an accepted URL (see higgsfield_base_url)
    """
    global _higgsfield_cached
    if _higgsfield_override is not None:
        return _higgsfield_override
    if _higgsfield_cached is not None:
        return _higgsfield_cached

    raw = os.getenv("HF_KEY", "").strip()
    if not raw:
        raise RuntimeError("HF_KEY is not set (Higgsfield video generation)")
    key_id, sep, secret = raw.partition(":")
    if not sep or not key_id.strip() or not secret.strip():
        raise RuntimeError("HF_KEY must be key_id:key_secret")

    from .higgsfield.client import HiggsfieldClient

    _higgsfield_cached = HiggsfieldClient(key_id.strip(), secret.strip(), base_url=higgsfield_base_url())
    return _higgsfield_cached


async def close_higgsfield_client() -> None:
    """Close the lazily-built Higgsfield client, if one was ever created.

    A no-op when video was never used. Failures are swallowed: this runs in the
    CLI's teardown `finally`, where raising would replace the command's result.
    """
    global _higgsfield_cached
    client = _higgsfield_cached
    _higgsfield_cached = None
    if client is None:
        return
    try:
        await client.aclose()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask the command's outcome
        logger.debug("Closing the Higgsfield client failed: %s", exc)


# ---------- ElevenLabs client (optional TTS provider) ----------
# Mirrors the OpenAI seam above, with one deliberate difference: the client is
# built lazily and cached rather than installed eagerly by the CLI runtime, so
# `sanzaru <cmd> --help` never pays the elevenlabs import. The SDK is an optional
# extra, so every import of it is function-local.
_elevenlabs_override: "AsyncElevenLabs | None" = None
_elevenlabs_cached: "AsyncElevenLabs | None" = None

_ELEVENLABS_INSTALL_HINT = (
    "provider='elevenlabs' requires the optional extra — install with: uv pip install 'sanzaru[elevenlabs]'"
)


def set_elevenlabs_client(client: "AsyncElevenLabs | None") -> None:
    """Install a process-wide AsyncElevenLabs override; None restores default resolution.

    Used by tests and by callers that want to supply a pre-configured client.
    Also drops the lazily-built cache so the next get_elevenlabs_client() re-resolves.
    """
    global _elevenlabs_override, _elevenlabs_cached
    _elevenlabs_override = client
    _elevenlabs_cached = None


def get_elevenlabs_client() -> "AsyncElevenLabs":
    """Get an ElevenLabs async client, reusing one connection pool per process.

    `ELEVENLABS_BASE_URL` overrides the API endpoint, mirroring what
    `OPENAI_BASE_URL` does for the OpenAI seam above. The difference is where
    the read happens: `AsyncOpenAI` picks its variable up from the environment
    itself, while the Fern-generated ElevenLabs client only accepts `base_url`
    as a constructor argument, so this function has to do it. Sandboxed
    deployments point the variable at a loopback credential proxy that holds
    the real key, which is why the endpoint must be overridable without
    touching call sites.

    Returns:
        The installed override (see set_elevenlabs_client), else a cached client

    Raises:
        ImportError: If the `elevenlabs` extra is not installed
        RuntimeError: If ELEVENLABS_API_KEY is not set
    """
    global _elevenlabs_cached
    if _elevenlabs_override is not None:
        return _elevenlabs_override
    if _elevenlabs_cached is not None:
        return _elevenlabs_cached

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (required for provider='elevenlabs')")

    try:
        from elevenlabs.client import AsyncElevenLabs
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ImportError(_ELEVENLABS_INSTALL_HINT) from exc

    # Unset or blank leaves base_url=None, so the SDK keeps its own production
    # default rather than being handed an empty string it would treat as a URL.
    base_url = (os.getenv("ELEVENLABS_BASE_URL") or "").strip() or None
    _elevenlabs_cached = AsyncElevenLabs(api_key=api_key, base_url=base_url)
    return _elevenlabs_cached


async def close_elevenlabs_client() -> None:
    """Close the lazily-built ElevenLabs client, if one was ever created.

    A no-op when the provider was never used, so unrelated CLI commands pay
    nothing (not even the import).

    elevenlabs 2.60 exposes no `aclose()` (nor any other close method) on
    AsyncElevenLabs, so the connection pool has to be reached through the
    generated wrapper: AsyncElevenLabs._client_wrapper.httpx_client is the SDK's
    AsyncHttpClient, which holds the real httpx.AsyncClient. `aclose()` is still
    tried first so a future SDK that grows one wins. Every hop is getattr-guarded
    and failures are swallowed: this runs in the CLI's teardown `finally`, where
    raising would replace the command's own result.
    """
    global _elevenlabs_cached
    client = _elevenlabs_cached
    _elevenlabs_cached = None
    if client is None:
        return

    aclose = getattr(client, "aclose", None)
    if not callable(aclose):
        wrapper = getattr(client, "_client_wrapper", None)
        http_client = getattr(wrapper, "httpx_client", None)
        # The wrapper's `httpx_client` is the SDK's own AsyncHttpClient, which
        # nests the httpx.AsyncClient under the same attribute name.
        pool = getattr(http_client, "httpx_client", http_client)
        aclose = getattr(pool, "aclose", None)

    if not callable(aclose):
        logger.debug("ElevenLabs client exposes no close method; leaving its connection pool to GC")
        return
    try:
        await aclose()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask the command's outcome
        logger.debug("Closing the ElevenLabs client failed: %s", exc)


# ---------- Path configuration (runtime) ----------

# Mapping from path_type to (individual env var, subdirectory under SANZARU_MEDIA_PATH)
_MEDIA_SUBDIRS: dict[str, tuple[str, str]] = {
    "video": ("VIDEO_PATH", "videos"),
    "reference": ("IMAGE_PATH", "images"),
    "audio": ("AUDIO_PATH", "audio"),
}

_ERROR_NAMES: dict[str, str] = {
    "video": "Video download directory",
    "reference": "Image directory",
    "audio": "Audio files directory",
}


def _resolve_media_path(path_type: Literal["video", "reference", "audio"]) -> tuple[str | None, str, bool]:
    """Resolve path string from individual env var or SANZARU_MEDIA_PATH.

    Priority: individual env var > SANZARU_MEDIA_PATH/{subdir} > None.

    Args:
        path_type: Path type to resolve

    Returns:
        (path_str, env_var_name_for_errors, using_unified) tuple
    """
    env_var, subdir = _MEDIA_SUBDIRS[path_type]

    individual = os.getenv(env_var)
    if individual and individual.strip():
        return individual.strip(), env_var, False

    unified = os.getenv("SANZARU_MEDIA_PATH")
    if unified and unified.strip():
        return os.path.join(unified.strip(), subdir), "SANZARU_MEDIA_PATH", True

    return None, env_var, False


def is_path_configured(path_type: Literal["video", "reference", "audio"]) -> bool:
    """True when a media directory is configured for path_type (env present; not validated)."""
    return _resolve_media_path(path_type)[0] is not None


def peek_media_path(path_type: Literal["video", "reference", "audio"]) -> pathlib.Path | None:
    """The configured media directory for path_type, without validating or creating it.

    `get_path()` is the real accessor and has side effects a read-only lookup
    must not have: under `SANZARU_MEDIA_PATH` it mkdirs the subdirectory. The
    CLI's bare-name ambiguity check only wants to know whether a same-named file
    *exists* in the library, so it asks here. None when nothing is configured.
    """
    path_str = _resolve_media_path(path_type)[0]
    if path_str is None:
        return None
    try:
        return pathlib.Path(path_str.strip()).resolve()
    except (ValueError, OSError):
        return None


@lru_cache(maxsize=3)
def get_path(path_type: Literal["video", "reference", "audio"]) -> pathlib.Path:
    """Get and validate a configured path from environment.

    Supports two configuration modes:
    1. Individual env vars: VIDEO_PATH, IMAGE_PATH, AUDIO_PATH (take precedence)
    2. Unified root: SANZARU_MEDIA_PATH (auto-creates videos/, images/, audio/ subdirs)

    Creates paths lazily at runtime, so this works with both `uv run` and `mcp run`.

    Security: Rejects symlinks in environment variable paths to prevent directory traversal.

    Args:
        path_type: Either "video" for VIDEO_PATH, "reference" for IMAGE_PATH, or "audio" for AUDIO_PATH

    Returns:
        Validated absolute path

    Raises:
        RuntimeError: If environment variable not set, malformed, path doesn't exist, isn't a directory, or is a symlink
    """
    error_name = _ERROR_NAMES[path_type]
    path_str, env_var, using_unified = _resolve_media_path(path_type)

    # Validate env var is set and not empty/whitespace
    if not path_str:
        individual_var = _MEDIA_SUBDIRS[path_type][0]
        raise RuntimeError(f"{error_name} not configured. Set {individual_var} or SANZARU_MEDIA_PATH")

    # Strip whitespace and resolve path with error handling
    try:
        path = pathlib.Path(path_str.strip()).resolve()
    except (ValueError, OSError) as e:
        raise RuntimeError(f"Invalid {error_name} path '{path_str}': {e}") from e

    # Security: Reject symlinks in configured paths (env vars only, not user filenames)
    # Check the original path before resolution to catch symlinks. `is_symlink()`
    # alone: `exists()` follows the link, so a *dangling* one answered False here
    # and the auto-create branch below then mkdir'ed its target through the link.
    original_path = pathlib.Path(path_str.strip())
    try:
        if original_path.is_symlink():
            raise RuntimeError(f"{error_name} cannot be a symbolic link: {path_str}")
    except PermissionError as e:
        raise RuntimeError(f"Cannot validate {error_name}: permission denied for {path_str}") from e

    # Auto-create subdirectories when using unified SANZARU_MEDIA_PATH
    if using_unified and not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Auto-created directory: %s", path)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to auto-create {error_name} at {path}: {e}") from e

    # Validate path exists and is a directory
    if not path.exists():
        raise RuntimeError(f"{env_var}: {error_name} does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"{env_var}: {error_name} is not a directory: {path}")

    return path
