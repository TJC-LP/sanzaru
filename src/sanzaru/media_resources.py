# SPDX-License-Identifier: MIT
"""Media as MCP resources, so a person can attach their own creations.

The tools are for the model; this is for the person sitting in front of it. A
rendered episode or a generated poster becomes something they can drop into the
next conversation from the client's own attachment menu, instead of hunting for
it on disk.

**Templates, not a listing, and that is a performance decision.** The obvious
shape — one concrete resource per file — is the wrong one here:

- `MCPServer.list_resources()` reads a *static* registry, so a per-file resource
  is a snapshot taken at import time. Every file generated afterwards is
  invisible until the process restarts, which is every file that matters.
- The high-level server exposes no cursor on resource listing, so the answer is
  one unbounded array. A tenant with a few hundred renders makes every
  `resources/list` pay for all of them, on a call clients make at connect time.
- On the Databricks backend a listing is an HTTP round trip *per media type*,
  and the payload the client actually wanted was one file.

So a media type is one template — ``sanzaru://image/{filename}`` — and discovery
happens through `completion/complete`, which filters server-side and returns at
most a hundred names for what the person has typed so far. Nothing enumerates
the backend, and the cost of finding a file no longer scales with how much has
been generated.

Three further things keep the hot paths cheap:

- **A short-TTL listing cache, keyed by identity.** Completion fires per
  keystroke; without this, each one is a fresh storage listing. The key includes
  the caller's identity because on a shared deployment the listing *is*
  per-tenant, and a cache that ignored that would serve one person's filenames
  to another.
- **`stat()` before `read()`.** An oversized file is refused from its metadata,
  so the bytes are never transferred. On Databricks that is the difference
  between a HEAD and a full GET of something too big to return anyway.
- **Content types come from an allowlist**, never from `mimetypes.guess_type()`
  of a caller-chosen name — the same rule the `/media` route follows, and for
  the same reason: a stored `.html` must not come back as an executable
  document.

One protocol limitation worth knowing before trying to "fix" it: a template
declares **one** MIME type for every file it can produce
(`ResourceTemplate.create_resource` copies `self.mime_type` onto the resource it
builds, and a read may only return `str` or `bytes`). Images here are variously
PNG, JPEG and WebP, so declaring any single type would mislabel the others —
serving a JPEG as `image/png` is a worse failure than declaring nothing. The
templates therefore leave it unset and the URI carries the extension, which is
what clients fall back to. `content_type_for` still gates *which* files may be
read at all, so the allowlist is enforced either way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mcp_types import Completion

from .config import logger
from .storage import get_storage
from .storage.protocol import FileInfo, PathType
from .user_context import get_user_context

URI_SCHEME = "sanzaru"

#: The resource-facing media name and the storage path type behind it. The names
#: match `view_media`'s `media_type` argument rather than the storage vocabulary,
#: so a person reading a URI and a model calling a tool use the same word
#: ("image", not "reference").
MEDIA_PATH_TYPES: dict[str, PathType] = {
    "image": "reference",
    "video": "video",
    "audio": "audio",
}

#: What may be served, and as what. Deliberately an allowlist and deliberately
#: not `mimetypes.guess_type`: the extension comes from a caller-chosen filename,
#: and anything unrecognised is refused rather than guessed at.
MEDIA_CONTENT_TYPES: dict[str, str] = {
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

#: Ceiling on a single resource read. A resource is delivered as one base64 blob
#: with no chunking anywhere in the protocol, so this is the point past which the
#: reply stops being something a client can reasonably hold. Bigger files still
#: have `view_media`, the viewer's download button, and the `/media` route.
MAX_RESOURCE_BYTES = 32 * 1024 * 1024

#: The protocol's own ceiling on returned completion values.
COMPLETION_LIMIT = 100

#: How long a per-identity listing may be reused. Long enough to absorb a burst
#: of keystrokes, short enough that a file generated mid-conversation shows up
#: without anyone reconnecting.
LISTING_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    files: tuple[FileInfo, ...]


#: Keyed by (identity, media) — never by media alone. On a shared deployment the
#: storage backend resolves a different directory per caller, so a cache that
#: dropped the identity would hand one tenant another's filenames.
_listing_cache: dict[tuple[str, str], _CacheEntry] = {}


def _identity_key() -> str:
    """Cache partition for the caller, or "" in single-tenant mode."""
    context = get_user_context()
    return context.email if context is not None else ""


def clear_listing_cache() -> None:
    """Drop every cached listing. For tests and for a backend swap."""
    _listing_cache.clear()


def content_type_for(filename: str) -> str | None:
    """The MIME type this server will serve `filename` as, or None if it will not."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return MEDIA_CONTENT_TYPES.get(f".{suffix}")


async def list_media(media: str, *, use_cache: bool = True) -> tuple[FileInfo, ...]:
    """Servable files of one media type, newest first, cached per identity.

    Only files with an allowlisted extension are returned: offering a name the
    read path would refuse is worse than not offering it.
    """
    path_type = MEDIA_PATH_TYPES[media]
    key = (_identity_key(), media)
    now = time.monotonic()

    if use_cache:
        cached = _listing_cache.get(key)
        if cached is not None and cached.expires_at > now:
            return cached.files

    storage = get_storage()
    try:
        found = await storage.list_files(path_type)
    except (OSError, ValueError) as exc:
        # A missing or unreadable directory is an empty menu, not a failure: the
        # person is typing into a completion box, and an exception there surfaces
        # as a broken client rather than as "nothing yet".
        logger.debug("media resources: listing %s failed: %s", media, exc)
        return ()

    servable = tuple(
        sorted(
            (info for info in found if content_type_for(info.name) is not None),
            key=lambda info: info.modified_timestamp,
            reverse=True,
        )
    )
    _listing_cache[key] = _CacheEntry(expires_at=now + LISTING_TTL_SECONDS, files=servable)
    return servable


async def read_media(media: str, filename: str) -> bytes:
    """Return one media file's bytes for a resource read.

    Args:
        media: One of `MEDIA_PATH_TYPES`.
        filename: Bare filename. Path traversal is refused by the template's
            `ResourceSecurity` before this runs, and again by the storage
            backend's own containment check.

    Raises:
        ValueError: Unknown media type, extension this server will not serve,
            missing file, or a file past `MAX_RESOURCE_BYTES`.
    """
    if media not in MEDIA_PATH_TYPES:
        raise ValueError(f"unknown media type {media!r}; expected one of {', '.join(sorted(MEDIA_PATH_TYPES))}")
    if content_type_for(filename) is None:
        raise ValueError(f"refusing to serve {filename!r}: extension is not an allowlisted media type")

    path_type = MEDIA_PATH_TYPES[media]
    storage = get_storage()

    # Metadata first. The size check is worth a round trip precisely because it
    # saves transferring a file we would then refuse.
    try:
        info = await storage.stat(path_type, filename)
    except FileNotFoundError as exc:
        raise ValueError(f"{media} not found: {filename}") from exc

    if info.size_bytes > MAX_RESOURCE_BYTES:
        raise ValueError(
            f"{filename} is {info.size_bytes / 1048576:.1f} MB, over the "
            f"{MAX_RESOURCE_BYTES / 1048576:.0f} MB resource limit; "
            f"use view_media or the /media route for a file this size"
        )

    data = await storage.read(path_type, filename)
    logger.info("media resource read: %s/%s (%d bytes)", media, filename, len(data))
    return data


def _matches(name: str, partial: str) -> bool:
    """Case-insensitive prefix match, falling back to substring.

    Prefix first because that is what someone typing a filename means; substring
    as well because generated names lead with a stem the person did not choose
    (`img_1789570535.png`), so the memorable part is often in the middle.
    """
    if not partial:
        return True
    lowered = name.lower()
    needle = partial.lower()
    return lowered.startswith(needle) or needle in lowered


async def complete_media_filename_for(media: str, partial: str) -> Completion:
    """Filename completions for one media template, filtered server-side.

    The whole reason this exists instead of a resource listing: the client
    receives at most `COMPLETION_LIMIT` names matching what has been typed,
    rather than everything in the directory. `total` and `has_more` tell it when
    there is more behind the filter, so narrowing the input is visibly useful.
    """
    if media not in MEDIA_PATH_TYPES:
        return Completion(values=[], total=0, has_more=False)

    files = await list_media(media)
    matching = [info.name for info in files if _matches(info.name, partial)]
    return Completion(
        values=matching[:COMPLETION_LIMIT],
        total=len(matching),
        has_more=len(matching) > COMPLETION_LIMIT,
    )


def template_uri(media: str) -> str:
    """The URI template for one media type."""
    return f"{URI_SCHEME}://{media}/{{filename}}"


def media_for_template_uri(uri: str) -> str | None:
    """The media type a template URI refers to, or None if it is not one of ours.

    Matches the template form (`sanzaru://image/{filename}`) and a concrete URI
    alike, because a client asking for completions may send either.
    """
    prefix = f"{URI_SCHEME}://"
    if not uri.startswith(prefix):
        return None
    media = uri[len(prefix) :].split("/", 1)[0]
    return media if media in MEDIA_PATH_TYPES else None
