# SPDX-License-Identifier: MIT
"""Job ids and model slugs — the two caller-supplied strings that reach a URL path.

A Higgsfield request id is a bare UUID. sanzaru hands it out as `hf_<uuid>` so
`wait_for` (and `sanzaru wait`) can tell a video job from an image job
(`resp_…`) by prefix, the way they always have, and so a stale Sora id
(`video_…`) can be recognised and explained instead of producing a confusing
404. The prefix is stripped, and the rest must parse as a UUID, before any id
is interpolated into `/requests/{id}/…`.

The model slug is interpolated into the *path* of the generation POST
(`POST /<slug>`), so a caller-chosen slug could otherwise address another
endpoint entirely — `requests/<id>/cancel`, `files/generate-upload-url`,
`estimate/…`. `validate_slug` confines it to catalog-shaped paths.
"""

from __future__ import annotations

import re
import uuid

from ..utils import validate_resource_id

JOB_PREFIX = "hf_"

SORA_RETIRED = (
    "Sora video ids no longer work: OpenAI retired the Videos API and every Sora model on 2026-09-24. "
    "sanzaru now generates video with Higgsfield — create a new job with create_video (ids look like hf_…)."
)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*){1,5}$")
_RESERVED_FIRST_SEGMENTS = frozenset({"requests", "files", "estimate", "models"})


def to_job_id(request_id: str) -> str:
    """`<uuid>` → `hf_<uuid>` (canonical lower-case form)."""
    return JOB_PREFIX + str(uuid.UUID(request_id))


def to_request_id(job_id: str, param: str = "video_id") -> str:
    """`hf_<uuid>` → `<uuid>`, refusing anything else before it reaches a URL."""
    validate_resource_id(job_id, param)
    if job_id.startswith("video_"):
        raise ValueError(SORA_RETIRED)
    if not job_id.startswith(JOB_PREFIX):
        raise ValueError(f"{param}={job_id!r:.80} is not a sanzaru video job id (expected hf_<uuid>)")
    try:
        return str(uuid.UUID(job_id[len(JOB_PREFIX) :]))
    except ValueError:
        raise ValueError(f"{param}={job_id!r:.80} is not a sanzaru video job id (expected hf_<uuid>)") from None


def validate_slug(slug: str) -> str:
    """Return `slug` if it is shaped like a catalog model id, else raise ValueError."""
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug) or ".." in slug:
        raise ValueError(
            f"model={slug!r:.80} is not a Higgsfield model id: expected a catalog path such as "
            "'bytedance/seedance-2.5/text-to-video' (lower-case, '/'-separated)"
        )
    if slug.split("/", 1)[0] in _RESERVED_FIRST_SEGMENTS:
        raise ValueError(f"model={slug!r:.80} names an API endpoint, not a model")
    return slug
