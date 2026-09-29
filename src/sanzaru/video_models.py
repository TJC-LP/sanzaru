# SPDX-License-Identifier: MIT
"""The curated Higgsfield video models, and the one place their dialects are translated.

Every Higgsfield endpoint has its own request schema, and every schema is
`additionalProperties: false`, so one unknown key is a rejected request. The
dialects genuinely differ: Seedance takes `generate_audio: bool` and
`end_image_url`, Kling takes `sound: "on"|"off"` and `last_image_url`; Seedance
runs 4-30 s, Kling 3-15 s; aspect ratio applies only to text-to-video (image-
to-video frames follow the image). Callers speak one vocabulary — duration,
aspect_ratio, resolution, audio, a start and end image — and `build_arguments`
translates and validates it here, *before* any network call, so a bad value is
a readable ValueError rather than a paid-for round trip or an upload wasted on
a request the API was always going to refuse.

Any other catalog model stays reachable as a raw slug (`bytedance/…/…`):
`build_raw_arguments` maps the common names verbatim and merges `extra`, and
the free remote estimate is what validates it.

Stdlib only (plus `higgsfield.ids`, also stdlib), so the CLI can offer these
ids as `--model` choices without loading httpx — the same reason
`mainline_models.py` exists. Schemas were read from docs.higgsfield.ai on the
date in each spec's `verified` field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .higgsfield.ids import validate_slug

Operation = Literal["text", "image", "edit", "extend"]
AspectRatio = Literal["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
Resolution = Literal["480p", "720p", "1080p"]
AudioStyle = Literal["bool", "on_off"]

ALL_ASPECT_RATIOS: frozenset[str] = frozenset({"16:9", "4:3", "1:1", "3:4", "9:16", "21:9"})

# Keys sanzaru fills from its own arguments. `extra` may not set them, or a
# caller could bypass validation by smuggling `duration` through the side door.
MANAGED_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "duration",
        "aspect_ratio",
        "resolution",
        "generate_audio",
        "sound",
        "image_url",
        "end_image_url",
        "last_image_url",
        "video_url",
        "image_urls",
    }
)


@dataclass(frozen=True, slots=True)
class VideoModelSpec:
    """One curated model: its endpoints per operation and the rules its schemas enforce."""

    id: str
    family: str
    slugs: Mapping[Operation, str]
    durations: tuple[int, int]
    default_duration: int
    text_aspect_ratios: frozenset[str]
    resolutions: frozenset[str]
    default_resolution: str | None
    """None means the tier fixes the resolution (Kling std/pro/4k), so passing one is an error."""
    audio_param: str | None
    audio_style: AudioStyle
    end_image_param: str | None
    prompt_max_chars: int | None
    prompt_required_ops: frozenset[str]
    extra_keys: Mapping[Operation, frozenset[str]]
    verified: str
    price_note: str
    reference_images_ops: frozenset[str] = field(default_factory=frozenset)
    """Operations that accept extra reference images (sent as `image_urls`)."""

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(self.slugs)


_SEEDANCE_BASE = "bytedance/seedance-2.5"

_KLING_EXTRA = frozenset({"cfg_scale", "multi_shots", "multi_prompt"})


def _kling(tier: str, price_note: str) -> VideoModelSpec:
    return VideoModelSpec(
        id=f"kling-3.0-{tier}",
        family="kling-3.0",
        slugs={
            "text": f"kling-video/v3.0/{tier}/text-to-video",
            "image": f"kling-video/v3.0/{tier}/image-to-video",
        },
        durations=(3, 15),
        default_duration=5,
        text_aspect_ratios=frozenset({"16:9", "9:16", "1:1"}),
        resolutions=frozenset(),
        default_resolution=None,
        audio_param="sound",
        audio_style="on_off",
        end_image_param="last_image_url",
        prompt_max_chars=2500,
        prompt_required_ops=frozenset({"text", "image"}),
        extra_keys={"text": _KLING_EXTRA, "image": _KLING_EXTRA},
        verified="2026-09-28",
        price_note=price_note,
    )


VIDEO_MODELS: dict[str, VideoModelSpec] = {
    "seedance-2.5": VideoModelSpec(
        id="seedance-2.5",
        family="seedance-2.5",
        slugs={
            "text": f"{_SEEDANCE_BASE}/text-to-video",
            "image": f"{_SEEDANCE_BASE}/image-to-video",
            "edit": f"{_SEEDANCE_BASE}/video-edit",
            "extend": f"{_SEEDANCE_BASE}/video-extend",
        },
        durations=(4, 30),
        default_duration=5,
        text_aspect_ratios=ALL_ASPECT_RATIOS,
        resolutions=frozenset({"480p", "720p"}),
        default_resolution="720p",
        audio_param="generate_audio",
        audio_style="bool",
        end_image_param="end_image_url",
        prompt_max_chars=None,
        # image-to-video takes an optional prompt; the rest require one.
        prompt_required_ops=frozenset({"text", "edit", "extend"}),
        extra_keys={
            "text": frozenset({"bitrate_mode", "output_format"}),
            "image": frozenset({"bitrate_mode"}),
            "edit": frozenset({"bitrate_mode", "video_urls", "audio_urls"}),
            "extend": frozenset({"bitrate_mode", "video_urls", "audio_urls"}),
        },
        verified="2026-09-28",
        price_note="~$0.46/s at 720p, ~$0.21/s at 480p (edit/extend also bill the source clip)",
        reference_images_ops=frozenset({"edit", "extend"}),
    ),
    "kling-3.0-std": _kling("std", "~$0.35 per 5 s"),
    "kling-3.0-pro": _kling("pro", "~$0.46 per 5 s"),
    "kling-3.0-4k": _kling("4k", "~$1.16 per 5 s"),
    # Turbo's schema was not read: only its slugs and price were confirmed, so
    # it gets no audio or end-frame mapping and no extra keys. Anything more is
    # reachable through the raw slug until someone verifies it.
    "kling-3.0-turbo": VideoModelSpec(
        id="kling-3.0-turbo",
        family="kling-3.0",
        slugs={
            "text": "kling-video/v3.0-turbo/text-to-video",
            "image": "kling-video/v3.0-turbo/image-to-video",
        },
        durations=(3, 15),
        default_duration=5,
        text_aspect_ratios=frozenset({"16:9", "9:16", "1:1"}),
        resolutions=frozenset(),
        default_resolution=None,
        audio_param=None,
        audio_style="bool",
        end_image_param=None,
        prompt_max_chars=2500,
        prompt_required_ops=frozenset({"text", "image"}),
        extra_keys={"text": frozenset(), "image": frozenset()},
        verified="2026-09-28 (slugs and price only)",
        price_note="~$0.31 per 5 s",
    ),
}

DEFAULT_VIDEO_MODEL = "seedance-2.5"
VIDEO_MODEL_IDS: tuple[str, ...] = tuple(VIDEO_MODELS)


def resolve_model(model: str) -> VideoModelSpec | None:
    """A curated id → its spec; a catalog slug (contains '/') → None, meaning raw mode."""
    if model in VIDEO_MODELS:
        return VIDEO_MODELS[model]
    if "/" in model:
        return None
    raise ValueError(
        f"unknown video model {model!r:.80}: use one of {', '.join(VIDEO_MODEL_IDS)}, "
        "or a full Higgsfield catalog slug such as 'bytedance/seedance-2.5/text-to-video'"
    )


def _check_duration(duration: int, low: int, high: int, model: str) -> None:
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError(f"duration must be a whole number of seconds, got {duration!r}")
    if not low <= duration <= high:
        raise ValueError(f"duration must be between {low} and {high} seconds for {model}, got {duration}")


def _check_extra(extra: Mapping[str, object] | None, allowed: frozenset[str], where: str) -> dict[str, object]:
    if not extra:
        return {}
    managed = sorted(set(extra) & MANAGED_KEYS)
    if managed:
        raise ValueError(
            f"extra may not set {', '.join(managed)}: pass them as the dedicated arguments so they are validated"
        )
    unknown = sorted(set(extra) - allowed)
    if unknown:
        accepted = ", ".join(sorted(allowed)) or "none"
        raise ValueError(
            f"extra key(s) {', '.join(unknown)} are not accepted by {where} (accepted: {accepted}). "
            "Pass the full catalog slug as `model` to send arbitrary fields."
        )
    return dict(extra)


def build_arguments(
    spec: VideoModelSpec,
    op: Operation,
    *,
    prompt: str | None = None,
    duration: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    audio: bool | None = None,
    image_url: str | None = None,
    end_image_url: str | None = None,
    video_url: str | None = None,
    reference_image_urls: list[str] | None = None,
    extra: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Translate the common vocabulary into `spec`'s request body for `op`.

    Returns `(slug, arguments)`. Only keys the caller set are sent, so the
    API's own defaults apply to the rest. Raises ValueError, before any
    network call, for anything the endpoint would refuse.
    """
    where = f"{spec.id} ({op})"
    if op not in spec.slugs:
        raise ValueError(
            f"{spec.id} does not support {op}; it supports {', '.join(spec.operations)}. "
            f"{'seedance-2.5 supports edit and extend.' if op in ('edit', 'extend') else ''}".rstrip()
        )
    args: dict[str, object] = {}

    text = (prompt or "").strip()
    if op in spec.prompt_required_ops and not text:
        raise ValueError(f"{where} needs a non-empty prompt")
    if spec.prompt_max_chars is not None and len(text) > spec.prompt_max_chars:
        raise ValueError(f"prompt is {len(text)} characters; {spec.id} accepts at most {spec.prompt_max_chars}")
    if text:
        args["prompt"] = text

    if duration is not None:
        if op == "edit":
            raise ValueError("duration is not accepted for edit: it is derived from the source video")
        _check_duration(duration, *spec.durations, spec.id)
        args["duration"] = duration

    if aspect_ratio is not None:
        if op != "text":
            raise ValueError(
                f"aspect_ratio applies to text-to-video only; for {op} the framing follows the input "
                "(crop the reference first with prepare_reference_image)"
            )
        if aspect_ratio not in spec.text_aspect_ratios:
            raise ValueError(
                f"aspect_ratio {aspect_ratio!r} is not supported by {spec.id}; "
                f"use one of {', '.join(sorted(spec.text_aspect_ratios))}"
            )
        args["aspect_ratio"] = aspect_ratio

    if resolution is not None:
        if spec.default_resolution is None:
            raise ValueError(
                f"{spec.id} has a fixed resolution; pick the tier instead (kling-3.0-std, kling-3.0-pro, kling-3.0-4k)"
            )
        if resolution not in spec.resolutions:
            raise ValueError(
                f"resolution {resolution!r} is not supported by {spec.id}; "
                f"use one of {', '.join(sorted(spec.resolutions))}"
            )
        args["resolution"] = resolution

    if audio is not None:
        if spec.audio_param is None:
            raise ValueError(f"{spec.id} has no audio control in sanzaru's table; omit audio")
        args[spec.audio_param] = audio if spec.audio_style == "bool" else ("on" if audio else "off")

    if op == "image":
        if not image_url:
            raise ValueError("image-to-video needs a reference image")
        args["image_url"] = image_url
    elif image_url:
        raise ValueError(f"a reference image (start frame) applies to image-to-video only, not {op}")

    if end_image_url:
        if op != "image":
            raise ValueError("an end image needs a reference image: it is the last frame of image-to-video")
        if spec.end_image_param is None:
            raise ValueError(f"{spec.id} does not accept an end frame")
        args[spec.end_image_param] = end_image_url

    if op in ("edit", "extend"):
        if not video_url:
            raise ValueError(f"{op} needs a source video")
        args["video_url"] = video_url
    elif video_url:
        raise ValueError(f"a source video applies to edit and extend only, not {op}")

    if reference_image_urls:
        if op not in spec.reference_images_ops:
            raise ValueError(f"reference images are not accepted by {where}")
        if len(reference_image_urls) > 30:
            raise ValueError(f"at most 30 reference images, got {len(reference_image_urls)}")
        args["image_urls"] = list(reference_image_urls)

    args.update(_check_extra(extra, spec.extra_keys.get(op, frozenset()), where))
    return spec.slugs[op], args


def build_raw_arguments(
    slug: str,
    *,
    prompt: str | None = None,
    duration: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    audio: bool | None = None,
    image_url: str | None = None,
    end_image_url: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Arguments for an uncurated catalog slug: common names verbatim, `extra` merged.

    No local schema exists, so nothing beyond the slug's shape is checked here;
    the free remote estimate rejects a bad body before anything is submitted.
    """
    validate_slug(slug)
    if end_image_url:
        raise ValueError(
            "end frames are not mapped for raw slugs (the field name varies by model); "
            "upload the image and pass its URL through extra under the model's own key"
        )
    args: dict[str, object] = {}
    if prompt and prompt.strip():
        args["prompt"] = prompt.strip()
    if duration is not None:
        _check_duration(duration, 1, 60, slug)
        args["duration"] = duration
    if aspect_ratio is not None:
        args["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        args["resolution"] = resolution
    if audio is not None:
        args["generate_audio"] = audio
    if image_url:
        args["image_url"] = image_url
    if extra:
        overlap = sorted(set(extra) & set(args))
        if overlap:
            raise ValueError(f"extra repeats {', '.join(overlap)}, already set by the dedicated arguments")
        args.update(extra)
    return slug, args
