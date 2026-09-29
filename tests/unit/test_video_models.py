"""The curated Higgsfield models and the translation into each one's schema.

Every Higgsfield schema is additionalProperties:false, so a key the endpoint
does not know is a rejected request. These pin the dialect differences and the
refusals that must happen before any network call.
"""

import pytest

from sanzaru.video_models import (
    DEFAULT_VIDEO_MODEL,
    VIDEO_MODEL_IDS,
    VIDEO_MODELS,
    build_arguments,
    build_raw_arguments,
    resolve_model,
)

pytestmark = pytest.mark.unit

SEEDANCE = VIDEO_MODELS["seedance-2.5"]
KLING = VIDEO_MODELS["kling-3.0-std"]
TURBO = VIDEO_MODELS["kling-3.0-turbo"]


def test_default_is_seedance():
    assert DEFAULT_VIDEO_MODEL == "seedance-2.5"
    assert DEFAULT_VIDEO_MODEL in VIDEO_MODEL_IDS


def test_resolve_model():
    assert resolve_model("kling-3.0-pro") is VIDEO_MODELS["kling-3.0-pro"]
    assert resolve_model("bytedance/seedance-2.0/text-to-video") is None
    with pytest.raises(ValueError, match="unknown video model"):
        resolve_model("sora-2")


class TestSeedance:
    def test_text_sends_only_what_was_set(self):
        slug, args = build_arguments(SEEDANCE, "text", prompt="  waves  ")
        assert slug == "bytedance/seedance-2.5/text-to-video"
        assert args == {"prompt": "waves"}

    def test_text_full(self):
        _, args = build_arguments(
            SEEDANCE, "text", prompt="waves", duration=8, aspect_ratio="9:16", resolution="480p", audio=False
        )
        assert args == {
            "prompt": "waves",
            "duration": 8,
            "aspect_ratio": "9:16",
            "resolution": "480p",
            "generate_audio": False,
        }

    def test_image_with_end_frame(self):
        slug, args = build_arguments(SEEDANCE, "image", image_url="https://x/a.png", end_image_url="https://x/b.png")
        assert slug.endswith("image-to-video")
        assert args == {"image_url": "https://x/a.png", "end_image_url": "https://x/b.png"}

    def test_image_prompt_is_optional(self):
        build_arguments(SEEDANCE, "image", image_url="https://x/a.png")

    def test_edit_refuses_duration(self):
        with pytest.raises(ValueError, match="derived from the source"):
            build_arguments(SEEDANCE, "edit", prompt="p", video_url="https://x/v.mp4", duration=5)

    def test_extend_takes_duration_and_references(self):
        slug, args = build_arguments(
            SEEDANCE,
            "extend",
            prompt="p",
            video_url="https://x/v.mp4",
            duration=6,
            reference_image_urls=["https://x/r.png"],
        )
        assert slug.endswith("video-extend")
        assert args == {"prompt": "p", "duration": 6, "video_url": "https://x/v.mp4", "image_urls": ["https://x/r.png"]}

    @pytest.mark.parametrize("op", ["image", "edit", "extend"])
    def test_aspect_ratio_only_for_text(self, op):
        kwargs = {"image_url": "https://x/a.png"} if op == "image" else {"video_url": "https://x/v.mp4"}
        with pytest.raises(ValueError, match="prepare_reference_image"):
            build_arguments(SEEDANCE, op, prompt="p", aspect_ratio="16:9", **kwargs)

    @pytest.mark.parametrize("duration", [3, 31, 5.0, True])
    def test_duration_range_and_type(self, duration):
        with pytest.raises(ValueError, match="duration"):
            build_arguments(SEEDANCE, "text", prompt="p", duration=duration)

    def test_resolution_enum(self):
        with pytest.raises(ValueError, match="1080p"):
            build_arguments(SEEDANCE, "text", prompt="p", resolution="1080p")

    @pytest.mark.parametrize("op", ["text", "edit", "extend"])
    def test_prompt_required(self, op):
        kwargs = {} if op == "text" else {"video_url": "https://x/v.mp4"}
        with pytest.raises(ValueError, match="non-empty prompt"):
            build_arguments(SEEDANCE, op, prompt="   ", **kwargs)

    def test_image_op_needs_image(self):
        with pytest.raises(ValueError, match="reference image"):
            build_arguments(SEEDANCE, "image", prompt="p")

    def test_end_image_needs_image_op(self):
        with pytest.raises(ValueError, match="end image"):
            build_arguments(SEEDANCE, "text", prompt="p", end_image_url="https://x/b.png")

    def test_edit_needs_source(self):
        with pytest.raises(ValueError, match="source video"):
            build_arguments(SEEDANCE, "edit", prompt="p")

    def test_extra_allowlist(self):
        _, args = build_arguments(SEEDANCE, "text", prompt="p", extra={"bitrate_mode": "standard"})
        assert args["bitrate_mode"] == "standard"
        with pytest.raises(ValueError, match="not accepted"):
            build_arguments(SEEDANCE, "image", image_url="https://x/a.png", extra={"output_format": "mov"})

    def test_extra_cannot_smuggle_managed_keys(self):
        with pytest.raises(ValueError, match="dedicated arguments"):
            build_arguments(SEEDANCE, "text", prompt="p", extra={"duration": 99})

    def test_references_refused_where_unsupported(self):
        with pytest.raises(ValueError, match="reference images"):
            build_arguments(SEEDANCE, "text", prompt="p", reference_image_urls=["https://x/r.png"])


class TestKling:
    def test_audio_is_sound_on_off(self):
        _, on = build_arguments(KLING, "text", prompt="p", audio=True)
        _, off = build_arguments(KLING, "text", prompt="p", audio=False)
        assert on["sound"] == "on" and off["sound"] == "off"
        assert "generate_audio" not in on

    def test_end_frame_is_last_image_url(self):
        slug, args = build_arguments(KLING, "image", prompt="p", image_url="https://x/a", end_image_url="https://x/b")
        assert slug == "kling-video/v3.0/std/image-to-video"
        assert args["last_image_url"] == "https://x/b" and "end_image_url" not in args

    def test_resolution_is_tier_fixed(self):
        with pytest.raises(ValueError, match="fixed resolution"):
            build_arguments(KLING, "text", prompt="p", resolution="720p")

    def test_aspect_and_duration_limits(self):
        with pytest.raises(ValueError, match="aspect_ratio"):
            build_arguments(KLING, "text", prompt="p", aspect_ratio="4:3")
        with pytest.raises(ValueError, match="between 3 and 15"):
            build_arguments(KLING, "text", prompt="p", duration=20)

    def test_prompt_length(self):
        with pytest.raises(ValueError, match="at most 2500"):
            build_arguments(KLING, "text", prompt="x" * 2501)

    def test_no_edit(self):
        with pytest.raises(ValueError, match="does not support edit"):
            build_arguments(KLING, "edit", prompt="p", video_url="https://x/v.mp4")

    def test_extra_keys(self):
        _, args = build_arguments(KLING, "text", prompt="p", extra={"cfg_scale": 0.7})
        assert args["cfg_scale"] == 0.7

    @pytest.mark.parametrize("tier", ["std", "pro", "4k"])
    def test_tier_slugs(self, tier):
        slug, _ = build_arguments(VIDEO_MODELS[f"kling-3.0-{tier}"], "text", prompt="p")
        assert slug == f"kling-video/v3.0/{tier}/text-to-video"


class TestTurbo:
    def test_unverified_controls_are_refused(self):
        with pytest.raises(ValueError, match="audio"):
            build_arguments(TURBO, "text", prompt="p", audio=True)
        with pytest.raises(ValueError, match="end frame"):
            build_arguments(TURBO, "image", prompt="p", image_url="https://x/a", end_image_url="https://x/b")

    def test_basic_text(self):
        slug, args = build_arguments(TURBO, "text", prompt="p", duration=5)
        assert slug == "kling-video/v3.0-turbo/text-to-video"
        assert args == {"prompt": "p", "duration": 5}


class TestRaw:
    def test_common_names_and_extra(self):
        slug, args = build_raw_arguments(
            "minimax/h3/text-to-video", prompt="p", duration=6, audio=True, extra={"resolution_mode": "x"}
        )
        assert slug == "minimax/h3/text-to-video"
        assert args == {"prompt": "p", "duration": 6, "generate_audio": True, "resolution_mode": "x"}

    @pytest.mark.parametrize("slug", ["requests/abc/cancel", "files/generate-upload-url", "a//b", "../x", "UPPER/x"])
    def test_slug_guard(self, slug):
        with pytest.raises(ValueError):
            build_raw_arguments(slug, prompt="p")

    def test_end_frame_not_mapped(self):
        with pytest.raises(ValueError, match="extra"):
            build_raw_arguments("wan/v2.7/image-to-video", image_url="https://x/a", end_image_url="https://x/b")

    def test_extra_overlap_refused(self):
        with pytest.raises(ValueError, match="repeats"):
            build_raw_arguments("wan/v2.7/text-to-video", prompt="p", extra={"prompt": "q"})
