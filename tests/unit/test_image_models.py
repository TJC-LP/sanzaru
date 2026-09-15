"""image_models: the per-model capability table the image tools consult."""

import pytest

from sanzaru.image_models import (
    GPT_IMAGE_2_5_MODELS,
    capabilities_for,
    check_background,
    check_quality,
    honors_input_fidelity,
)


@pytest.mark.unit
class TestCapabilitiesLookup:
    @pytest.mark.parametrize(
        ("model", "family"),
        [
            ("gpt-image-2.5-sunburst", "gpt-image-2.5"),
            ("gpt-image-2.5-flare", "gpt-image-2.5"),
            ("gpt-image-2.5-sunburst-2026-09-08", "gpt-image-2.5"),
            ("gpt-image-2", "gpt-image-2"),
            ("gpt-image-2-2026-04-21", "gpt-image-2"),
            ("gpt-image-1.5", "gpt-image-1.5"),
            ("gpt-image-1", "gpt-image-1"),
            ("gpt-image-1-mini", "gpt-image-1-mini"),
        ],
    )
    def test_snapshots_resolve_to_their_family(self, model, family):
        caps = capabilities_for(model)
        assert caps is not None
        assert caps.family == family

    def test_gpt_image_1_5_is_not_mistaken_for_gpt_image_1(self):
        """Longest prefix wins: "gpt-image-1" is a prefix of "gpt-image-1.5"."""
        assert capabilities_for("gpt-image-1.5").family == "gpt-image-1.5"
        assert capabilities_for("gpt-image-1-mini").family == "gpt-image-1-mini"

    @pytest.mark.parametrize("model", ["dall-e-3", "dall-e-2", "gpt-image-9-nova"])
    def test_unknown_models_get_no_client_side_rules(self, model):
        assert capabilities_for(model) is None
        check_background(model, "transparent")
        check_quality(model, "max")
        assert honors_input_fidelity(model) is True

    def test_both_2_5_variants_share_one_capability_row(self):
        rows = {capabilities_for(m) for m in GPT_IMAGE_2_5_MODELS}
        assert len(rows) == 1
        (caps,) = rows
        assert caps.transparent_background and caps.input_fidelity and caps.arbitrary_resolutions
        assert {"xhigh", "max"} <= caps.qualities


@pytest.mark.unit
class TestBackgroundRule:
    @pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-2-2026-04-21"])
    def test_gpt_image_2_rejects_transparent(self, model):
        with pytest.raises(ValueError, match="gpt-image-2 does not support transparent"):
            check_background(model, "transparent")

    @pytest.mark.parametrize("model", ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-1.5", "gpt-image-1"])
    def test_transparent_capable_models_pass(self, model):
        check_background(model, "transparent")

    def test_the_error_names_a_model_that_can_do_it(self):
        with pytest.raises(ValueError, match="gpt-image-2.5"):
            check_background("gpt-image-2", "transparent")

    @pytest.mark.parametrize("background", ["auto", "opaque", None])
    def test_non_transparent_backgrounds_never_raise(self, background):
        check_background("gpt-image-2", background)


@pytest.mark.unit
class TestQualityRule:
    @pytest.mark.parametrize("quality", ["xhigh", "max"])
    def test_extended_levels_are_2_5_only(self, quality):
        check_quality("gpt-image-2.5-flare", quality)
        check_quality("gpt-image-2.5-sunburst-2026-09-08", quality)
        with pytest.raises(ValueError, match=f"gpt-image-2 does not support quality={quality!r}"):
            check_quality("gpt-image-2", quality)
        with pytest.raises(ValueError, match="gpt-image-1.5 does not support"):
            check_quality("gpt-image-1.5", quality)

    @pytest.mark.parametrize("quality", ["auto", "low", "medium", "high", None])
    def test_base_levels_pass_everywhere(self, quality):
        for model in ("gpt-image-2.5-flare", "gpt-image-2", "gpt-image-1.5", "gpt-image-1-mini"):
            check_quality(model, quality)

    def test_the_error_lists_the_accepted_values(self):
        with pytest.raises(ValueError, match="accepted: auto, high, low, medium"):
            check_quality("gpt-image-2", "max")


@pytest.mark.unit
class TestInputFidelityRule:
    def test_gpt_image_2_strips_it(self):
        assert honors_input_fidelity("gpt-image-2") is False
        assert honors_input_fidelity("gpt-image-2-2026-04-21") is False

    @pytest.mark.parametrize("model", ["gpt-image-2.5-sunburst", "gpt-image-2.5-flare", "gpt-image-1.5", "gpt-image-1"])
    def test_other_gpt_image_models_forward_it(self, model):
        assert honors_input_fidelity(model) is True

    def test_mini_never_supported_it(self):
        assert honors_input_fidelity("gpt-image-1-mini") is False
