# SPDX-License-Identifier: MIT
"""What each GPT image model can do, in one place.

The image tools used to key every model-specific rule on `DEFAULT_IMAGE_MODEL`
("if the model is the default, reject transparent backgrounds"), which was
only true while the default was gpt-image-2. gpt-image-2.5 supports transparent
output and two extra quality levels, so the rules have to be looked up per
model. Everything here mirrors the pinned openai SDK's own parameter docstrings
(`openai/types/image_generate_params.py`, `image_edit_params.py`); when the SDK
is bumped, re-read those and update this table rather than the call sites.
"""

from dataclasses import dataclass
from typing import Literal

from openai.types import ImageModel

ImageQuality = Literal["auto", "low", "medium", "high", "xhigh", "max"]
"""Every quality the Images API accepts for a GPT image model. Which ones a
given model honours is `ImageCapabilities.qualities`."""

_BASE_QUALITIES: frozenset[str] = frozenset({"auto", "low", "medium", "high"})
_EXTENDED_QUALITIES: frozenset[str] = _BASE_QUALITIES | {"xhigh", "max"}


@dataclass(frozen=True, slots=True)
class ImageCapabilities:
    """The knobs one model family accepts."""

    family: str
    """Short label used in error messages ("gpt-image-2.5")."""
    transparent_background: bool
    """Whether `background="transparent"` is accepted (png/webp output only)."""
    input_fidelity: bool
    """Whether `input_fidelity` is honoured on edits. gpt-image-2 rejects the
    flag outright (it always processes inputs at high fidelity), so the
    wrappers strip it rather than forward it."""
    qualities: frozenset[str]
    arbitrary_resolutions: bool
    """Any WIDTHxHEIGHT with both edges multiples of 16, ratio within 3:1 and
    the model's pixel limits — as opposed to the three fixed GPT image sizes."""


_GPT_IMAGE_2_5 = ImageCapabilities(
    family="gpt-image-2.5",
    transparent_background=True,
    input_fidelity=True,
    qualities=_EXTENDED_QUALITIES,
    arbitrary_resolutions=True,
)
_GPT_IMAGE_2 = ImageCapabilities(
    family="gpt-image-2",
    transparent_background=False,
    input_fidelity=False,
    qualities=_BASE_QUALITIES,
    arbitrary_resolutions=True,
)
_GPT_IMAGE_1_5 = ImageCapabilities(
    family="gpt-image-1.5",
    transparent_background=True,
    input_fidelity=True,
    qualities=_BASE_QUALITIES,
    arbitrary_resolutions=False,
)
_GPT_IMAGE_1 = ImageCapabilities(
    family="gpt-image-1",
    transparent_background=True,
    input_fidelity=True,
    qualities=_BASE_QUALITIES,
    arbitrary_resolutions=False,
)
_GPT_IMAGE_1_MINI = ImageCapabilities(
    family="gpt-image-1-mini",
    transparent_background=True,
    input_fidelity=False,
    qualities=_BASE_QUALITIES,
    arbitrary_resolutions=False,
)

# Dated snapshots bill and behave like their base model. Longest prefix wins in
# `capabilities_for`, so "gpt-image-1.5" is not mistaken for "gpt-image-1".
_CAPABILITIES: dict[str, ImageCapabilities] = {
    "gpt-image-2.5-sunburst": _GPT_IMAGE_2_5,
    "gpt-image-2.5-flare": _GPT_IMAGE_2_5,
    "gpt-image-2": _GPT_IMAGE_2,
    "gpt-image-1.5": _GPT_IMAGE_1_5,
    "gpt-image-1-mini": _GPT_IMAGE_1_MINI,
    "gpt-image-1": _GPT_IMAGE_1,
    "chatgpt-image-latest": _GPT_IMAGE_2_5,
}

GPT_IMAGE_2_5_MODELS: tuple[ImageModel, ...] = ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")
"""The two gpt-image-2.5 variants (snapshot 2026-09-08). Sunburst is tuned for
editing precision, Flare for fast everyday generation; they are priced the same
as gpt-image-2."""

TRANSPARENT_CAPABLE_EXAMPLE: ImageModel = "gpt-image-2.5-flare"
"""Named in the transparent-background error so the message stays true after a
default change: it used to hard-code gpt-image-1.5."""


def capabilities_for(model: str) -> ImageCapabilities | None:
    """The capability row for `model`, or None for a model this table does not know.

    Unknown models (dall-e-*, or a GPT image model newer than this table) get no
    client-side validation at all: the API is the authority and its rejection is
    returned verbatim. Refusing an unknown name here would turn every SDK bump
    into a sanzaru release.
    """
    best = max((known for known in _CAPABILITIES if model.startswith(known)), key=len, default="")
    return _CAPABILITIES[best] if best else None


def check_background(model: str, background: str | None) -> None:
    """Raise before the request when `model` cannot produce the requested background."""
    if background != "transparent":
        return
    caps = capabilities_for(model)
    if caps is not None and not caps.transparent_background:
        raise ValueError(
            f"{caps.family} does not support transparent backgrounds. "
            f"Use a gpt-image-2.5 model (e.g. {TRANSPARENT_CAPABLE_EXAMPLE}) or gpt-image-1.5 for transparent output."
        )


def check_quality(model: str, quality: str | None) -> None:
    """Raise before the request when `quality` is outside what `model` accepts.

    `xhigh` and `max` exist only on gpt-image-2.5; on any other GPT image model
    the API rejects them, and a client-side check turns that into the CLI's
    usage exit code with the accepted values spelled out.
    """
    if quality is None:
        return
    caps = capabilities_for(model)
    if caps is not None and quality not in caps.qualities:
        raise ValueError(
            f"{caps.family} does not support quality={quality!r}; accepted: {', '.join(sorted(caps.qualities))}."
        )


def honors_input_fidelity(model: str) -> bool:
    """Whether `input_fidelity` should be forwarded on an edit for `model`.

    Unknown models get the flag forwarded — the API decides. Known models that
    reject it (gpt-image-2) have it stripped, because the request would fail
    for a parameter the caller may not even have set deliberately.
    """
    caps = capabilities_for(model)
    return caps is None or caps.input_fidelity
