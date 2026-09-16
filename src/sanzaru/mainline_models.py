# SPDX-License-Identifier: MIT
"""The mainline (text) models sanzaru lets a caller pick, kept import-light.

`create_image` drives the Responses API image_generation tool through a
mainline model: the image model renders, this one reads the prompt, calls the
tool and revises the prompt. Its tokens bill on top of the image, but an image
turn is a few hundred tokens, so even the flagship adds cents — the choice is
exposed so a caller can trade cost for judgement rather than to save money.

Lives outside config.py because the CLI shows these as `--model` choices at
import time, and `sanzaru --help` must not pay for the openai import that
config.py carries (enforced by the startup-weight test).

Verified against the account's model list on 2026-09-15; all four support the
image_generation tool per OpenAI's model pages.
"""

from typing import Literal

MainlineModel = Literal["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
"""gpt-6-astra: flagship ($10/$50 per 1M in/out). gpt-5.6-sol: the "gpt-5.6"
alias, professional work ($4/$20). gpt-5.6-terra: balanced ($2/$12).
gpt-5.6-luna: cost-optimised ($0.20/$1.20)."""

MAINLINE_MODELS: tuple[MainlineModel, ...] = ("gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
DEFAULT_MAINLINE_MODEL: MainlineModel = "gpt-6-astra"
