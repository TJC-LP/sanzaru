# SPDX-License-Identifier: MIT
"""The CLI spells its model lists out as literals so `sanzaru --help` never
imports openai (tests/cli/test_root.py). That makes drift silent: when
eleven_v4 was added to the constants, `--model eleven_v4` stayed a usage error.
"""

import pytest

from sanzaru.audio import constants
from sanzaru.cli import audio as cli_audio


@pytest.mark.unit
def test_elevenlabs_models_match_the_constants():
    assert cli_audio._ELEVENLABS_MODELS == constants.ELEVENLABS_MODELS


@pytest.mark.unit
def test_elevenlabs_default_matches_the_constants():
    assert cli_audio._ELEVENLABS_DEFAULT == constants.DEFAULT_ELEVENLABS_MODEL


@pytest.mark.unit
def test_openai_tts_models_match_the_constants():
    assert cli_audio._TTS_MODELS == constants.OPENAI_TTS_MODELS
