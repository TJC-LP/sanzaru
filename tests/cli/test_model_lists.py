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


@pytest.mark.unit
def test_audio_chat_models_match_the_constants():
    """The gpt-4o audio-preview models were retired while the CLI still defaulted to one."""
    assert cli_audio._AUDIO_CHAT_MODELS == constants.AUDIO_CHAT_MODELS
    assert cli_audio._AUDIO_CHAT_MODELS[0] == constants.DEFAULT_AUDIO_CHAT_MODEL


@pytest.mark.unit
def test_video_model_choices_track_the_curated_table():
    """`--model` accepts every curated id (and the default) plus any catalog slug."""
    from sanzaru.cli import video as cli_video
    from sanzaru.video_models import DEFAULT_VIDEO_MODEL, VIDEO_MODEL_IDS

    for model_id in (*VIDEO_MODEL_IDS, DEFAULT_VIDEO_MODEL, "vendor/model/text-to-video"):
        assert cli_video._MODEL.convert(model_id, None, None) == model_id
