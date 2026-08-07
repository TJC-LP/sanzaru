# SPDX-License-Identifier: MIT
"""Integration tests for `sanzaru audio` and `sanzaru podcast` (tool layer mocked)."""

import json

import pytest
from click.testing import CliRunner

from sanzaru.cli import cli

pytest.importorskip("pydub", reason="audio CLI tests exercise modules that import pydub")

from sanzaru.audio.models import AudioProcessingResult, ChatResult, TranscriptionResult, TTSResult  # noqa: E402
from sanzaru.tools.podcast import PodcastResult  # noqa: E402


@pytest.mark.integration
def test_audio_transcribe_single_file(mocker):
    transcribe = mocker.patch(
        "sanzaru.tools.audio.transcribe_audio",
        mocker.AsyncMock(return_value=TranscriptionResult(text="hello world")),
    )

    result = CliRunner().invoke(cli, ["audio", "transcribe", "meeting.mp3"])

    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["result"]["text"] == "hello world"
    assert parsed["input"] == {"index": 0, "file": "meeting.mp3"}
    assert transcribe.call_args.kwargs["input_file_name"] == "meeting.mp3"


@pytest.mark.integration
def test_audio_transcribe_enhance_routes_to_enhancement(mocker):
    enhanced = mocker.patch(
        "sanzaru.tools.audio.transcribe_with_enhancement",
        mocker.AsyncMock(return_value=TranscriptionResult(text="professional text")),
    )
    plain = mocker.patch("sanzaru.tools.audio.transcribe_audio", mocker.AsyncMock())

    result = CliRunner().invoke(cli, ["audio", "transcribe", "talk.mp3", "--enhance", "professional"])

    assert result.exit_code == 0
    assert enhanced.call_args.kwargs["enhancement_type"] == "professional"
    plain.assert_not_called()


@pytest.mark.integration
def test_audio_transcribe_prompt_and_enhance_conflict():
    result = CliRunner().invoke(cli, ["audio", "transcribe", "talk.mp3", "--prompt", "names", "--enhance", "detailed"])

    assert result.exit_code == 2
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"


@pytest.mark.integration
def test_audio_transcribe_multi_file_partial_failure(mocker):
    async def fake_transcribe(input_file_name, **kwargs):
        if input_file_name == "bad.mp3":
            raise ValueError("unsupported codec")
        return TranscriptionResult(text=f"text of {input_file_name}")

    mocker.patch("sanzaru.tools.audio.transcribe_audio", fake_transcribe)

    result = CliRunner().invoke(cli, ["audio", "transcribe", "good.mp3", "bad.mp3"])

    assert result.exit_code == 6
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert len(lines) == 2
    failures = [line for line in lines if not line["ok"]]
    assert failures[0]["input"]["file"] == "bad.mp3"


@pytest.mark.integration
def test_audio_speak_writes_to_output(mocker, tmp_path):
    async def fake_tts(text_prompt, model, voice, instructions, speed, output_file_name, provider, voice_settings):
        (tmp_path / output_file_name).write_bytes(b"mp3")
        return TTSResult(output_file=output_file_name)

    tts = mocker.patch("sanzaru.tools.audio.create_audio", mocker.AsyncMock(side_effect=fake_tts))

    result = CliRunner().invoke(
        cli, ["audio", "speak", "hello there", "--voice", "nova", "-o", str(tmp_path / "greeting.mp3")]
    )

    assert result.exit_code == 0, result.stderr
    assert tts.call_args.kwargs["output_file_name"] == "greeting.mp3"
    assert tts.call_args.kwargs["voice"] == "nova"
    parsed = json.loads(result.stdout)
    assert parsed["result"]["file"]["path"] == str(tmp_path / "greeting.mp3")
    assert (tmp_path / "greeting.mp3").exists()


@pytest.mark.integration
def test_audio_convert_moves_output_across_dirs(mocker, tmp_path):
    """Input pins the audio dir; -o elsewhere triggers the tmp-write + move path."""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    src = in_dir / "raw.wav"
    src.write_bytes(b"wav")

    async def fake_convert(input_file_name, target_format, output_file_name):
        (in_dir / output_file_name).write_bytes(b"converted")
        return AudioProcessingResult(output_file=output_file_name)

    convert = mocker.patch("sanzaru.tools.audio.convert_audio", mocker.AsyncMock(side_effect=fake_convert))

    result = CliRunner().invoke(cli, ["audio", "convert", str(src), "-o", str(out_dir / "final.mp3")])

    assert result.exit_code == 0, result.stderr
    assert convert.call_args.kwargs["output_file_name"] == "final__sanzaru_tmp.mp3"
    parsed = json.loads(result.stdout)
    assert parsed["result"]["file"]["path"] == str(out_dir / "final.mp3")
    assert (out_dir / "final.mp3").read_bytes() == b"converted"
    assert not (in_dir / "final__sanzaru_tmp.mp3").exists()


@pytest.mark.integration
def test_audio_files_latest(mocker):
    from sanzaru.audio.models import FilePathSupportParams

    mocker.patch(
        "sanzaru.tools.audio.get_latest_audio",
        mocker.AsyncMock(
            return_value=FilePathSupportParams(file_name="newest.mp3", modified_time=100.0, size_bytes=10, format="mp3")
        ),
    )

    result = CliRunner().invoke(cli, ["audio", "files", "--latest"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"]["file_name"] == "newest.mp3"


@pytest.mark.integration
def test_audio_chat(mocker):
    chat = mocker.patch(
        "sanzaru.tools.audio.chat_with_audio",
        mocker.AsyncMock(return_value=ChatResult(text="two speakers")),
    )

    result = CliRunner().invoke(cli, ["audio", "chat", "call.mp3", "--prompt", "how many speakers?"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"]["text"] == "two speakers"
    assert chat.call_args.kwargs["user_prompt"] == "how many speakers?"


@pytest.mark.integration
def test_podcast_generate_from_stdin_script(mocker, tmp_path):
    script = {
        "title": "Test Cast",
        "speakers": [{"id": "h", "name": "Host", "voice": "nova", "speed": 1.0, "instructions": "warm"}],
        "segments": [{"speaker": "h", "text": "Welcome!"}],
        "config": {},
    }

    async def fake_podcast(parsed_script, model, provider, filename=None):
        (tmp_path / "podcast_1.mp3").write_bytes(b"audio")
        return PodcastResult(
            output_file="podcast_1.mp3",
            title=parsed_script["title"],
            segment_count=1,
            estimated_duration_seconds=3.2,
            speakers=["Host"],
            transcript="Host: Welcome!",
        )

    generate = mocker.patch("sanzaru.tools.podcast.generate_podcast", mocker.AsyncMock(side_effect=fake_podcast))

    result = CliRunner().invoke(
        cli,
        ["podcast", "generate", "-", "-o", str(tmp_path / "episode.mp3")],
        input=json.dumps(script),
    )

    assert result.exit_code == 0, result.stderr
    assert generate.call_args.args[0]["title"] == "Test Cast"
    parsed = json.loads(result.stdout)
    assert parsed["result"]["file"]["path"] == str(tmp_path / "episode.mp3")
    assert (tmp_path / "episode.mp3").read_bytes() == b"audio"


@pytest.mark.integration
def test_podcast_render_mode_flag_beats_config(mocker, tmp_path):
    """--render-mode is an override, so it wins over a render_mode already in
    the script's config rather than deferring to it."""
    script = {
        "title": "Test Cast",
        "speakers": [{"id": "h", "name": "Host", "voice": "nova", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "h", "text": "Welcome!"}],
        "config": {"default_pause_ms": 500, "normalize_loudness": True, "output_format": "mp3"},
    }
    script["config"]["render_mode"] = "segments"

    async def fake_podcast(parsed_script, model, provider, filename=None):
        return PodcastResult(
            output_file="ep.mp3",
            title="Test Cast",
            segment_count=1,
            estimated_duration_seconds=3.2,
            speakers=["Host"],
            transcript="Host: Welcome!",
        )

    generate = mocker.patch("sanzaru.tools.podcast.generate_podcast", mocker.AsyncMock(side_effect=fake_podcast))
    mocker.patch("sanzaru.cli.podcast.finalize_output", mocker.AsyncMock(return_value=str(tmp_path / "ep.mp3")))

    result = CliRunner().invoke(
        cli,
        ["podcast", "generate", "-", "--render-mode", "dialogue"],
        input=json.dumps(script),
    )

    assert result.exit_code == 0, result.stderr
    assert generate.call_args.args[0]["config"]["render_mode"] == "dialogue"


@pytest.mark.integration
def test_podcast_render_mode_flag_leaves_config_alone_when_absent(mocker, tmp_path):
    """Without the flag the script's own render_mode must survive untouched."""
    script = {
        "title": "Test Cast",
        "speakers": [{"id": "h", "name": "Host", "voice": "nova", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "h", "text": "Welcome!"}],
        "config": {
            "default_pause_ms": 500,
            "normalize_loudness": True,
            "output_format": "mp3",
            "render_mode": "dialogue",
        },
    }

    async def fake_podcast(parsed_script, model, provider, filename=None):
        return PodcastResult(
            output_file="ep.mp3",
            title="Test Cast",
            segment_count=1,
            estimated_duration_seconds=3.2,
            speakers=["Host"],
            transcript="Host: Welcome!",
        )

    generate = mocker.patch("sanzaru.tools.podcast.generate_podcast", mocker.AsyncMock(side_effect=fake_podcast))
    mocker.patch("sanzaru.cli.podcast.finalize_output", mocker.AsyncMock(return_value=str(tmp_path / "ep.mp3")))

    result = CliRunner().invoke(cli, ["podcast", "generate", "-"], input=json.dumps(script))

    assert result.exit_code == 0, result.stderr
    assert generate.call_args.args[0]["config"]["render_mode"] == "dialogue"


@pytest.mark.integration
def test_podcast_render_mode_on_non_object_config_is_usage_error():
    """Merging the flag into a config that is not an object must be a usage
    error, not a TypeError surfacing as an internal failure."""
    script = {"title": "T", "speakers": [], "segments": [], "config": []}

    result = CliRunner().invoke(
        cli,
        ["podcast", "generate", "-", "--render-mode", "dialogue"],
        input=json.dumps(script),
    )

    assert result.exit_code == 2, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"
    assert "'config' must be a JSON object" in parsed["error"]["message"]


@pytest.mark.integration
def test_podcast_render_mode_applies_to_a_script_without_a_config(mocker, tmp_path):
    """`config` became optional in #36, so the flag now has to create one.

    It used to be merged only into an existing config, because fabricating one
    would have masked the "missing required field: 'config'" diagnostic. That
    error is gone — withholding the merge would now drop the flag in silence.
    """
    script = {
        "title": "T",
        "speakers": [{"name": "Host", "voice": "nova"}],
        "segments": [{"speaker": "Host", "text": "Hi."}],
    }

    async def fake_podcast(parsed_script, model, provider, filename=None):
        return PodcastResult(
            output_file="ep.mp3",
            title="T",
            segment_count=1,
            estimated_duration_seconds=1.0,
            speakers=["Host"],
            transcript="Host: Hi.",
        )

    generate = mocker.patch("sanzaru.tools.podcast.generate_podcast", mocker.AsyncMock(side_effect=fake_podcast))
    mocker.patch("sanzaru.cli.podcast.finalize_output", mocker.AsyncMock(return_value=str(tmp_path / "ep.mp3")))

    result = CliRunner().invoke(
        cli,
        ["podcast", "generate", "-", "--render-mode", "dialogue"],
        input=json.dumps(script),
    )

    assert result.exit_code == 0, result.stderr
    assert generate.call_args.args[0]["config"]["render_mode"] == "dialogue"


@pytest.mark.integration
def test_podcast_generate_invalid_json_is_usage_error():
    result = CliRunner().invoke(cli, ["podcast", "generate", "{not json"])

    assert result.exit_code == 2
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"
    assert "not valid JSON" in parsed["error"]["message"]


# ==================== PROVIDER SELECTION ====================


@pytest.mark.integration
def test_audio_speak_elevenlabs_passes_provider_and_settings(mocker, tmp_path):
    tts = mocker.patch(
        "sanzaru.tools.audio.create_audio",
        mocker.AsyncMock(return_value=TTSResult(output_file="out.mp3")),
    )
    (tmp_path / "out.mp3").write_bytes(b"mp3")

    result = CliRunner().invoke(
        cli,
        [
            "audio",
            "speak",
            "hello",
            "--provider",
            "elevenlabs",
            "--voice",
            "voice_abc",
            "--voice-settings",
            '{"stability": 0.5}',
            "-o",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    kwargs = tts.call_args.kwargs
    assert kwargs["provider"] == "elevenlabs"
    assert kwargs["voice"] == "voice_abc"
    assert kwargs["voice_settings"] == {"stability": 0.5}
    # eleven_v3 is the ElevenLabs default, not the OpenAI one.
    assert kwargs["model"] == "eleven_v3"


@pytest.mark.integration
def test_audio_speak_defaults_are_unchanged(mocker, tmp_path):
    tts = mocker.patch(
        "sanzaru.tools.audio.create_audio",
        mocker.AsyncMock(return_value=TTSResult(output_file="out.mp3")),
    )
    (tmp_path / "out.mp3").write_bytes(b"mp3")

    result = CliRunner().invoke(cli, ["audio", "speak", "hello", "-o", str(tmp_path)])

    assert result.exit_code == 0, result.stderr
    kwargs = tts.call_args.kwargs
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-4o-mini-tts"
    assert kwargs["voice"] == "alloy"
    assert kwargs["voice_settings"] is None


@pytest.mark.integration
def test_audio_speak_elevenlabs_requires_a_voice():
    result = CliRunner().invoke(cli, ["audio", "speak", "hello", "--provider", "elevenlabs"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["type"] == "usage"


@pytest.mark.integration
def test_audio_speak_rejects_elevenlabs_voice_on_openai():
    result = CliRunner().invoke(cli, ["audio", "speak", "hello", "--voice", "voice_abc"])

    assert result.exit_code == 2
    assert "not an OpenAI voice" in json.loads(result.stdout)["error"]["message"]


@pytest.mark.integration
def test_audio_speak_rejects_cross_provider_model():
    result = CliRunner().invoke(
        cli, ["audio", "speak", "hello", "--provider", "elevenlabs", "--voice", "v1", "--model", "tts-1"]
    )

    assert result.exit_code == 2
    assert "not valid for --provider elevenlabs" in json.loads(result.stdout)["error"]["message"]


@pytest.mark.integration
def test_audio_speak_rejects_malformed_voice_settings():
    result = CliRunner().invoke(cli, ["audio", "speak", "hello", "--voice-settings", "not-json"])

    assert result.exit_code == 2
    assert "not valid JSON" in json.loads(result.stdout)["error"]["message"]


@pytest.mark.integration
def test_audio_speak_rejects_non_numeric_voice_setting(tmp_path):
    """A wrong value *type* is usage (exit 2), like every other --voice-settings
    complaint — it used to reach the provider's range check as a TypeError and
    surface as internal/exit 1. Unmocked on purpose: validation runs before any
    client is built, so nothing goes over the wire."""
    result = CliRunner().invoke(
        cli,
        [
            "audio",
            "speak",
            "hello",
            "--provider",
            "elevenlabs",
            "--voice",
            "voice_abc",
            "--voice-settings",
            '{"stability": "high"}',
            "-o",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"
    assert "must be a number" in parsed["error"]["message"]


@pytest.mark.integration
def test_audio_speak_rejects_non_object_voice_settings():
    result = CliRunner().invoke(cli, ["audio", "speak", "hello", "--voice-settings", "[1,2]"])

    assert result.exit_code == 2
    assert "must be a JSON object" in json.loads(result.stdout)["error"]["message"]


@pytest.mark.integration
def test_podcast_generate_passes_provider(mocker, tmp_path):
    script = {
        "title": "Test Cast",
        "speakers": [{"id": "h", "name": "Host", "voice": "v1", "speed": 1.0, "instructions": ""}],
        "segments": [{"speaker": "h", "text": "Welcome!"}],
        "config": {"default_pause_ms": 500, "normalize_loudness": True, "output_format": "mp3"},
    }

    async def fake_podcast(parsed_script, model, provider, filename=None):
        return PodcastResult(
            output_file="ep.mp3",
            title="Test Cast",
            segment_count=1,
            estimated_duration_seconds=3.2,
            speakers=["Host"],
            transcript="Host: Welcome!",
        )

    generate = mocker.patch("sanzaru.tools.podcast.generate_podcast", mocker.AsyncMock(side_effect=fake_podcast))
    mocker.patch("sanzaru.cli.podcast.finalize_output", mocker.AsyncMock(return_value=str(tmp_path / "ep.mp3")))

    result = CliRunner().invoke(cli, ["podcast", "generate", "-", "--provider", "elevenlabs"], input=json.dumps(script))

    assert result.exit_code == 0, result.stderr
    assert generate.call_args.kwargs["provider"] == "elevenlabs"
    assert generate.call_args.kwargs["model"] == "eleven_v3"


@pytest.mark.integration
def test_podcast_generate_rejects_cross_provider_model():
    result = CliRunner().invoke(cli, ["podcast", "generate", "{}", "--provider", "elevenlabs", "--model", "tts-1"])

    assert result.exit_code == 2
    assert "not valid for --provider elevenlabs" in json.loads(result.stdout)["error"]["message"]
