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
    async def fake_tts(text_prompt, model, voice, instructions, speed, output_file_name):
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

    async def fake_podcast(parsed_script, model):
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
def test_podcast_generate_invalid_json_is_usage_error():
    result = CliRunner().invoke(cli, ["podcast", "generate", "{not json"])

    assert result.exit_code == 2
    parsed = json.loads(result.stdout)
    assert parsed["error"]["type"] == "usage"
    assert "not valid JSON" in parsed["error"]["message"]
