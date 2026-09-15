# SPDX-License-Identifier: MIT
"""Unit tests for the audio format allowlist and its ffmpeg demuxer mapping."""

import pytest

from sanzaru.audio.constants import (
    AUDIO_DEMUXER_BY_EXTENSION,
    DECODABLE_AUDIO_EXTENSIONS,
    SAFE_AUDIO_EXTENSIONS,
    safe_audio_format,
)

# The comma-separated aliases `ffmpeg -demuxers` prints for the demuxers we
# rely on. An entry outside this set is one ffmpeg answers "Unknown input
# format" to, which is exactly the failure the map exists to prevent.
FFMPEG_DEMUXER_ALIASES = {
    "aac",
    "aiff",
    "amr",
    "asf",
    "au",
    "caf",
    "flac",
    "matroska",
    "webm",
    "mov",
    "mp4",
    "m4a",
    "3gp",
    "3g2",
    "mj2",
    "mp3",
    "mpeg",
    "ogg",
    "wav",
}


@pytest.mark.unit
class TestSafeAudioFormatReturnsWhatFfmpegAccepts:
    """`format=` wants a demuxer name; seven allowlisted extensions are not one."""

    @pytest.mark.parametrize(
        "ext,demuxer",
        [
            ("wma", "asf"),
            ("mka", "matroska"),
            ("aif", "aiff"),
            ("aifc", "aiff"),
            ("m4b", "mov"),
            ("mpga", "mp3"),
            ("oga", "ogg"),
            ("opus", "ogg"),
        ],
    )
    def test_extensions_that_are_not_demuxer_names_are_mapped(self, ext, demuxer):
        assert safe_audio_format(ext) == demuxer
        assert safe_audio_format(f".{ext}") == demuxer
        assert safe_audio_format(f"track.{ext.upper()}") == demuxer

    @pytest.mark.parametrize("ext", ["mp3", "wav", "flac", "ogg", "aac", "m4a", "mp4", "webm", "aiff"])
    def test_extensions_that_already_are_demuxer_names_pass_through(self, ext):
        assert safe_audio_format(ext) == ext

    def test_every_mapped_value_is_a_real_ffmpeg_demuxer_alias(self):
        unknown = {ext: fmt for ext, fmt in AUDIO_DEMUXER_BY_EXTENSION.items() if fmt not in FFMPEG_DEMUXER_ALIASES}
        assert not unknown, f"ffmpeg has no demuxer named: {unknown}"

    def test_decodable_set_is_the_map_key_set(self):
        assert frozenset(AUDIO_DEMUXER_BY_EXTENSION) == DECODABLE_AUDIO_EXTENSIONS

    def test_safe_output_extensions_are_all_decodable(self):
        """An output this server creates must be something it can also read back."""
        assert SAFE_AUDIO_EXTENSIONS <= DECODABLE_AUDIO_EXTENSIONS


@pytest.mark.unit
class TestSafeAudioFormatRefusesPlaylistDemuxers:
    """pydub picks the demuxer from `format=`; playlist demuxers read other local files (CWE-610)."""

    @pytest.mark.parametrize("name", ["evil.hls", "evil.concat", "evil.m3u8", "evil.dash", "evil.txt", "evil"])
    def test_anything_outside_the_allowlist_is_refused(self, name):
        with pytest.raises(ValueError, match="unsupported audio format"):
            safe_audio_format(name)

    def test_a_custom_allowlist_narrows_but_still_maps(self):
        assert safe_audio_format("out.mpga", allowed=SAFE_AUDIO_EXTENSIONS) == "mp3"
        with pytest.raises(ValueError, match="unsupported audio format 'wma'"):
            safe_audio_format("out.wma", allowed=SAFE_AUDIO_EXTENSIONS)
