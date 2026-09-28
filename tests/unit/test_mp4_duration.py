"""Reading a source clip's duration from its MP4 header, without ffprobe.

Edit/extend bill the source's seconds, so the duration is needed to price (and
cap) a job before its upload.
"""

import shutil
import struct
import subprocess

import pytest

from sanzaru.higgsfield.media import content_type_for, mp4_duration_seconds

pytestmark = pytest.mark.unit


def _atom(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _mvhd_v0(timescale: int, duration: int) -> bytes:
    # version+flags, creation, modification, timescale, duration, then padding
    return _atom(b"mvhd", b"\x00\x00\x00\x00" + struct.pack(">IIII", 0, 0, timescale, duration) + b"\x00" * 80)


def _mvhd_v1(timescale: int, duration: int) -> bytes:
    return _atom(b"mvhd", b"\x01\x00\x00\x00" + struct.pack(">QQIQ", 0, 0, timescale, duration) + b"\x00" * 80)


FTYP = _atom(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")


def test_mvhd_v0():
    data = FTYP + _atom(b"free", b"") + _atom(b"moov", _mvhd_v0(1000, 5500))
    assert mp4_duration_seconds(data) == pytest.approx(5.5)


def test_mvhd_v1():
    data = FTYP + _atom(b"moov", _mvhd_v1(90000, 90000 * 12))
    assert mp4_duration_seconds(data) == pytest.approx(12.0)


def test_mdat_before_moov_with_64_bit_size():
    payload = b"\x00" * 32
    mdat = struct.pack(">I4sQ", 1, b"mdat", 16 + len(payload)) + payload
    data = FTYP + mdat + _atom(b"moov", _mvhd_v0(600, 2400))
    assert mp4_duration_seconds(data) == pytest.approx(4.0)


def test_size_zero_runs_to_end_of_file():
    moov_payload = _mvhd_v0(1000, 3000)
    data = FTYP + struct.pack(">I4s", 0, b"moov") + moov_payload
    assert mp4_duration_seconds(data) == pytest.approx(3.0)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not an mp4 at all",
        FTYP,  # no moov
        FTYP + _atom(b"moov", _mvhd_v0(1000, 5000))[:20],  # truncated
        FTYP + _atom(b"moov", _mvhd_v0(0, 5000)),  # zero timescale
    ],
)
def test_unreadable_is_none(data):
    assert mp4_duration_seconds(data) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_mp4(tmp_path):
    out = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=3", "-r", "10", str(out)],
        check=True,
    )
    # The movie header includes the encoder's edit-list delay (x264 here reports
    # 3.2 s for 30 frames at 10 fps), so it can run slightly long. For pricing
    # that errs in the safe direction: a cap sees a small over-estimate.
    assert mp4_duration_seconds(out.read_bytes()) == pytest.approx(3.0, abs=0.3)


class TestContentTypes:
    @pytest.mark.parametrize(
        ("name", "expected"), [("a.PNG", "image/png"), ("a.jpeg", "image/jpeg"), ("a.webp", "image/webp")]
    )
    def test_images(self, name, expected):
        assert content_type_for(name, "image") == expected

    def test_video_mp4_only(self):
        assert content_type_for("clip.mp4", "video") == "video/mp4"
        with pytest.raises(ValueError, match="MP4"):
            content_type_for("clip.mov", "video")

    def test_bad_image(self):
        with pytest.raises(ValueError, match="JPEG"):
            content_type_for("a.bmp", "image")
