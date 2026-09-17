# SPDX-License-Identifier: MIT
"""The inspection tools: what the model actually receives when it looks at media.

Two things here are worth more than the rest. `visual_tokens` is checked against
the published per-size table rather than against itself, so a change to the patch
size fails against the real contract. And the MCP registration test asserts an
`ImageContent` block comes back, because the return annotation is the only thing
standing between a picture and a base64 string dumped as JSON text.
"""

import importlib
import io
import shutil
import subprocess
import sys

import pytest
from mcp.server.mcpserver.utilities.types import Image
from PIL import Image as PILImage

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.tools.inspect import (
    DECODABLE_VIDEO_EXTENSIONS,
    DEFAULT_LONG_EDGE,
    MAX_FRAMES,
    MAX_IMAGE_BYTES,
    MAX_LONG_EDGE,
    inspect_image,
    inspect_video_frame,
    safe_video_demuxer,
    visual_tokens,
)

pytestmark = pytest.mark.anyio

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH",
)


def _noise(width: int, height: int) -> PILImage.Image:
    """An image that does not compress, so the size ladder is actually exercised."""
    import os

    return PILImage.frombytes("RGB", (width, height), os.urandom(width * height * 3))


def _blocks(result: list[Image | str]) -> tuple[list[Image], list[str]]:
    return (
        [block for block in result if isinstance(block, Image)],
        [block for block in result if isinstance(block, str)],
    )


def _decode(image: Image) -> PILImage.Image:
    assert image.data is not None
    return PILImage.open(io.BytesIO(image.data))


@pytest.fixture
def image_storage(mocker, tmp_reference_path):
    mocker.patch(
        "sanzaru.tools.inspect.get_storage",
        return_value=LocalStorageBackend(path_overrides={"reference": tmp_reference_path}),
    )
    return tmp_reference_path


@pytest.fixture
def video_storage(mocker, tmp_video_path):
    mocker.patch(
        "sanzaru.tools.inspect.get_storage",
        return_value=LocalStorageBackend(path_overrides={"video": tmp_video_path}),
    )
    return tmp_video_path


@pytest.mark.unit
class TestVisualTokens:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            ((200, 200), 64),
            ((1000, 1000), 1296),
            ((1092, 1092), 1521),
            ((1456, 819), 1560),
            ((2576, 1449), 4784),
        ],
    )
    def test_matches_the_published_cost_table(self, size, expected):
        """Pinned to the documented per-size costs, not to our own arithmetic."""
        assert visual_tokens(*size) == expected

    def test_default_long_edge_stays_under_the_standard_tier_budget(self):
        # A square at the default is the worst case, and still affordable.
        assert visual_tokens(DEFAULT_LONG_EDGE, DEFAULT_LONG_EDGE) < 3300


@pytest.mark.unit
class TestVideoDemuxerAllowlist:
    def test_maps_extensions_onto_real_demuxers(self):
        assert safe_video_demuxer("clip.mp4") == "mp4"
        assert safe_video_demuxer("clip.webm") == "matroska"
        assert safe_video_demuxer("MOV") == "mov"
        assert safe_video_demuxer(".m4v") == "mp4"

    @pytest.mark.parametrize("bad", ["playlist.hls", "evil.concat", "list.m3u8", "clip.dash", "track.wav"])
    def test_refuses_playlist_and_non_video_containers(self, bad):
        """The whole point: content must never choose the demuxer (CWE-22 via concat/hls)."""
        with pytest.raises(ValueError, match="unsupported video format"):
            safe_video_demuxer(bad)

    def test_allowlist_excludes_every_playlist_demuxer(self):
        assert DECODABLE_VIDEO_EXTENSIONS.isdisjoint({"hls", "concat", "dash", "m3u8"})


@pytest.mark.integration
class TestInspectImage:
    async def test_returns_an_image_then_a_note(self, image_storage):
        PILImage.new("RGB", (640, 480), (10, 20, 30)).save(image_storage / "flat.png")

        result = await inspect_image("flat.png")

        images, notes = _blocks(result)
        assert len(images) == 1 and len(notes) == 1
        assert _decode(images[0]).size == (640, 480)
        assert "640x480" in notes[0]
        assert "visual tokens" in notes[0]

    async def test_downscales_to_the_long_edge_and_says_so(self, image_storage):
        PILImage.new("RGB", (4000, 2000), (200, 100, 50)).save(image_storage / "big.png")

        images, notes = _blocks(await inspect_image("big.png"))

        assert max(_decode(images[0]).size) == DEFAULT_LONG_EDGE
        assert _decode(images[0]).size == (1568, 784)  # aspect ratio preserved
        assert "4000x2000" in notes[0] and "shown at 1568x784" in notes[0]

    async def test_never_upscales(self, image_storage):
        PILImage.new("RGB", (64, 64)).save(image_storage / "tiny.png")

        images, notes = _blocks(await inspect_image("tiny.png", max_dimension=2000))

        assert _decode(images[0]).size == (64, 64)
        assert "full size" in notes[0]

    async def test_region_crops_in_source_coordinates(self, image_storage):
        # Left half red, right half blue: the crop proves which pixels arrived.
        img = PILImage.new("RGB", (400, 200), (255, 0, 0))
        img.paste(PILImage.new("RGB", (200, 200), (0, 0, 255)), (200, 0))
        img.save(image_storage / "halves.png")

        images, notes = _blocks(await inspect_image("halves.png", region=[200, 0, 400, 200]))

        cropped = _decode(images[0]).convert("RGB")
        assert cropped.size == (200, 200)
        assert cropped.getpixel((100, 100)) == (0, 0, 255)
        assert "cropped to [200, 0, 400, 200]" in notes[0]

    @pytest.mark.parametrize(
        ("region", "match"),
        [
            ([0, 0, 10], "left, top, right, bottom"),
            ([10, 0, 5, 20], "right > left"),
            ([0, 0, 500, 20], "outside"),
            ([-5, 0, 10, 20], "outside"),
        ],
    )
    async def test_bad_regions_are_usage_errors(self, image_storage, region, match):
        PILImage.new("RGB", (100, 100)).save(image_storage / "x.png")

        with pytest.raises(ValueError, match=match):
            await inspect_image("x.png", region=region)

    @pytest.mark.parametrize("dimension", [0, -1, MAX_LONG_EDGE + 1])
    async def test_max_dimension_is_bounded(self, image_storage, dimension):
        PILImage.new("RGB", (10, 10)).save(image_storage / "x.png")

        with pytest.raises(ValueError, match="max_dimension must be between"):
            await inspect_image("x.png", max_dimension=dimension)

    async def test_falls_off_the_format_ladder_to_stay_under_the_cap(self, image_storage):
        """Incompressible pixels at full size: PNG cannot fit, so the format changes."""
        _noise(2576, 2576).save(image_storage / "noise.png")

        images, notes = _blocks(await inspect_image("noise.png", max_dimension=MAX_LONG_EDGE))

        assert images[0].data is not None
        assert len(images[0].data) <= MAX_IMAGE_BYTES
        assert images[0]._mime_type != "image/png"
        assert "re-encoded to stay within" in notes[0]

    async def test_honours_a_requested_format_that_fits(self, image_storage):
        PILImage.new("RGB", (200, 200), (7, 7, 7)).save(image_storage / "s.png")

        images, _ = _blocks(await inspect_image("s.png", image_format="jpeg"))

        assert images[0]._mime_type == "image/jpeg"

    async def test_transparency_survives_the_default_format(self, image_storage):
        PILImage.new("RGBA", (50, 50), (255, 0, 0, 0)).save(image_storage / "alpha.png")

        images, _ = _blocks(await inspect_image("alpha.png"))

        assert _decode(images[0]).mode in ("RGBA", "LA", "P")

    async def test_alpha_is_flattened_onto_white_for_jpeg(self, image_storage):
        PILImage.new("RGBA", (50, 50), (0, 0, 0, 0)).save(image_storage / "clear.png")

        images, _ = _blocks(await inspect_image("clear.png", image_format="jpeg"))

        # White, not the black a dropped alpha channel would leave behind.
        assert _decode(images[0]).convert("RGB").getpixel((25, 25)) == (255, 255, 255)

    async def test_missing_and_unreadable_files_are_usage_errors(self, image_storage):
        (image_storage / "notanimage.png").write_text("plain text")

        with pytest.raises(ValueError):
            await inspect_image("absent.png")
        with pytest.raises(ValueError, match="Not a readable image"):
            await inspect_image("notanimage.png")


@ffmpeg_required
@pytest.mark.integration
class TestInspectVideoFrame:
    @staticmethod
    def _make_clip(path, *, seconds: int = 4) -> None:
        """A real mp4 whose colour changes over time, so frame order is checkable."""
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"color=c=red:s=320x240:d={seconds}",
                "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={seconds}",
                "-filter_complex", f"[0][1]blend=all_expr='A*(1-T/{seconds})+B*(T/{seconds})'",
                "-pix_fmt", "yuv420p", "-r", "10", str(path),
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip

    async def test_samples_evenly_spaced_frames_in_time_order(self, video_storage):
        self._make_clip(video_storage / "clip.mp4")

        result = await inspect_video_frame("clip.mp4")

        images, notes = _blocks(result)
        assert len(images) == 3 and len(notes) == 3
        # Interleaved image-then-note, and ascending timestamps.
        assert isinstance(result[0], Image) and isinstance(result[1], str)
        offsets = [float(note.split("@ ")[1].split("s")[0]) for note in notes]
        assert offsets == sorted(offsets)
        # The clip fades red -> blue, so blue must increase across the samples.
        blues = [_decode(img).convert("RGB").getpixel((160, 120))[2] for img in images]
        assert blues[0] < blues[-1]

    async def test_explicit_timestamps_win_over_the_frame_count(self, video_storage):
        self._make_clip(video_storage / "clip.mp4")

        images, notes = _blocks(await inspect_video_frame("clip.mp4", frames=8, timestamps=[0.5, 2.0]))

        assert len(images) == 2
        assert "@ 0.50s" in notes[0] and "@ 2.00s" in notes[1]

    async def test_frames_are_downscaled_to_the_requested_edge(self, video_storage):
        self._make_clip(video_storage / "clip.mp4")

        images, _ = _blocks(await inspect_video_frame("clip.mp4", frames=1, max_dimension=64))

        assert max(_decode(images[0]).size) == 64

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"frames": 0}, "frames must be between"),
            ({"frames": MAX_FRAMES + 1}, "frames must be between"),
            ({"timestamps": []}, "at least one value"),
            ({"timestamps": [-1.0]}, "must not be negative"),
            ({"timestamps": [99.0]}, "under the clip duration"),
            ({"timestamps": [0.0] * (MAX_FRAMES + 1)}, "at most"),
        ],
    )
    async def test_argument_validation(self, video_storage, kwargs, match):
        self._make_clip(video_storage / "clip.mp4")

        with pytest.raises(ValueError, match=match):
            await inspect_video_frame("clip.mp4", **kwargs)

    async def test_refuses_a_container_outside_the_allowlist_before_touching_ffmpeg(self, video_storage):
        (video_storage / "evil.concat").write_text("file /etc/passwd\n")

        with pytest.raises(ValueError, match="unsupported video format"):
            await inspect_video_frame("evil.concat")

    async def test_a_non_video_file_fails_as_a_usage_error(self, video_storage):
        (video_storage / "fake.mp4").write_text("not actually an mp4")

        with pytest.raises(ValueError, match="ffprobe|ffmpeg"):
            await inspect_video_frame("fake.mp4")


@pytest.mark.integration
class TestMcpRegistration:
    @pytest.fixture
    async def server(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
        module = importlib.reload(importlib.import_module("sanzaru.server"))
        try:
            yield module
        finally:
            sys.modules.pop("sanzaru.server", None)

    async def test_both_tools_are_registered_with_the_expected_arguments(self, server):
        schemas = {tool.name: tool.input_schema for tool in await server.mcp.list_tools()}

        assert set(schemas["inspect_image"]["properties"]) == {
            "filename",
            "max_dimension",
            "region",
            "image_format",
        }
        assert schemas["inspect_image"]["required"] == ["filename"]
        assert schemas["inspect_image"]["properties"]["max_dimension"]["default"] == DEFAULT_LONG_EDGE
        assert set(schemas["inspect_video_frame"]["properties"]) == {
            "filename",
            "frames",
            "timestamps",
            "max_dimension",
        }

    async def test_the_tools_advertise_no_output_schema_so_content_blocks_survive(self, server):
        """A declared output schema would make the SDK serialize the image as JSON."""
        tools = {tool.name: tool for tool in await server.mcp.list_tools()}

        assert tools["inspect_image"].output_schema is None
        assert tools["inspect_video_frame"].output_schema is None

    async def test_calling_inspect_image_yields_real_image_content(self, server, mocker, tmp_path):
        """The end that matters: the model receives an image block, not text."""
        images = tmp_path / "images"
        images.mkdir(exist_ok=True)
        PILImage.new("RGB", (120, 90), (1, 2, 3)).save(images / "pic.png")
        mocker.patch(
            "sanzaru.tools.inspect.get_storage",
            return_value=LocalStorageBackend(path_overrides={"reference": images}),
        )

        result = await server.mcp.call_tool("inspect_image", {"filename": "pic.png"}, mocker.MagicMock())

        kinds = [block.type for block in result.content]
        assert kinds == ["image", "text"]
        assert result.content[0].mime_type == "image/png"
        assert result.structured_content is None
        assert "120x90" in result.content[1].text
