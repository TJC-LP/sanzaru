# SPDX-License-Identifier: MIT
"""Media resource templates: the attach surface, and the costs it refuses to pay.

The assertions worth keeping honest are the performance ones, because they are
invisible at runtime: that a listing is reused inside its TTL, that the cache is
partitioned by identity, and that an oversized file is refused from its metadata
without transferring any bytes.
"""

import pytest

from sanzaru import media_resources
from sanzaru.media_resources import (
    COMPLETION_LIMIT,
    LISTING_TTL_SECONDS,
    MAX_RESOURCE_BYTES,
    complete_media_filename_for,
    content_type_for,
    list_media,
    media_for_template_uri,
    read_media,
    template_uri,
)
from sanzaru.storage.protocol import FileInfo
from sanzaru.user_context import UserContext, reset_user_context, set_user_context

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _clean_cache():
    media_resources.clear_listing_cache()
    yield
    media_resources.clear_listing_cache()


def _info(name: str, *, size: int = 10, modified: float = 0.0) -> FileInfo:
    return FileInfo(name=name, size_bytes=size, modified_timestamp=modified)


@pytest.fixture
def storage(mocker):
    backend = mocker.MagicMock()
    backend.list_files = mocker.AsyncMock(return_value=[])
    backend.stat = mocker.AsyncMock(return_value=_info("a.png"))
    backend.read = mocker.AsyncMock(return_value=b"bytes")
    mocker.patch("sanzaru.media_resources.get_storage", return_value=backend)
    return backend


@pytest.mark.unit
class TestContentTypes:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a.png", "image/png"),
            ("a.JPG", "image/jpeg"),
            ("a.mp4", "video/mp4"),
            ("a.mp3", "audio/mpeg"),
            ("a.txt", None),
            ("a.html", None),
            ("noextension", None),
            ("a.json", None),
        ],
    )
    def test_allowlist_decides_and_is_case_insensitive(self, name, expected):
        assert content_type_for(name) == expected

    def test_uri_helpers_round_trip(self):
        assert template_uri("image") == "sanzaru://image/{filename}"
        assert media_for_template_uri("sanzaru://image/{filename}") == "image"
        assert media_for_template_uri("sanzaru://video/clip.mp4") == "video"
        assert media_for_template_uri("sanzaru://nope/x") is None
        assert media_for_template_uri("ui://sanzaru/media-viewer.html") is None
        assert media_for_template_uri("file:///etc/passwd") is None


@pytest.mark.integration
class TestListing:
    async def test_drops_files_the_read_path_would_refuse(self, storage):
        storage.list_files.return_value = [_info("keep.png"), _info("notes.txt"), _info("page.html")]

        names = [info.name for info in await list_media("image")]

        assert names == ["keep.png"]

    async def test_newest_first(self, storage):
        storage.list_files.return_value = [
            _info("old.png", modified=100.0),
            _info("new.png", modified=300.0),
            _info("mid.png", modified=200.0),
        ]

        assert [i.name for i in await list_media("image")] == ["new.png", "mid.png", "old.png"]

    async def test_reuses_the_listing_inside_its_ttl(self, storage):
        """Completion fires per keystroke; each one must not be a storage call."""
        storage.list_files.return_value = [_info("a.png")]

        for _ in range(5):
            await list_media("image")

        storage.list_files.assert_awaited_once()

    async def test_refetches_once_the_ttl_has_passed(self, storage, mocker):
        storage.list_files.return_value = [_info("a.png")]
        clock = mocker.patch("sanzaru.media_resources.time.monotonic", return_value=0.0)

        await list_media("image")
        clock.return_value = LISTING_TTL_SECONDS + 0.1
        await list_media("image")

        assert storage.list_files.await_count == 2

    async def test_cache_is_partitioned_by_identity(self, storage):
        """A shared deployment resolves a different directory per caller."""
        storage.list_files.return_value = [_info("alice.png")]
        token = set_user_context(UserContext(email="alice@example.com"))
        try:
            assert [i.name for i in await list_media("image")] == ["alice.png"]
        finally:
            reset_user_context(token)

        storage.list_files.return_value = [_info("bob.png")]
        token = set_user_context(UserContext(email="bob@example.com"))
        try:
            # Would be alice.png if the key ignored identity.
            assert [i.name for i in await list_media("image")] == ["bob.png"]
        finally:
            reset_user_context(token)

    async def test_separate_media_types_do_not_share_an_entry(self, storage):
        storage.list_files.side_effect = [[_info("a.png")], [_info("b.mp4")]]

        assert [i.name for i in await list_media("image")] == ["a.png"]
        assert [i.name for i in await list_media("video")] == ["b.mp4"]

    async def test_a_failed_listing_is_an_empty_menu_not_an_error(self, storage):
        storage.list_files.side_effect = OSError("volume unreachable")

        assert await list_media("image") == ()

    async def test_use_cache_false_bypasses_the_cache(self, storage):
        storage.list_files.return_value = [_info("a.png")]

        await list_media("image")
        await list_media("image", use_cache=False)

        assert storage.list_files.await_count == 2


@pytest.mark.integration
class TestRead:
    async def test_returns_the_bytes(self, storage):
        storage.read.return_value = b"\x89PNG payload"

        assert await read_media("image", "a.png") == b"\x89PNG payload"
        storage.read.assert_awaited_once_with("reference", "a.png")

    async def test_refuses_an_oversized_file_without_transferring_it(self, storage):
        """The reason stat() is worth a round trip: no GET of a file we would refuse."""
        storage.stat.return_value = _info("huge.mp4", size=MAX_RESOURCE_BYTES + 1)

        with pytest.raises(ValueError, match="over the .* resource limit"):
            await read_media("video", "huge.mp4")

        storage.read.assert_not_called()

    async def test_allows_a_file_exactly_at_the_limit(self, storage):
        storage.stat.return_value = _info("edge.mp4", size=MAX_RESOURCE_BYTES)

        await read_media("video", "edge.mp4")

        storage.read.assert_awaited_once()

    async def test_refuses_an_extension_outside_the_allowlist_before_any_io(self, storage):
        with pytest.raises(ValueError, match="not an allowlisted media type"):
            await read_media("image", "notes.txt")

        storage.stat.assert_not_called()
        storage.read.assert_not_called()

    async def test_refuses_an_unknown_media_type(self, storage):
        with pytest.raises(ValueError, match="unknown media type"):
            await read_media("hologram", "a.png")

    async def test_a_missing_file_is_a_usage_error(self, storage):
        storage.stat.side_effect = FileNotFoundError("nope")

        with pytest.raises(ValueError, match="not found"):
            await read_media("image", "gone.png")


@pytest.mark.integration
class TestCompletion:
    async def test_prefix_and_substring_both_match_case_insensitively(self, storage):
        storage.list_files.return_value = [
            _info("poster_final.png"),
            _info("img_1789570535.png"),
            _info("Cat_Poster.png"),
        ]

        prefix = await complete_media_filename_for("image", "poster")
        # Generated names lead with a stem nobody chose, so the memorable part
        # is often in the middle — substring matching is what makes those findable.
        middle = await complete_media_filename_for("image", "1789570")

        assert "poster_final.png" in prefix.values
        assert "Cat_Poster.png" in prefix.values
        assert middle.values == ["img_1789570535.png"]

    async def test_empty_partial_offers_everything_available(self, storage):
        storage.list_files.return_value = [_info("a.png"), _info("b.png")]

        result = await complete_media_filename_for("image", "")

        assert set(result.values) == {"a.png", "b.png"}
        assert result.total == 2
        assert result.has_more is False

    async def test_caps_the_values_but_reports_the_true_total(self, storage):
        storage.list_files.return_value = [_info(f"img_{i}.png") for i in range(COMPLETION_LIMIT + 25)]

        result = await complete_media_filename_for("image", "img")

        assert len(result.values) == COMPLETION_LIMIT
        assert result.total == COMPLETION_LIMIT + 25
        assert result.has_more is True

    async def test_no_matches_is_an_empty_completion(self, storage):
        storage.list_files.return_value = [_info("a.png")]

        result = await complete_media_filename_for("image", "zzz")

        assert result.values == [] and result.total == 0

    async def test_unknown_media_type_completes_nothing(self, storage):
        result = await complete_media_filename_for("hologram", "")

        assert result.values == [] and result.total == 0
        storage.list_files.assert_not_called()


@pytest.mark.integration
class TestServerWiring:
    @pytest.fixture
    async def server(self, monkeypatch, tmp_path):
        import importlib
        import sys

        monkeypatch.setenv("SANZARU_MEDIA_PATH", str(tmp_path))
        module = importlib.reload(importlib.import_module("sanzaru.server"))
        try:
            yield module
        finally:
            sys.modules.pop("sanzaru.server", None)

    async def test_one_template_per_media_type_and_no_per_file_resources(self, server):
        templates = {t.uri_template: t for t in await server.mcp.list_resource_templates()}

        assert set(templates) == {
            "sanzaru://image/{filename}",
            "sanzaru://video/{filename}",
            "sanzaru://audio/{filename}",
        }
        # The static registry holds only the viewer's UI resource: media is never
        # enumerated into resources/list.
        assert [str(r.uri) for r in await server.mcp.list_resources()] == ["ui://sanzaru/media-viewer.html"]

    async def test_templates_declare_binary_and_a_user_audience(self, server):
        templates = {t.uri_template: t for t in await server.mcp.list_resource_templates()}
        template = templates["sanzaru://image/{filename}"]

        # Not text/plain, which is what leaving mime_type unset would produce.
        assert template.mime_type == "application/octet-stream"
        assert template.annotations is not None
        assert template.annotations.audience == ["user"]

    async def test_reading_a_template_returns_the_file_bytes(self, server, tmp_path):
        images = tmp_path / "images"
        images.mkdir(exist_ok=True)
        (images / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")

        contents = list(await server.mcp.read_resource("sanzaru://image/pic.png"))

        assert len(contents) == 1
        assert contents[0].content == b"\x89PNG\r\n\x1a\nnot-a-real-png"
        assert contents[0].mime_type == "application/octet-stream"

    @pytest.mark.parametrize(
        "uri",
        [
            "sanzaru://image/../../../etc/passwd",
            "sanzaru://image/%2e%2e%2fsecret.png",
            "sanzaru://image/%2E%2E%5Csecret.png",
            "sanzaru://hologram/x.png",
        ],
    )
    async def test_traversal_and_unknown_media_do_not_resolve(self, server, uri):
        from mcp.server.mcpserver.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await server.mcp.read_resource(uri)
