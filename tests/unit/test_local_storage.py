# SPDX-License-Identifier: MIT
"""Unit tests for LocalStorageBackend."""

import pathlib

import pytest

from sanzaru.storage.local import LocalStorageBackend
from sanzaru.storage.protocol import FileInfo, StorageBackend

# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


def test_local_storage_is_storage_backend():
    """LocalStorageBackend satisfies the StorageBackend runtime protocol."""
    backend = LocalStorageBackend(path_overrides={"video": pathlib.Path("/tmp")})
    assert isinstance(backend, StorageBackend)


# ------------------------------------------------------------------
# read
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_read_existing_file(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    (ref / "hero.png").write_bytes(b"PNG_BYTES")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    data = await backend.read("reference", "hero.png")
    assert data == b"PNG_BYTES"


@pytest.mark.unit
async def test_read_nonexistent_file(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    with pytest.raises(ValueError, match="File not found"):
        await backend.read("reference", "nope.png")


@pytest.mark.unit
async def test_read_path_traversal(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    with pytest.raises(ValueError, match="path traversal"):
        await backend.read("reference", "../../etc/passwd")


@pytest.mark.unit
async def test_read_rejects_symlink(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    real = ref / "real.png"
    real.write_bytes(b"REAL")
    link = ref / "link.png"
    link.symlink_to(real)

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    with pytest.raises(ValueError, match="symbolic link"):
        await backend.read("reference", "link.png")


# ------------------------------------------------------------------
# write
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_write_creates_file(tmp_path):
    vid = tmp_path / "vids"
    vid.mkdir()

    backend = LocalStorageBackend(path_overrides={"video": vid})
    display = await backend.write("video", "out.mp4", b"VIDEO_DATA")

    assert (vid / "out.mp4").read_bytes() == b"VIDEO_DATA"
    assert "out.mp4" in display


@pytest.mark.unit
async def test_write_path_traversal(tmp_path):
    vid = tmp_path / "vids"
    vid.mkdir()

    backend = LocalStorageBackend(path_overrides={"video": vid})
    with pytest.raises(ValueError, match="path traversal"):
        await backend.write("video", "../escape.mp4", b"BAD")


# ------------------------------------------------------------------
# write_stream
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_write_stream(tmp_path):
    vid = tmp_path / "vids"
    vid.mkdir()

    async def chunks():
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    backend = LocalStorageBackend(path_overrides={"video": vid})
    display = await backend.write_stream("video", "streamed.mp4", chunks())

    assert (vid / "streamed.mp4").read_bytes() == b"chunk1chunk2chunk3"
    assert "streamed.mp4" in display


# ------------------------------------------------------------------
# list_files
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_list_files_basic(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    (ref / "a.png").write_bytes(b"A")
    (ref / "b.jpg").write_bytes(b"BB")
    (ref / "c.txt").write_bytes(b"CCC")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    files = await backend.list_files("reference")

    names = {f.name for f in files}
    assert names == {"a.png", "b.jpg", "c.txt"}
    assert all(isinstance(f, FileInfo) for f in files)


@pytest.mark.unit
async def test_list_files_with_extension_filter(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    (ref / "a.png").write_bytes(b"A")
    (ref / "b.jpg").write_bytes(b"B")
    (ref / "c.txt").write_bytes(b"C")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    files = await backend.list_files("reference", extensions={".png", ".jpg"})

    names = {f.name for f in files}
    assert names == {"a.png", "b.jpg"}


@pytest.mark.unit
async def test_list_files_with_glob_pattern(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    (ref / "cat_01.png").write_bytes(b"A")
    (ref / "cat_02.png").write_bytes(b"B")
    (ref / "dog_01.png").write_bytes(b"C")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    files = await backend.list_files("reference", pattern="cat*")

    names = {f.name for f in files}
    assert names == {"cat_01.png", "cat_02.png"}


# ------------------------------------------------------------------
# stat
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_stat(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    (ref / "img.png").write_bytes(b"12345")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    info = await backend.stat("reference", "img.png")

    assert info.name == "img.png"
    assert info.size_bytes == 5
    assert info.modified_timestamp > 0


@pytest.mark.unit
async def test_stat_nonexistent(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    with pytest.raises(ValueError, match="File not found"):
        await backend.stat("reference", "nope.png")


# ------------------------------------------------------------------
# exists
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_exists_true(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    (ref / "img.png").write_bytes(b"X")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    assert await backend.exists("reference", "img.png") is True


@pytest.mark.unit
async def test_exists_false(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    assert await backend.exists("reference", "nope.png") is False


@pytest.mark.unit
async def test_exists_symlink_returns_false(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    real = ref / "real.png"
    real.write_bytes(b"REAL")
    link = ref / "link.png"
    link.symlink_to(real)

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    assert await backend.exists("reference", "link.png") is False


@pytest.mark.unit
async def test_exists_traversal_returns_false(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    assert await backend.exists("reference", "../../etc/passwd") is False


# ------------------------------------------------------------------
# local_path / local_tempfile
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_local_path_yields_real_path(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()
    real = ref / "img.png"
    real.write_bytes(b"DATA")

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    async with backend.local_path("reference", "img.png") as p:
        assert isinstance(p, pathlib.Path)
        assert p == real.resolve()
        assert p.read_bytes() == b"DATA"


@pytest.mark.unit
async def test_local_tempfile_yields_dest_path(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    async with backend.local_tempfile("reference", "new.png") as p:
        assert isinstance(p, pathlib.Path)
        p.write_bytes(b"WRITTEN")

    assert (ref / "new.png").read_bytes() == b"WRITTEN"


# ------------------------------------------------------------------
# resolve_display_path
# ------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_display_path(tmp_path):
    ref = tmp_path / "refs"
    ref.mkdir()

    backend = LocalStorageBackend(path_overrides={"reference": ref})
    display = backend.resolve_display_path("reference", "img.png")

    assert display.endswith("img.png")
    assert str(ref) in display
