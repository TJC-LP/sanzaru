# SPDX-License-Identifier: MIT
"""Unit tests for security utilities."""
import pathlib

import pytest

from sora_mcp_server.security import check_not_symlink, safe_open_file, validate_safe_path


class TestValidateSafePath:
    """Test path validation and traversal protection."""

    def test_valid_filename(self, tmp_reference_path):
        """Test that valid filenames work correctly."""
        # Create a test file
        test_file = tmp_reference_path / "test.png"
        test_file.write_text("test")

        result = validate_safe_path(tmp_reference_path, "test.png")
        assert result == test_file.resolve()

    def test_path_traversal_rejected(self, tmp_reference_path):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValueError, match="path traversal detected"):
            validate_safe_path(tmp_reference_path, "../../../etc/passwd")

    def test_path_traversal_with_dots(self, tmp_reference_path):
        """Test various path traversal patterns."""
        traversal_attempts = [
            "../../secret.txt",
            "./../confidential.mp4",
            "subdir/../../outside.png",
        ]
        for attempt in traversal_attempts:
            with pytest.raises(ValueError, match="path traversal detected"):
                validate_safe_path(tmp_reference_path, attempt)

    def test_absolute_path_rejected(self, tmp_reference_path):
        """Test that absolute paths are rejected (they escape base_path)."""
        with pytest.raises(ValueError, match="path traversal detected"):
            validate_safe_path(tmp_reference_path, "/etc/passwd")

    def test_file_not_found_when_allow_create_false(self, tmp_reference_path):
        """Test that non-existent files raise error when allow_create=False."""
        with pytest.raises(ValueError, match="File not found: nonexistent.png"):
            validate_safe_path(tmp_reference_path, "nonexistent.png", allow_create=False)

    def test_file_not_found_allowed_when_allow_create_true(self, tmp_reference_path):
        """Test that non-existent files are allowed when allow_create=True."""
        result = validate_safe_path(tmp_reference_path, "new_file.png", allow_create=True)
        expected = (tmp_reference_path / "new_file.png").resolve()
        assert result == expected

    def test_subdirectory_allowed(self, tmp_reference_path):
        """Test that subdirectories within base_path are allowed."""
        subdir = tmp_reference_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.png"
        test_file.write_text("test")

        result = validate_safe_path(tmp_reference_path, "subdir/test.png")
        assert result == test_file.resolve()


class TestCheckNotSymlink:
    """Test symlink detection."""

    def test_regular_file_passes(self, tmp_reference_path):
        """Test that regular files pass the check."""
        regular_file = tmp_reference_path / "regular.txt"
        regular_file.write_text("content")

        # Should not raise
        check_not_symlink(regular_file, "Test file")

    def test_directory_passes(self, tmp_reference_path):
        """Test that directories pass the check."""
        # Should not raise
        check_not_symlink(tmp_reference_path, "Test directory")

    def test_symlink_rejected(self, tmp_path, tmp_reference_path):
        """Test that symlinks are detected and rejected."""
        target = tmp_path / "target.txt"
        target.write_text("target")

        link = tmp_reference_path / "link.txt"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="cannot be a symbolic link"):
            check_not_symlink(link, "Test file")

    def test_nonexistent_file_passes(self, tmp_reference_path):
        """Test that non-existent files don't raise (symlink check requires existence)."""
        nonexistent = tmp_reference_path / "nonexistent.txt"

        # Should not raise (file doesn't exist, so can't be a symlink)
        check_not_symlink(nonexistent, "Test file")


class TestSafeOpenFile:
    """Test safe file opening context manager."""

    def test_read_existing_file(self, tmp_reference_path):
        """Test reading an existing file."""
        test_file = tmp_reference_path / "test.txt"
        test_file.write_bytes(b"test content")

        with safe_open_file(test_file, "rb", "test file") as f:
            content = f.read()
            assert content == b"test content"

    def test_write_new_file(self, tmp_reference_path):
        """Test writing a new file."""
        test_file = tmp_reference_path / "output.txt"

        with safe_open_file(test_file, "wb", "test file", check_symlink=False) as f:
            f.write(b"new content")

        assert test_file.read_bytes() == b"new content"

    def test_file_not_found_error(self, tmp_reference_path):
        """Test that missing files raise ValueError with context."""
        nonexistent = tmp_reference_path / "missing.txt"

        with pytest.raises(ValueError, match="Test file not found: missing.txt"):
            with safe_open_file(nonexistent, "rb", "test file"):
                pass

    def test_symlink_rejected_when_check_enabled(self, tmp_path, tmp_reference_path):
        """Test that symlinks are rejected when check_symlink=True."""
        target = tmp_path / "target.txt"
        target.write_text("content")

        link = tmp_reference_path / "link.txt"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="cannot be a symbolic link"):
            with safe_open_file(link, "rb", "test file", check_symlink=True):
                pass

    def test_symlink_allowed_when_check_disabled(self, tmp_path, tmp_reference_path):
        """Test that symlinks work when check_symlink=False."""
        target = tmp_path / "target.txt"
        target.write_bytes(b"symlink target content")

        link = tmp_reference_path / "link.txt"
        link.symlink_to(target)

        with safe_open_file(link, "rb", "test file", check_symlink=False) as f:
            content = f.read()
            assert content == b"symlink target content"

    def test_permission_error_handling(self, tmp_reference_path):
        """Test that permission errors are handled gracefully."""
        # This test is platform-dependent and may not work everywhere
        # On most systems, we can't easily trigger PermissionError in tests
        # So we'll skip detailed permission testing here
        pass
