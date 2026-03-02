"""Tests for utility functions module."""

from pathlib import Path

from clawbot.util.utils import ensure_directory, generate_session_id


class TestGenerateSessionId:
    """Tests for generate_session_id function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        session_id = generate_session_id()
        assert isinstance(session_id, str)

    def test_returns_hex_string(self) -> None:
        """Test that returned string is valid hex."""
        session_id = generate_session_id()
        # Should not raise
        int(session_id, 16)

    def test_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        ids = [generate_session_id() for _ in range(100)]
        # All should be unique
        assert len(ids) == len(set(ids))

    def test_id_length(self) -> None:
        """Test UUID hex length (32 characters for UUID4)."""
        session_id = generate_session_id()
        assert len(session_id) == 32

    def test_id_format(self) -> None:
        """Test that ID is lowercase hex."""
        session_id = generate_session_id()
        assert session_id == session_id.lower()


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_new_directory(self, tmp_path) -> None:
        """Test creating a new directory."""
        new_dir = tmp_path / "new_dir"
        result = ensure_directory(new_dir)
        assert result == new_dir
        assert new_dir.is_dir()

    def test_accepts_string_path(self, tmp_path) -> None:
        """Test that string paths are accepted."""
        new_dir = tmp_path / "string_dir"
        result = ensure_directory(str(new_dir))
        assert isinstance(result, Path)
        assert new_dir.is_dir()

    def test_existing_directory_no_op(self, tmp_path) -> None:
        """Test that existing directory is not modified."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        result = ensure_directory(existing_dir)
        assert result == existing_dir

    def test_creates_nested_directories(self, tmp_path) -> None:
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        result = ensure_directory(nested_dir)
        assert result == nested_dir
        assert nested_dir.is_dir()

    def test_raises_on_existing_file(self, tmp_path) -> None:
        """Test that existing file raises ValueError."""
        file_path = tmp_path / "a_file.txt"
        file_path.write_text("content")

        try:
            ensure_directory(file_path)
        except ValueError as e:
            assert "exists but is not a directory" in str(e)

    def test_expands_user_home(self) -> None:
        """Test that ~ is expanded to home directory."""

        # Path with ~ should be expanded
        result = ensure_directory("~/test_dir")
        # Should expand to actual home directory
        assert result == Path("~/test_dir").expanduser()
        assert result.is_dir()

        # Cleanup
        import shutil
        shutil.rmtree(result, ignore_errors=True)
