"""Tests for file system tools module."""

from pathlib import Path

import pytest

from clawbot.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _resolve_path,
)


class TestResolvePath:
    """Tests for _resolve_path helper function."""

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Test that absolute paths are returned as-is."""
        abs_path = tmp_path / "test.txt"
        abs_path.touch()
        result = _resolve_path(str(abs_path))
        assert result == abs_path.resolve()

    def test_relative_path_with_workspace(self, tmp_path: Path) -> None:
        """Test that relative paths are resolved against workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        rel_file = workspace / "file.txt"
        rel_file.touch()

        result = _resolve_path("file.txt", workspace=workspace)
        assert result == rel_file.resolve()

    def test_relative_path_without_workspace(self) -> None:
        """Test that relative paths without workspace use current dir."""
        result = _resolve_path("test.txt")
        assert result.is_absolute()

    def test_allowed_dir_restriction(self, tmp_path: Path) -> None:
        """Test that allowed_dir restricts access."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        with pytest.raises(PermissionError, match="outside allowed directory"):
            _resolve_path(str(outside / "file.txt"), allowed_dir=allowed)

    def test_allowed_dir_inside_permitted(self, tmp_path: Path) -> None:
        """Test that paths inside allowed_dir are permitted."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        inside_file = allowed / "subdir" / "file.txt"
        inside_file.parent.mkdir()
        inside_file.touch()

        result = _resolve_path(str(inside_file), allowed_dir=allowed)
        assert result == inside_file.resolve()

    def test_expands_user_path(self) -> None:
        """Test that ~ is expanded to home directory."""
        result = _resolve_path("~/test_file.txt")
        assert "~" not in str(result)


class TestReadFileTool:
    """Tests for ReadFileTool class."""

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        tool = ReadFileTool()
        assert tool.name == "read_file"

    def test_description_property(self) -> None:
        """Test that description property returns correct value."""
        tool = ReadFileTool()
        assert "Read" in tool.description

    def test_parameters_property(self) -> None:
        """Test that parameters property returns correct schema."""
        tool = ReadFileTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "path" in params["required"]

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading an existing file."""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!"
        test_file.write_text(content)

        tool = ReadFileTool()
        result = await tool.execute(path=str(test_file))
        assert result == content

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self) -> None:
        """Test reading a non-existent file."""
        tool = ReadFileTool()
        result = await tool.execute(path="/nonexistent/file.txt")
        assert "Error: File not found" in result

    @pytest.mark.asyncio
    async def test_read_directory_instead_of_file(self, tmp_path: Path) -> None:
        """Test reading a directory instead of a file."""
        tool = ReadFileTool()
        result = await tool.execute(path=str(tmp_path))
        assert "Error: Not a file" in result

    @pytest.mark.asyncio
    async def test_read_with_workspace(self, tmp_path: Path) -> None:
        """Test reading file with workspace resolution."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_file.write_text("workspace content")

        tool = ReadFileTool(workspace=workspace)
        result = await tool.execute(path="test.txt")
        assert result == "workspace content"

    @pytest.mark.asyncio
    async def test_read_outside_allowed_dir(self, tmp_path: Path) -> None:
        """Test reading file outside allowed directory."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        test_file = outside / "test.txt"
        test_file.write_text("outside content")

        tool = ReadFileTool(allowed_dir=allowed)
        result = await tool.execute(path=str(test_file))
        assert "Error:" in result


class TestWriteFileTool:
    """Tests for WriteFileTool class."""

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        tool = WriteFileTool()
        assert tool.name == "write_file"

    def test_description_property(self) -> None:
        """Test that description property returns correct value."""
        tool = WriteFileTool()
        assert "Write" in tool.description

    def test_parameters_property(self) -> None:
        """Test that parameters property returns correct schema."""
        tool = WriteFileTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "content" in params["properties"]
        assert "path" in params["required"]
        assert "content" in params["required"]

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path: Path) -> None:
        """Test writing to a new file."""
        test_file = tmp_path / "new.txt"
        content = "New content"

        tool = WriteFileTool()
        result = await tool.execute(path=str(test_file), content=content)

        assert "Successfully wrote" in result
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if needed."""
        nested_file = tmp_path / "a" / "b" / "c" / "file.txt"
        content = "Nested content"

        tool = WriteFileTool()
        await tool.execute(path=str(nested_file), content=content)

        assert nested_file.exists()
        assert nested_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that writing overwrites existing content."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Old content")
        new_content = "New content"

        tool = WriteFileTool()
        await tool.execute(path=str(test_file), content=new_content)

        assert test_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_with_workspace(self, tmp_path: Path) -> None:
        """Test writing file with workspace resolution."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        rel_file = workspace / "relative.txt"

        tool = WriteFileTool(workspace=workspace)
        await tool.execute(path="relative.txt", content="workspace content")

        assert rel_file.exists()
        assert rel_file.read_text() == "workspace content"

    @pytest.mark.asyncio
    async def test_write_outside_allowed_dir(self, tmp_path: Path) -> None:
        """Test writing file outside allowed directory."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside" / "file.txt"

        tool = WriteFileTool(allowed_dir=allowed)
        result = await tool.execute(path=str(outside), content="content")

        assert "Error:" in result


class TestEditFileTool:
    """Tests for EditFileTool class."""

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        tool = EditFileTool()
        assert tool.name == "edit_file"

    def test_description_property(self) -> None:
        """Test that description property returns correct value."""
        tool = EditFileTool()
        assert "Edit" in tool.description

    def test_parameters_property(self) -> None:
        """Test that parameters property returns correct schema."""
        tool = EditFileTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "old_text" in params["properties"]
        assert "new_text" in params["properties"]
        assert "path" in params["required"]
        assert "old_text" in params["required"]
        assert "new_text" in params["required"]

    @pytest.mark.asyncio
    async def test_edit_file_single_occurrence(self, tmp_path: Path) -> None:
        """Test editing a file with single occurrence of old_text."""
        test_file = tmp_path / "edit.txt"
        test_file.write_text("Hello World")

        tool = EditFileTool()
        result = await tool.execute(
            path=str(test_file),
            old_text="World",
            new_text="Universe"
        )

        assert "Successfully edited" in result
        assert test_file.read_text() == "Hello Universe"

    @pytest.mark.asyncio
    async def test_edit_file_multiple_occurrences(self, tmp_path: Path) -> None:
        """Test editing a file with multiple occurrences of old_text."""
        test_file = tmp_path / "edit.txt"
        test_file.write_text("test test test")

        tool = EditFileTool()
        result = await tool.execute(
            path=str(test_file),
            old_text="test",
            new_text="prod"
        )

        assert "Warning: old_text appears" in result
        assert "3 times" in result

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self, tmp_path: Path) -> None:
        """Test editing when old_text is not found."""
        test_file = tmp_path / "edit.txt"
        test_file.write_text("Some content here")

        tool = EditFileTool()
        result = await tool.execute(
            path=str(test_file),
            old_text="nonexistent",
            new_text="replacement"
        )

        assert "Error: old_text not found" in result

    @pytest.mark.asyncio
    async def test_edit_file_nonexistent_file(self) -> None:
        """Test editing a non-existent file."""
        tool = EditFileTool()
        result = await tool.execute(
            path="/nonexistent/file.txt",
            old_text="old",
            new_text="new"
        )

        assert "Error: File not found" in result

    @pytest.mark.asyncio
    async def test_edit_with_workspace(self, tmp_path: Path) -> None:
        """Test editing file with workspace resolution."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "edit.txt"
        test_file.write_text("original content")

        tool = EditFileTool(workspace=workspace)
        await tool.execute(
            path="edit.txt",
            old_text="original",
            new_text="modified"
        )

        assert test_file.read_text() == "modified content"

    @pytest.mark.asyncio
    async def test_edit_preserves_file_structure(self, tmp_path: Path) -> None:
        """Test that editing preserves file structure."""
        test_file = tmp_path / "structure.txt"
        original = "Line 1\nLine 2\nLine 3\n"
        test_file.write_text(original)

        tool = EditFileTool()
        await tool.execute(
            path=str(test_file),
            old_text="Line 2",
            new_text="Modified Line 2"
        )

        expected = "Line 1\nModified Line 2\nLine 3\n"
        assert test_file.read_text() == expected


class TestListDirTool:
    """Tests for ListDirTool class."""

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        tool = ListDirTool()
        assert tool.name == "list_dir"

    def test_description_property(self) -> None:
        """Test that description property returns correct value."""
        tool = ListDirTool()
        assert "List" in tool.description

    def test_parameters_property(self) -> None:
        """Test that parameters property returns correct schema."""
        tool = ListDirTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "path" in params["required"]

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_path: Path) -> None:
        """Test listing an empty directory."""
        tool = ListDirTool()
        result = await tool.execute(path=str(tmp_path))
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_list_directory_with_files(self, tmp_path: Path) -> None:
        """Test listing a directory with files."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()

        tool = ListDirTool()
        result = await tool.execute(path=str(tmp_path))

        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "📄" in result

    @pytest.mark.asyncio
    async def test_list_directory_with_subdirs(self, tmp_path: Path) -> None:
        """Test listing a directory with subdirectories."""
        (tmp_path / "subdir").mkdir()

        tool = ListDirTool()
        result = await tool.execute(path=str(tmp_path))

        assert "subdir" in result
        assert "📁" in result

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self) -> None:
        """Test listing a non-existent directory."""
        tool = ListDirTool()
        result = await tool.execute(path="/nonexistent/dir")
        assert "Error: Directory not found" in result

    @pytest.mark.asyncio
    async def test_list_file_instead_of_directory(self, tmp_path: Path) -> None:
        """Test listing a file instead of directory."""
        test_file = tmp_path / "file.txt"
        test_file.touch()

        tool = ListDirTool()
        result = await tool.execute(path=str(test_file))
        assert "Error: Not a directory" in result

    @pytest.mark.asyncio
    async def test_list_with_workspace(self, tmp_path: Path) -> None:
        """Test listing directory with workspace resolution."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").touch()

        tool = ListDirTool(workspace=workspace)
        result = await tool.execute(path=".")

        assert "file.txt" in result

    @pytest.mark.asyncio
    async def test_list_sorted_alphabetically(self, tmp_path: Path) -> None:
        """Test that directory listing is sorted."""
        (tmp_path / "zebra.txt").touch()
        (tmp_path / "apple.txt").touch()
        (tmp_path / "mango.txt").touch()

        tool = ListDirTool()
        result = await tool.execute(path=str(tmp_path))

        lines = result.split("\n")
        assert lines == sorted(lines)
