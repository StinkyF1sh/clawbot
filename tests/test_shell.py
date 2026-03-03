"""Tests for shell execution tool module."""

import asyncio
import os
import pytest
from pathlib import Path

from clawbot.tools.shell import ExecTool


class TestExecToolInit:
    """Tests for ExecTool initialization."""

    def test_default_timeout(self) -> None:
        """Test default timeout value."""
        tool = ExecTool()
        assert tool.timeout == 60

    def test_custom_timeout(self) -> None:
        """Test custom timeout value."""
        tool = ExecTool(timeout=30)
        assert tool.timeout == 30

    def test_default_working_dir(self) -> None:
        """Test default working directory."""
        tool = ExecTool()
        assert tool.working_dir is None

    def test_custom_working_dir(self) -> None:
        """Test custom working directory."""
        tool = ExecTool(working_dir="/tmp")
        assert tool.working_dir == "/tmp"

    def test_default_deny_patterns(self) -> None:
        """Test default deny patterns are set."""
        tool = ExecTool()
        assert len(tool.deny_patterns) > 0
        assert isinstance(tool.deny_patterns, list)

    def test_custom_deny_patterns(self) -> None:
        """Test custom deny patterns."""
        custom = [r"custom_pattern"]
        tool = ExecTool(deny_patterns=custom)
        assert tool.deny_patterns == custom

    def test_default_allow_patterns(self) -> None:
        """Test default allow patterns is empty list."""
        tool = ExecTool()
        assert tool.allow_patterns == []

    def test_custom_allow_patterns(self) -> None:
        """Test custom allow patterns."""
        custom = [r"allowed_pattern"]
        tool = ExecTool(allow_patterns=custom)
        assert tool.allow_patterns == custom

    def test_default_restrict_to_workspace(self) -> None:
        """Test default restrict_to_workspace value."""
        tool = ExecTool()
        assert tool.restrict_to_workspace is False

    def test_custom_restrict_to_workspace(self) -> None:
        """Test custom restrict_to_workspace value."""
        tool = ExecTool(restrict_to_workspace=True)
        assert tool.restrict_to_workspace is True


class TestExecToolProperties:
    """Tests for ExecTool properties."""

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        tool = ExecTool()
        assert tool.name == "exec"

    def test_description_property(self) -> None:
        """Test that description property returns correct value."""
        tool = ExecTool()
        assert "Execute" in tool.description
        assert "shell" in tool.description

    def test_parameters_property(self) -> None:
        """Test that parameters property returns correct schema."""
        tool = ExecTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "command" in params["properties"]
        assert "working_dir" in params["properties"]
        assert "command" in params["required"]
        assert "working_dir" not in params["required"]


class TestExecToolExecute:
    """Tests for ExecTool.execute method."""

    @pytest.mark.asyncio
    async def test_execute_simple_command(self) -> None:
        """Test executing a simple command."""
        tool = ExecTool()
        result = await tool.execute(command="echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_with_output(self) -> None:
        """Test executing command with output."""
        tool = ExecTool()
        result = await tool.execute(command="echo test123")
        assert "test123" in result

    @pytest.mark.asyncio
    async def test_execute_with_working_dir(self, tmp_path: Path) -> None:
        """Test executing command with custom working directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        tool = ExecTool()
        result = await tool.execute(command="dir", working_dir=str(test_dir))

        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_execute_with_default_working_dir(self, tmp_path: Path) -> None:
        """Test executing command with tool's default working directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        tool = ExecTool(working_dir=str(test_dir))
        result = await tool.execute(command="dir")

        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self) -> None:
        """Test executing command that produces stderr."""
        tool = ExecTool()
        result = await tool.execute(command="python -c \"import sys; print('error', file=sys.stderr)\"")
        assert "STDERR:" in result or "error" in result

    @pytest.mark.asyncio
    async def test_execute_command_with_nonzero_exit(self) -> None:
        """Test executing command that exits with non-zero code."""
        tool = ExecTool()
        result = await tool.execute(command="exit 1")
        assert "Exit code: 1" in result

    @pytest.mark.asyncio
    async def test_execute_command_no_output(self) -> None:
        """Test executing command with no output."""
        tool = ExecTool()
        result = await tool.execute(command="exit 0")
        assert "(no output)" in result

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        """Test command timeout."""
        tool = ExecTool(timeout=1)
        result = await tool.execute(command="python -c \"import time; time.sleep(5)\"")
        assert "timed out" in result
        assert "1 seconds" in result

    @pytest.mark.asyncio
    async def test_execute_truncates_long_output(self) -> None:
        """Test that long output is truncated."""
        tool = ExecTool()
        
        async def long_output_cmd():
            return "python -c \"print('x' * 15000, end='')\""
        
        result = await tool.execute(command=await long_output_cmd())
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self) -> None:
        """Test that exceptions are caught and returned as error."""
        tool = ExecTool()
        result = await tool.execute(command="python -c \"raise ValueError('test error')\"")
        assert "Error" in result or "error" in result.lower() or "ValueError" in result


class TestExecToolGuardCommand:
    """Tests for ExecTool._guard_command method."""

    def test_allow_safe_command(self) -> None:
        """Test that safe commands are allowed."""
        tool = ExecTool()
        result = tool._guard_command("echo hello", os.getcwd())
        assert result is None

    def test_block_rm_rf(self) -> None:
        """Test that rm -rf is blocked."""
        tool = ExecTool()
        result = tool._guard_command("rm -rf /", os.getcwd())
        assert "blocked" in result
        assert "dangerous pattern" in result

    def test_block_rm_r(self) -> None:
        """Test that rm -r is blocked."""
        tool = ExecTool()
        result = tool._guard_command("rm -r /tmp", os.getcwd())
        assert "blocked" in result

    def test_block_rm_rf_combined(self) -> None:
        """Test that rm -rf is blocked."""
        tool = ExecTool()
        result = tool._guard_command("rm -rf /tmp", os.getcwd())
        assert "blocked" in result

    def test_block_del_f(self) -> None:
        """Test that del /f is blocked."""
        tool = ExecTool()
        result = tool._guard_command("del /f file.txt", os.getcwd())
        assert "blocked" in result

    def test_block_del_q(self) -> None:
        """Test that del /q is blocked."""
        tool = ExecTool()
        result = tool._guard_command("del /q file.txt", os.getcwd())
        assert "blocked" in result

    def test_block_rmdir_s(self) -> None:
        """Test that rmdir /s is blocked."""
        tool = ExecTool()
        result = tool._guard_command("rmdir /s dir", os.getcwd())
        assert "blocked" in result

    def test_block_format_command(self) -> None:
        """Test that format command is blocked."""
        tool = ExecTool()
        result = tool._guard_command("format C:", os.getcwd())
        assert "blocked" in result

    def test_block_mkfs(self) -> None:
        """Test that mkfs is blocked."""
        tool = ExecTool()
        result = tool._guard_command("mkfs.ext4 /dev/sda", os.getcwd())
        assert "blocked" in result

    def test_block_diskpart(self) -> None:
        """Test that diskpart is blocked."""
        tool = ExecTool()
        result = tool._guard_command("diskpart", os.getcwd())
        assert "blocked" in result

    def test_block_dd(self) -> None:
        """Test that dd is blocked."""
        tool = ExecTool()
        result = tool._guard_command("dd if=/dev/zero of=/dev/sda", os.getcwd())
        assert "blocked" in result

    def test_block_write_to_disk(self) -> None:
        """Test that writing to disk is blocked."""
        tool = ExecTool()
        result = tool._guard_command("echo test > /dev/sda", os.getcwd())
        assert "blocked" in result

    def test_block_shutdown(self) -> None:
        """Test that shutdown is blocked."""
        tool = ExecTool()
        result = tool._guard_command("shutdown now", os.getcwd())
        assert "blocked" in result

    def test_block_reboot(self) -> None:
        """Test that reboot is blocked."""
        tool = ExecTool()
        result = tool._guard_command("reboot", os.getcwd())
        assert "blocked" in result

    def test_block_poweroff(self) -> None:
        """Test that poweroff is blocked."""
        tool = ExecTool()
        result = tool._guard_command("poweroff", os.getcwd())
        assert "blocked" in result

    def test_block_fork_bomb(self) -> None:
        """Test that fork bomb is blocked."""
        tool = ExecTool()
        result = tool._guard_command(":(){ :|:& };:", os.getcwd())
        assert "blocked" in result

    def test_allowlist_blocks_command(self) -> None:
        """Test that allowlist blocks non-matching commands."""
        tool = ExecTool(allow_patterns=[r"^echo\b"])
        result = tool._guard_command("ls -la", os.getcwd())
        assert "blocked" in result
        assert "allowlist" in result

    def test_allowlist_allows_command(self) -> None:
        """Test that allowlist allows matching commands."""
        tool = ExecTool(allow_patterns=[r"^echo\b"])
        result = tool._guard_command("echo hello", os.getcwd())
        assert result is None

    def test_workspace_restricts_path_traversal_backslash(self) -> None:
        """Test that workspace restriction blocks path traversal with backslash."""
        tool = ExecTool(restrict_to_workspace=True)
        result = tool._guard_command("cd ..\\..\\etc", "C:\\workspace")
        assert "blocked" in result
        assert "path traversal" in result

    def test_workspace_restricts_path_traversal_slash(self) -> None:
        """Test that workspace restriction blocks path traversal with slash."""
        tool = ExecTool(restrict_to_workspace=True)
        result = tool._guard_command("cd ../../etc", "/workspace")
        assert "blocked" in result
        assert "path traversal" in result

    def test_workspace_allows_relative_paths(self) -> None:
        """Test that workspace restriction allows relative paths."""
        tool = ExecTool(restrict_to_workspace=True)
        result = tool._guard_command("cd subdir", os.getcwd())
        assert result is None

    def test_workspace_blocks_absolute_path_outside(self, tmp_path: Path) -> None:
        """Test that workspace restriction blocks paths outside working dir."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        tool = ExecTool(restrict_to_workspace=True)
        result = tool._guard_command(f"cd {outside}", str(workspace))
        assert "blocked" in result
        assert "outside working dir" in result

    def test_workspace_allows_path_inside(self, tmp_path: Path) -> None:
        """Test that workspace restriction allows paths inside working dir."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        inside = workspace / "subdir"
        inside.mkdir()

        tool = ExecTool(restrict_to_workspace=True)
        result = tool._guard_command("dir subdir", str(workspace))
        assert result is None

    def test_multiple_deny_patterns(self) -> None:
        """Test that multiple deny patterns are checked."""
        tool = ExecTool()
        result1 = tool._guard_command("rm -rf /", os.getcwd())
        result2 = tool._guard_command("shutdown now", os.getcwd())
        assert result1 is not None
        assert result2 is not None


class TestExecToolIntegration:
    """Integration tests for ExecTool."""

    @pytest.mark.asyncio
    async def test_execute_pipeline(self) -> None:
        """Test executing a command pipeline."""
        tool = ExecTool()
        result = await tool.execute(command="python -c \"print('hello')\" | findstr hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_redirect_output(self, tmp_path: Path) -> None:
        """Test executing command with output redirection."""
        output_file = tmp_path / "output.txt"

        tool = ExecTool()
        await tool.execute(command=f"echo test > {output_file}")

        assert output_file.exists()
        assert output_file.read_text().strip() == "test"

    @pytest.mark.asyncio
    async def test_execute_chained_commands(self) -> None:
        """Test executing chained commands."""
        tool = ExecTool()
        result = await tool.execute(command="echo first && echo second")
        assert "first" in result
        assert "second" in result

    @pytest.mark.asyncio
    async def test_execute_with_special_characters(self) -> None:
        """Test executing command with special characters."""
        tool = ExecTool()
        result = await tool.execute(command="echo 'hello world'")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_execute_environment_variables(self) -> None:
        """Test executing command with environment variables."""
        tool = ExecTool()
        result = await tool.execute(command="echo $HOME")
        assert len(result.strip()) > 0
