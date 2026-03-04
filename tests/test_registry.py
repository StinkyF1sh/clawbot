"""Tests for tool registry module."""

from typing import Any

import pytest

from clawbot.tools.base import Tool
from clawbot.tools.registry import ToolRegistry


class MockTool(Tool):
    """Mock tool for testing registry."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool",
        parameters: dict[str, Any] | None = None,
        result: str = "Mock result",
    ):
        self._name = name
        self._description = description
        self._parameters = parameters or {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"]
        }
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        return self._result


class TestToolRegistryInit:
    """Tests for ToolRegistry initialization."""

    def test_empty_registry(self) -> None:
        """Test that registry starts empty."""
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.tool_names == []

    def test_registry_tools_dict(self) -> None:
        """Test that tools are stored in internal dict."""
        registry = ToolRegistry()
        assert isinstance(registry._tools, dict)


class TestToolRegistryRegister:
    """Tests for ToolRegistry.register method."""

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        assert len(registry) == 1
        assert "mock_tool" in registry.tool_names

    def test_register_multiple_tools(self) -> None:
        """Test registering multiple tools."""
        registry = ToolRegistry()
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")
        tool3 = MockTool(name="tool3")

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        assert len(registry) == 3
        assert set(registry.tool_names) == {"tool1", "tool2", "tool3"}

    def test_register_overwrites_existing(self) -> None:
        """Test that registering with same name overwrites."""
        registry = ToolRegistry()
        tool1 = MockTool(name="tool", result="first")
        tool2 = MockTool(name="tool", result="second")

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 1
        assert registry.get("tool") is tool2


class TestToolRegistryUnregister:
    """Tests for ToolRegistry.unregister method."""

    def test_unregister_existing_tool(self) -> None:
        """Test unregistering an existing tool."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        registry.unregister("mock_tool")

        assert len(registry) == 0
        assert "mock_tool" not in registry.tool_names

    def test_unregister_nonexistent_tool(self) -> None:
        """Test unregistering a non-existent tool doesn't raise."""
        registry = ToolRegistry()
        registry.unregister("nonexistent")
        assert len(registry) == 0

    def test_unregister_from_multiple(self) -> None:
        """Test unregistering one tool from multiple."""
        registry = ToolRegistry()
        registry.register(MockTool(name="tool1"))
        registry.register(MockTool(name="tool2"))
        registry.register(MockTool(name="tool3"))

        registry.unregister("tool2")

        assert len(registry) == 2
        assert "tool2" not in registry.tool_names
        assert "tool1" in registry.tool_names
        assert "tool3" in registry.tool_names


class TestToolRegistryGet:
    """Tests for ToolRegistry.get method."""

    def test_get_existing_tool(self) -> None:
        """Test getting an existing tool."""
        registry = ToolRegistry()
        tool = MockTool(name="test")
        registry.register(tool)

        result = registry.get("test")
        assert result is tool

    def test_get_nonexistent_tool(self) -> None:
        """Test getting a non-existent tool returns None."""
        registry = ToolRegistry()
        result = registry.get("nonexistent")
        assert result is None


class TestToolRegistryHas:
    """Tests for ToolRegistry.has method."""

    def test_has_existing_tool(self) -> None:
        """Test checking for an existing tool."""
        registry = ToolRegistry()
        registry.register(MockTool(name="test"))
        assert registry.has("test") is True

    def test_has_nonexistent_tool(self) -> None:
        """Test checking for a non-existent tool."""
        registry = ToolRegistry()
        assert registry.has("nonexistent") is False

    def test_has_after_unregister(self) -> None:
        """Test checking for unregistered tool."""
        registry = ToolRegistry()
        registry.register(MockTool(name="test"))
        registry.unregister("test")
        assert registry.has("test") is False


class TestToolRegistryContains:
    """Tests for ToolRegistry.__contains__ method."""

    def test_contains_existing_tool(self) -> None:
        """Test __contains__ for existing tool."""
        registry = ToolRegistry()
        registry.register(MockTool(name="test"))
        assert "test" in registry

    def test_contains_nonexistent_tool(self) -> None:
        """Test __contains__ for non-existent tool."""
        registry = ToolRegistry()
        assert "nonexistent" not in registry


class TestToolRegistryGetDefinitions:
    """Tests for ToolRegistry.get_definitions method."""

    def test_empty_registry_definitions(self) -> None:
        """Test definitions for empty registry."""
        registry = ToolRegistry()
        definitions = registry.get_definitions()
        assert definitions == []

    def test_single_tool_definition(self) -> None:
        """Test definition for single tool."""
        registry = ToolRegistry()
        tool = MockTool(
            name="test_tool",
            description="Test description",
            parameters={"type": "object", "properties": {}}
        )
        registry.register(tool)

        definitions = registry.get_definitions()

        assert len(definitions) == 1
        assert definitions[0] == {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test description",
                "parameters": {"type": "object", "properties": {}}
            }
        }

    def test_multiple_tool_definitions(self) -> None:
        """Test definitions for multiple tools."""
        registry = ToolRegistry()
        registry.register(MockTool(name="tool1"))
        registry.register(MockTool(name="tool2"))

        definitions = registry.get_definitions()

        assert len(definitions) == 2
        names = {d["function"]["name"] for d in definitions}
        assert names == {"tool1", "tool2"}


class TestToolRegistryExecute:
    """Tests for ToolRegistry.execute method."""

    @pytest.mark.asyncio
    async def test_execute_existing_tool(self) -> None:
        """Test executing an existing tool."""
        registry = ToolRegistry()
        tool = MockTool(name="test", result="success")
        registry.register(tool)

        result = await registry.execute("test", {"input": "value"})
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self) -> None:
        """Test executing a non-existent tool."""
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})

        assert "Error: Tool 'nonexistent' not found" in result
        assert "Available:" in result

    @pytest.mark.asyncio
    async def test_execute_with_invalid_params(self) -> None:
        """Test executing tool with invalid parameters."""
        registry = ToolRegistry()
        tool = MockTool(parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"]
        })
        registry.register(tool)

        result = await registry.execute("mock_tool", {"value": "not an integer"})
        assert "Error: Invalid parameters" in result
        assert "should be integer" in result

    @pytest.mark.asyncio
    async def test_execute_with_missing_required_param(self) -> None:
        """Test executing tool with missing required parameter."""
        registry = ToolRegistry()
        tool = MockTool(parameters={
            "type": "object",
            "properties": {"required_param": {"type": "string"}},
            "required": ["required_param"]
        })
        registry.register(tool)

        result = await registry.execute("mock_tool", {})
        assert "Error: Invalid parameters" in result
        assert "missing required" in result

    @pytest.mark.asyncio
    async def test_execute_returns_error_with_hint(self) -> None:
        """Test that execution errors include analysis hint."""
        registry = ToolRegistry()
        tool = MockTool(parameters={
            "type": "object",
            "properties": {"num": {"type": "integer"}},
            "required": ["num"]
        })
        registry.register(tool)

        result = await registry.execute("mock_tool", {"num": "string"})
        assert "[Analyze the error above" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_prefix(self) -> None:
        """Test that tool errors starting with 'Error' get hint appended."""
        registry = ToolRegistry()

        class ErrorTool(Tool):
            @property
            def name(self) -> str:
                return "error_tool"

            @property
            def description(self) -> str:
                return "Always errors"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> str:
                return "Error: Something went wrong"

        registry.register(ErrorTool())
        result = await registry.execute("error_tool", {})

        assert "Error: Something went wrong" in result
        assert "[Analyze the error above" in result

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self) -> None:
        """Test that exceptions during execution are caught."""
        registry = ToolRegistry()

        class ExceptionTool(Tool):
            @property
            def name(self) -> str:
                return "exception_tool"

            @property
            def description(self) -> str:
                return "Always raises"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> str:
                raise ValueError("Test exception")

        registry.register(ExceptionTool())
        result = await registry.execute("exception_tool", {})

        assert "Error executing exception_tool" in result
        assert "Test exception" in result


class TestToolRegistryToolNames:
    """Tests for ToolRegistry.tool_names property."""

    def test_empty_registry_names(self) -> None:
        """Test tool_names for empty registry."""
        registry = ToolRegistry()
        assert registry.tool_names == []

    def test_names_after_registration(self) -> None:
        """Test tool_names after registering tools."""
        registry = ToolRegistry()
        registry.register(MockTool(name="alpha"))
        registry.register(MockTool(name="beta"))
        registry.register(MockTool(name="gamma"))

        names = registry.tool_names
        assert len(names) == 3
        assert set(names) == {"alpha", "beta", "gamma"}

    def test_names_returns_list(self) -> None:
        """Test that tool_names returns a list."""
        registry = ToolRegistry()
        registry.register(MockTool())
        assert isinstance(registry.tool_names, list)
