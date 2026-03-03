"""Tests for tool base class module."""

import pytest
from typing import Any

from clawbot.tools.base import Tool


class ConcreteTool(Tool):
    """Concrete implementation of Tool for testing."""

    def __init__(
        self,
        name: str = "test_tool",
        description: str = "A test tool",
        parameters: dict[str, Any] | None = None,
    ):
        self._name = name
        self._description = description
        self._parameters = parameters or {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "A test value"}
            },
            "required": ["value"]
        }

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
        return f"Executed with {kwargs}"


class TestToolProperties:
    """Tests for Tool abstract base class properties."""

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        tool = ConcreteTool(name="my_tool")
        assert tool.name == "my_tool"

    def test_description_property(self) -> None:
        """Test that description property returns correct value."""
        tool = ConcreteTool(description="Test description")
        assert tool.description == "Test description"

    def test_parameters_property(self) -> None:
        """Test that parameters property returns correct schema."""
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"]
        }
        tool = ConcreteTool(parameters=params)
        assert tool.parameters == params

    def test_abstract_base_class(self) -> None:
        """Test that Tool is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Tool()


class TestToolValidateParams:
    """Tests for Tool.validate_params method."""

    def test_valid_params(self) -> None:
        """Test validation with valid parameters."""
        tool = ConcreteTool()
        errors = tool.validate_params({"value": "test"})
        assert errors == []

    def test_missing_required_param(self) -> None:
        """Test validation with missing required parameter."""
        tool = ConcreteTool()
        errors = tool.validate_params({})
        assert len(errors) == 1
        assert "missing required value" in errors[0]

    def test_wrong_type_string(self) -> None:
        """Test validation with wrong type for string parameter."""
        tool = ConcreteTool()
        errors = tool.validate_params({"value": 123})
        assert len(errors) == 1
        assert "should be string" in errors[0]

    def test_wrong_type_integer(self) -> None:
        """Test validation with wrong type for integer parameter."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        })
        errors = tool.validate_params({"count": "not a number"})
        assert len(errors) == 1
        assert "should be integer" in errors[0]

    def test_wrong_type_number(self) -> None:
        """Test validation with wrong type for number parameter."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {"price": {"type": "number"}},
            "required": ["price"]
        })
        errors = tool.validate_params({"price": "not a number"})
        assert len(errors) == 1
        assert "should be number" in errors[0]

    def test_wrong_type_boolean(self) -> None:
        """Test validation with wrong type for boolean parameter."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"]
        })
        errors = tool.validate_params({"flag": "true"})
        assert len(errors) == 1
        assert "should be boolean" in errors[0]

    def test_wrong_type_array(self) -> None:
        """Test validation with wrong type for array parameter."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"]
        })
        errors = tool.validate_params({"items": "not a list"})
        assert len(errors) == 1
        assert "should be array" in errors[0]

    def test_wrong_type_object(self) -> None:
        """Test validation with wrong type for object parameter."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {"data": {"type": "object"}},
            "required": ["data"]
        })
        errors = tool.validate_params({"data": "not a dict"})
        assert len(errors) == 1
        assert "should be object" in errors[0]

    def test_enum_validation(self) -> None:
        """Test enum constraint validation."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"]
                }
            },
            "required": ["status"]
        })
        errors = tool.validate_params({"status": "unknown"})
        assert len(errors) == 1
        assert "must be one of" in errors[0]

        errors = tool.validate_params({"status": "active"})
        assert errors == []

    def test_minimum_validation(self) -> None:
        """Test minimum constraint validation for numbers."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["age"]
        })
        errors = tool.validate_params({"age": -1})
        assert len(errors) == 1
        assert "must be >= 0" in errors[0]

    def test_maximum_validation(self) -> None:
        """Test maximum constraint validation for numbers."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "score": {"type": "number", "maximum": 100}
            },
            "required": ["score"]
        })
        errors = tool.validate_params({"score": 101})
        assert len(errors) == 1
        assert "must be <= 100" in errors[0]

    def test_min_length_validation(self) -> None:
        """Test minLength constraint validation for strings."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "username": {"type": "string", "minLength": 3}
            },
            "required": ["username"]
        })
        errors = tool.validate_params({"username": "ab"})
        assert len(errors) == 1
        assert "must be at least 3 chars" in errors[0]

    def test_max_length_validation(self) -> None:
        """Test maxLength constraint validation for strings."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "maxLength": 5}
            },
            "required": ["code"]
        })
        errors = tool.validate_params({"code": "123456"})
        assert len(errors) == 1
        assert "must be at most 5 chars" in errors[0]

    def test_nested_object_validation(self) -> None:
        """Test validation of nested objects."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["user"]
        })
        errors = tool.validate_params({"user": {}})
        assert len(errors) == 1
        assert "missing required user.name" in errors[0]

    def test_array_items_validation(self) -> None:
        """Test validation of array items."""
        tool = ConcreteTool(parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["numbers"]
        })
        errors = tool.validate_params({"numbers": [1, "two", 3]})
        assert len(errors) == 1
        assert "[1] should be integer" in errors[0]

    def test_non_object_schema_raises(self) -> None:
        """Test that non-object schema raises ValueError."""
        tool = ConcreteTool(parameters={
            "type": "string",
            "description": "A string value"
        })
        with pytest.raises(ValueError, match="Schema must be object type"):
            tool.validate_params("test")


class TestToolToSchema:
    """Tests for Tool.to_schema method."""

    def test_to_schema_format(self) -> None:
        """Test that to_schema returns correct OpenAI function schema format."""
        tool = ConcreteTool(
            name="test_function",
            description="Test function description",
            parameters={"type": "object", "properties": {}}
        )
        schema = tool.to_schema()
        assert schema == {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function description",
                "parameters": {"type": "object", "properties": {}}
            }
        }

    def test_to_schema_includes_all_fields(self) -> None:
        """Test that all fields are included in schema."""
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"]
        }
        tool = ConcreteTool(parameters=params)
        schema = tool.to_schema()
        assert "type" in schema
        assert "function" in schema
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"] == params
