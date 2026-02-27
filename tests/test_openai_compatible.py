"""Tests for OpenAI-compatible provider implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from clawbot.config.loader import load_config
from clawbot.provider.base import ToolCallResult
from clawbot.provider.openai_compatible import OpenAICompatibleProvider


class TestOpenAICompatibleProviderInit:
    """Tests for OpenAICompatibleProvider initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        provider = OpenAICompatibleProvider()
        assert provider.api_key == "null"
        assert provider.api_base_url == "https://default.com/v1"
        assert provider.default_model == "default"

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        provider = OpenAICompatibleProvider(
            api_key="sk-test-key",
            api_base_url="https://api.example.com/v1",
            default_model="gpt-4",
        )
        assert provider.api_key == "sk-test-key"
        assert provider.api_base_url == "https://api.example.com/v1"
        assert provider.default_model == "gpt-4"

    def test_client_initialization(self) -> None:
        """Test that AsyncOpenAI client is initialized correctly."""
        provider = OpenAICompatibleProvider(
            api_key="sk-test",
            api_base_url="https://api.test.com/v1",
        )
        assert provider._client is not None
        assert provider._client.api_key == "sk-test"
        assert str(provider._client.base_url) == "https://api.test.com/v1/"


class TestGetDefaultModel:
    """Tests for get_default_model method."""

    def test_get_default_model(self) -> None:
        """Test getting the default model."""
        provider = OpenAICompatibleProvider(default_model="gpt-4-turbo")
        assert provider.get_default_model() == "gpt-4-turbo"


class TestParseArguments:
    """Tests for _parse_arguments method."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON string."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments('{"name": "test", "value": 123}')
        assert result == {"name": "test", "value": 123}

    def test_parse_json_with_nested_objects(self) -> None:
        """Test parsing JSON with nested objects."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments('{"data": {"nested": {"key": "value"}}}')
        assert result == {"data": {"nested": {"key": "value"}}}

    def test_parse_json_with_arrays(self) -> None:
        """Test parsing JSON with arrays."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments('{"items": [1, 2, 3], "tags": ["a", "b"]}')
        assert result == {"items": [1, 2, 3], "tags": ["a", "b"]}

    def test_parse_invalid_json_returns_raw(self) -> None:
        """Test that invalid JSON returns raw string."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments("not a json")
        assert result == {"raw": "not a json"}

    def test_parse_malformed_json(self) -> None:
        """Test parsing malformed JSON."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments('{"incomplete": ')
        assert result == {"raw": '{"incomplete": '}

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments("")
        assert result == {"raw": ""}

    def test_parse_empty_object(self) -> None:
        """Test parsing empty JSON object."""
        provider = OpenAICompatibleProvider()
        result = provider._parse_arguments("{}")
        assert result == {}


class TestChat:
    """Tests for chat method."""

    @pytest.fixture
    def mock_openai_response(self) -> AsyncMock:
        """Create a mock OpenAI API response."""
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test123"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4"

        # Setup choices
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Hello, world!"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning_content = None

        # Setup usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.asyncio
    async def test_chat_basic_message(self, mock_openai_response: MagicMock) -> None:
        """Test basic chat with simple message."""
        provider = OpenAICompatibleProvider()

        mock_create = AsyncMock(return_value=mock_openai_response)
        provider._client.chat.completions.create = mock_create

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify API call was made correctly
        mock_create.assert_called_once_with(
            model="default",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify response
        assert result.content == "Hello, world!"
        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    @pytest.mark.asyncio
    async def test_chat_with_custom_model(self, mock_openai_response: MagicMock) -> None:
        """Test chat with custom model override."""
        provider = OpenAICompatibleProvider(default_model="default-model")

        mock_create = AsyncMock(return_value=mock_openai_response)
        provider._client.chat.completions.create = mock_create

        await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            model="custom-model",
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens(self, mock_openai_response: MagicMock) -> None:
        """Test chat with max_tokens parameter."""
        provider = OpenAICompatibleProvider()

        mock_create = AsyncMock(return_value=mock_openai_response)
        provider._client.chat.completions.create = mock_create

        await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1000,
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, mock_openai_response: MagicMock) -> None:
        """Test chat with temperature parameter."""
        provider = OpenAICompatibleProvider()

        mock_create = AsyncMock(return_value=mock_openai_response)
        provider._client.chat.completions.create = mock_create

        await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            temp=0.7,
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, mock_openai_response: MagicMock) -> None:
        """Test chat with tools parameter."""
        provider = OpenAICompatibleProvider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        mock_create = AsyncMock(return_value=mock_openai_response)
        provider._client.chat.completions.create = mock_create

        await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_with_all_parameters(self, mock_openai_response: MagicMock) -> None:
        """Test chat with all optional parameters."""
        provider = OpenAICompatibleProvider()
        tools = [{"type": "function", "function": {"name": "test"}}]

        mock_create = AsyncMock(return_value=mock_openai_response)
        provider._client.chat.completions.create = mock_create

        await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            model="custom-model",
            max_tokens=2000,
            temp=0.5,
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["max_tokens"] == 2000
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        """Test chat response with tool calls."""
        provider = OpenAICompatibleProvider()

        # Create mock response with tool calls
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "New York", "unit": "celsius"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "I'll check the weather for you."
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.message.reasoning_content = None
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        provider._client.chat.completions.create = mock_create

        result = await provider.chat(messages=[{"role": "user", "content": "test"}])

        assert result.content == "I'll check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {
            "location": "New York",
            "unit": "celsius",
        }
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_chat_with_multiple_tool_calls(self) -> None:
        """Test chat response with multiple tool calls."""
        provider = OpenAICompatibleProvider()

        # Create mock response with multiple tool calls
        mock_response = MagicMock()

        mock_tool_call1 = MagicMock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function.name = "get_weather"
        mock_tool_call1.function.arguments = '{"location": "NY"}'

        mock_tool_call2 = MagicMock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function.name = "get_time"
        mock_tool_call2.function.arguments = '{"timezone": "UTC"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "I'll help with both."
        mock_choice.message.tool_calls = [mock_tool_call1, mock_tool_call2]
        mock_choice.message.reasoning_content = None
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 25
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 40
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        provider._client.chat.completions.create = mock_create

        result = await provider.chat(messages=[{"role": "user", "content": "test"}])

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "get_time"

    @pytest.mark.asyncio
    async def test_chat_with_reasoning_content(self) -> None:
        """Test chat response with reasoning content (o1-style models)."""
        provider = OpenAICompatibleProvider()

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Final answer"
        mock_choice.message.reasoning_content = "Step-by-step reasoning..."
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        provider._client.chat.completions.create = mock_create

        result = await provider.chat(messages=[{"role": "user", "content": "test"}])

        assert result.content == "Final answer"
        assert result.reasoning_content == "Step-by-step reasoning..."

    @pytest.mark.asyncio
    async def test_chat_with_invalid_tool_call_arguments(self) -> None:
        """Test tool call with invalid JSON arguments falls back to raw."""
        provider = OpenAICompatibleProvider()

        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "broken_tool"
        mock_tool_call.function.arguments = "this is not json"

        mock_choice = MagicMock()
        mock_choice.message.content = "Calling tool..."
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.message.reasoning_content = None
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        provider._client.chat.completions.create = mock_create

        result = await provider.chat(messages=[{"role": "user", "content": "test"}])

        assert result.tool_calls[0].arguments == {"raw": "this is not json"}

    @pytest.mark.asyncio
    async def test_chat_with_empty_content_sanitization(self) -> None:
        """Test that empty content in messages is sanitized."""
        provider = OpenAICompatibleProvider()

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning_content = None
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        provider._client.chat.completions.create = mock_create

        # Message with empty string content
        await provider.chat(messages=[{"role": "user", "content": ""}])

        # Check that empty content was sanitized
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "(empty)"

    @pytest.mark.asyncio
    async def test_chat_with_null_usage(self) -> None:
        """Test chat response with null usage."""
        provider = OpenAICompatibleProvider()

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning_content = None
        mock_choice.finish_reason = "stop"
        mock_response.usage = None
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        provider._client.chat.completions.create = mock_create

        result = await provider.chat(messages=[{"role": "user", "content": "test"}])

        assert result.usage is None


class TestSanitizeEmptyContent:
    """Tests for inherited _sanitize_empty_content method."""

    def test_sanitize_empty_string_content(self) -> None:
        """Test sanitization of empty string content."""
        provider = OpenAICompatibleProvider()
        messages = [{"role": "user", "content": ""}]
        result = provider._sanitize_empty_content(messages)
        assert result[0]["content"] == "(empty)"

    def test_sanitize_assistant_with_tool_calls_empty_content(self) -> None:
        """Test assistant with tool calls has None content instead of (empty)."""
        provider = OpenAICompatibleProvider()
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}
        ]
        result = provider._sanitize_empty_content(messages)
        assert result[0]["content"] is None

    def test_sanitize_list_content_with_empty_text_items(self) -> None:
        """Test sanitization of list content with empty text items."""
        provider = OpenAICompatibleProvider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": ""},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ],
            }
        ]
        result = provider._sanitize_empty_content(messages)
        assert len(result[0]["content"]) == 2  # Empty text item filtered out
        assert result[0]["content"][0]["text"] == "Hello"

    def test_sanitize_no_changes_needed(self) -> None:
        """Test that valid messages are not modified."""
        provider = OpenAICompatibleProvider()
        messages = [{"role": "user", "content": "Hello"}]
        result = provider._sanitize_empty_content(messages)
        assert result == messages


class TestIntegrationWithConfig:
    """Integration tests with actual configuration."""

    def test_provider_from_default_config(self) -> None:
        """Test creating provider from default agent configuration."""
        config = load_config()
        agent_config = config.get_agent_config("default")
        provider_config = config.get_provider(agent_config.model)

        provider = OpenAICompatibleProvider(
            api_key=provider_config.api_key,
            api_base_url=config.get_api_base(agent_config.model),
            default_model=agent_config.model,
        )

        assert provider.api_key == provider_config.api_key
        assert provider.api_base_url == config.get_api_base(agent_config.model)
        assert provider.default_model == agent_config.model

    def test_provider_from_zhipu_config(self) -> None:
        """Test creating provider with Zhipu configuration."""
        config = load_config()
        zhipu_config = config.providers.zhipu

        provider = OpenAICompatibleProvider(
            api_key=zhipu_config.api_key,
            api_base_url=zhipu_config.api_base,
            default_model="glm-4.7",
        )

        assert provider.api_key == zhipu_config.api_key
        assert provider.api_base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_provider_from_openai_config(self) -> None:
        """Test creating provider with OpenAI configuration."""
        config = load_config()
        openai_config = config.providers.openai

        provider = OpenAICompatibleProvider(
            api_key=openai_config.api_key,
            api_base_url=openai_config.api_base,
            default_model="gpt-4",
        )

        assert provider.api_key == openai_config.api_key
        assert provider.api_base_url == "https://api.openai.com/v1"

    def test_provider_from_bailian_config(self) -> None:
        """Test creating provider with Bailian configuration."""
        config = load_config()
        bailian_config = config.providers.bailian

        provider = OpenAICompatibleProvider(
            api_key=bailian_config.api_key,
            api_base_url=bailian_config.api_base,
            default_model="qwen-max",
        )

        assert provider.api_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self) -> None:
        """Test creating LLMResponse with all fields."""
        from clawbot.provider.base import LLMResponse

        response = LLMResponse(
            content="Test content",
            tool_calls=[ToolCallResult(id="1", name="test", arguments={})],
            finish_reason="stop",
            usage={"total_tokens": 100},
            reasoning_content="Reasoning here",
        )

        assert response.content == "Test content"
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 100
        assert response.reasoning_content == "Reasoning here"

    def test_llm_response_defaults(self) -> None:
        """Test LLMResponse with default values."""
        from clawbot.provider.base import LLMResponse

        response = LLMResponse(content="Test")

        assert response.tool_calls == []
        assert response.finish_reason == "stop"


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""

    def test_tool_call_result_creation(self) -> None:
        """Test creating ToolCallResult."""
        result = ToolCallResult(
            id="call_123", name="get_weather", arguments={"location": "NY"}
        )

        assert result.id == "call_123"
        assert result.name == "get_weather"
        assert result.arguments == {"location": "NY"}
