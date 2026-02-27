"""OpenAI-compatible provider implementation"""

from typing import Any

from openai import AsyncOpenAI

from clawbot.provider.base import BaseProvider, LLMResponse, ToolCallResult


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible API provider using the official OpenAI SDK.

    This provider works with any OpenAI-compatible API endpoint, including:
    - OpenAI API
    - Zhipu AI
    - Bailian (DashScope)
    - Other compatible endpoints
    """


    def __init__(
        self,
        api_key: str ="null",
        api_base_url: str ="https://default.com/v1",
        default_model: str = "default",
    ):
        """Initialize the OpenAI-compatible provider.

        Args:
            api_key: API key for authentication.
            api_base_url: Base URL for the API endpoint.
            extra_headers: Additional headers to include in requests.
        """
        super().__init__(api_key, api_base_url)
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temp: float | None = None,
    ) -> LLMResponse:
        """Send a message to the LLM and get a response.

        Args:
            messages: List of message dictionaries with role and content.
            tools: List of tool definitions in OpenAI function format.
            model: Model name to use (overrides default).
            max_tokens: Maximum tokens in response.
            temp: Sampling temperature.

        Returns:
            LLMResponse containing the model's response.
        """
        # Sanitize empty content in messages
        sanitized_messages = self._sanitize_empty_content(messages)

        # Build request arguments
        request_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": sanitized_messages,
        }

        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if temp is not None:
            request_kwargs["temperature"] = temp
        if tools is not None:
            request_kwargs["tools"] = tools

        # Make the API call
        response = await self._client.chat.completions.create(**request_kwargs)

        # Extract the first choice
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls: list[ToolCallResult] | None = None
        if message.tool_calls:
            tool_calls = [
                ToolCallResult(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=self._parse_arguments(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        # Extract reasoning content if available
        reasoning_content: str | None = None
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_content = message.reasoning_content

        # Build usage info
        usage: dict[str, Any] | None = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def _parse_arguments(self, arguments: str) -> dict[str, Any]:
        """Parse tool call arguments from JSON string.

        Args:
            arguments: JSON string of arguments.

        Returns:
            Parsed dictionary of arguments.
        """
        import json

        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            # Return raw string as a single argument if parsing fails
            return {"raw": arguments}

    def get_default_model(self) -> str:
        """Get the default model for this provider.

        Returns:
            Default model name.
        """
        return self.default_model
