"""LiteLLM provider implementation for Clawbot."""

from typing import Any

from clawbot.provider.base import BaseProvider, LLMResponse, ToolCallResult


class LiteLLMProvider(BaseProvider):
    """LiteLLM provider - unified interface for multiple LLM providers."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base_url: str | None = None,
        default_model: str = "gpt-3.5-turbo",
    ):
        """Initialize the LiteLLM provider."""
        super().__init__(api_key, api_base_url)
        self.default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temp: float | None = None,
    ) -> LLMResponse:
        """Send a message to the LLM and get a response."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required. Install with: pip install litellm"
            )

        sanitized_messages = self._sanitize_empty_content(messages)

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

        if self.api_key:
            request_kwargs["api_key"] = self.api_key
        if self.api_base_url:
            request_kwargs["api_base"] = self.api_base_url

        response = await litellm.acompletion(**request_kwargs)

        choice = response.choices[0]
        message = choice.message

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

        reasoning_content: str | None = None
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_content = message.reasoning_content

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
        """Parse tool call arguments from JSON string."""
        import json

        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {"raw": arguments}

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.default_model
