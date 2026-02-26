"""Base provider for the Clawbot agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallResult:
    """Result of a tool call."""
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    tool_calls: list[ToolCallResult]|None = field(default_factory=list)
    finish_reason: str ="stop"
    useage: dict[str, Any]|None = field(default_factory=dict)
    reasoning_content: str|None = None

class BaseProvider(ABC):
    """Base provider for the Clawbot agent."""

    def __init__(self, api_key: str|None = None,api_base_url: str|None = None):
        """Initialize the provider."""
        self.api_key = api_key
        self.api_base_url = api_base_url

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools:list[dict[str,Any]]|None = None,
        model: str|None = None,
        max_tokens: int|None = None,
        temp: float|None = None
    ) -> LLMResponse:
        """Send a message to the LLM and get a response."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @staticmethod
    def _sanitize_empty_content(message: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize empty message from the LLM."""
        result: list[dict[str, Any]] = []
        for msg in message:
            content = msg.get("content")
            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (
                    msg.get("role") == "assistant"
                    and msg.get("tool_calls")
                ) else "(empty)"
                result.append(clean)
                continue
            if isinstance(content, list):
                filtered = [
                    item for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue
            result.append(msg)
        return result
