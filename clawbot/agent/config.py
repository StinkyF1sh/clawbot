"""Agent runtime configuration for Clawbot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clawbot.config.schema import AgentDefaults
from clawbot.storage.session import SessionConfig


@dataclass
class AgentRuntimeConfig:
    """Agent runtime configuration - mapped from ClawbotConfig.agents[agent_name]."""

    name: str
    workspace: str
    model: str
    max_tokens: int
    temperature: float
    max_tool_iterations: int
    memory_window: int

    @property
    def model_name(self) -> str:
        """Extract pure model name without provider prefix."""
        if "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

    @classmethod
    def from_agent_defaults(cls, name: str, cfg: AgentDefaults) -> "AgentRuntimeConfig":
        return cls(
            name=name,
            workspace=str(Path(cfg.workspace).expanduser()),
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            max_tool_iterations=cfg.max_tool_iterations,
            memory_window=cfg.memory_window,
        )

    def to_session_config(self) -> SessionConfig:
        return SessionConfig(
            workspace=self.workspace,
            max_window=self.memory_window,
        )


@dataclass
class ChannelMessage:
    """Raw message received from Channel."""

    channel_id: str
    channel_session_id: str
    agent_name: str
    content: str
    metadata: dict[str, Any] | None = None

    @property
    def resolved_session_id(self) -> str:
        """Generate global session_id from channel_id and channel_session_id."""
        return f"{self.channel_id}:{self.channel_session_id}"
