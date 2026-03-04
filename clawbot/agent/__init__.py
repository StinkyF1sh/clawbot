"""Agent package for Clawbot."""

from clawbot.agent.config import AgentRuntimeConfig, ChannelMessage
from clawbot.agent.context import (
    ContextBuilder,
    ConversationHistory,
    get_system_prompt,
)
from clawbot.agent.loop import (
    CliHandler,
    GlobalAgentLoop,
    SingleSessionAgentLoop,
)

__all__ = [
    # Config
    "AgentRuntimeConfig",
    "ChannelMessage",
    # Context
    "ContextBuilder",
    "ConversationHistory",
    "get_system_prompt",
    # Loop
    "GlobalAgentLoop",
    "SingleSessionAgentLoop",
    "CliHandler",
]
