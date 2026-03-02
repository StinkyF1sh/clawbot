"""Agent package for Clawbot."""

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
from clawbot.agent.session import (
    AgentRuntimeConfig,
    ChannelMessage,
)

__all__ = [
    # Session
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
