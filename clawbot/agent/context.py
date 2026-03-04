"""Context management for the Clawbot agent."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawbot.agent.config import AgentRuntimeConfig
    from clawbot.storage.session import SessionStorage


def get_system_prompt(workspace: str | None = None) -> str:
    """Generate system prompt with optional workspace context."""
    base_prompt = """You are a helpful AI assistant."""

    if workspace:
        base_prompt += f"""

You have access to the user's workspace at: {workspace}
You can read, write, and analyze files in this directory.
"""

    return base_prompt


class ConversationHistory:
    """Conversation history bound to a single session_id."""

    def __init__(self, session_id: str, storage: "SessionStorage"):
        self.session_id = session_id
        self.storage = storage
        self.messages: list[dict[str, Any]] = []
        self._loaded = False

    def append(self, message: dict[str, Any]) -> None:
        """Append a message in OpenAI format."""
        self.messages.append(message)

    def trim_to_window(self, max_window: int) -> None:
        """Keep only the latest max_window messages."""
        if len(self.messages) > max_window:
            self.messages = self.messages[-max_window:]

    def extend(self, messages: list[dict[str, Any]]) -> None:
        """Extend history with multiple messages."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all history."""
        self.messages.clear()

    def load(self) -> None:
        """Load conversation history from storage (lazy load, once only)."""
        if not self._loaded:
            self.messages = self.storage.load_session(self.session_id)
            self._loaded = True

    def save(self, message: dict[str, Any]) -> None:
        """Append message and save to storage."""
        self.append(message)
        self.storage.append_message(self.session_id, message)

    def append_assistant_response(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Append assistant response to history and save to storage."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.save(msg)

    def append_tool_response(
        self,
        tool_call_id: str,
        result: Any,
        error: str | None = None,
    ) -> None:
        """Append tool response to history and save to storage."""
        content = str(result) if error is None else f"Error: {error}"
        msg = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }
        self.save(msg)


class ContextBuilder:
    """Stateless utility class - globally shared."""

    def __init__(
        self,
        storage: "SessionStorage",
        default_workspace: str | None = None,
    ):
        self.storage = storage
        self.default_workspace = default_workspace

    def _get_max_window(self, agent_config: "AgentRuntimeConfig | None") -> int:
        return agent_config.memory_window if agent_config else 100

    def create_history(
        self,
        session_id: str,
    ) -> ConversationHistory:
        return ConversationHistory(session_id, self.storage)

    def build(
        self,
        session_id: str,
        user_input: str | list[dict],
        agent_config: "AgentRuntimeConfig | None" = None,
        history: ConversationHistory | None = None,
    ) -> list[dict[str, Any]]:
        """Build messages list for Provider.chat()."""
        messages: list[dict[str, Any]] = []

        workspace = (
            agent_config.workspace
            if agent_config
            else self.default_workspace
        )
        messages.append({
            "role": "system",
            "content": get_system_prompt(workspace),
        })

        if history:
            history.trim_to_window(self._get_max_window(agent_config))
            messages.extend(history.messages)
        else:
            hist = self.create_history(session_id)
            hist.load()
            hist.trim_to_window(self._get_max_window(agent_config))
            messages.extend(hist.messages)

        if user_input:
            messages.append({
                "role": "user",
                "content": user_input,
            })

        return messages
