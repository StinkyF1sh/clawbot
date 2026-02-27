"""Context management for the Clawbot agent."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawbot.storage.session import SessionConfig, SessionStorage


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
    """In-memory conversation history management."""

    def __init__(self, max_window: int = 100):
        """Initialize conversation history with max window size."""
        self.messages: list[dict[str, Any]] = []
        self.max_window = max_window

    def append(self, message: dict[str, Any]) -> None:
        """Append a message in OpenAI format."""
        self.messages.append(message)

    def trim_to_window(self, max_window: int | None = None) -> None:
        """Keep only the latest max_window messages."""
        window = max_window or self.max_window
        if len(self.messages) > window:
            self.messages = self.messages[-window:]

    def extend(self, messages: list[dict[str, Any]]) -> None:
        """Extend history with multiple messages."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all history."""
        self.messages.clear()

    def load_from_storage(self, session_id: str, storage: "SessionStorage") -> None:
        """Load conversation history from storage."""
        self.messages = storage.load_session(session_id)

    def append_and_save(
        self,
        message: dict[str, Any],
        session_id: str,
        storage: "SessionStorage",
    ) -> None:
        """Append a message and save to storage."""
        self.append(message)
        storage.append_message(session_id, message)

    def append_assistant_response(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
        session_id: str | None = None,
        storage: "SessionStorage | None" = None,
    ) -> None:
        """Append assistant response to history, optionally saving to storage."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.append(msg)

        if session_id and storage:
            storage.append_message(session_id, msg)

    def append_tool_response(
        self,
        tool_call_id: str,
        result: Any,
        error: str | None = None,
        session_id: str | None = None,
        storage: "SessionStorage | None" = None,
    ) -> None:
        """Append tool response to history, optionally saving to storage."""
        content = str(result) if error is None else f"Error: {error}"
        msg = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }
        self.append(msg)

        if session_id and storage:
            storage.append_message(session_id, msg)


class ContextBuilder:
    """Build messages list in OpenAI format."""

    def __init__(
        self,
        workspace: str | None = None,
        max_window: int = 100,
        storage: "SessionStorage | None" = None,
    ):
        """Initialize context builder."""
        self.workspace = workspace
        self.max_window = max_window
        self.storage = storage

    def load_history(self, session_id: str) -> ConversationHistory:
        """Load conversation history from storage."""
        if not self.storage:
            raise ValueError("SessionStorage not configured.")

        history = ConversationHistory(max_window=self.max_window)
        history.load_from_storage(session_id, self.storage)
        return history

    def create_history(self, max_window: int | None = None) -> ConversationHistory:
        """Create a new empty conversation history."""
        return ConversationHistory(max_window=max_window or self.max_window)

    def build(
        self,
        current_user_input: str | list[dict],
        history: ConversationHistory | None = None,
        config: "SessionConfig | None" = None,
    ) -> list[dict[str, Any]]:
        """Build messages list for provider.chat()."""
        messages: list[dict[str, Any]] = []

        # 1. System prompt
        workspace = config.workspace if config else self.workspace
        messages.append({
            "role": "system",
            "content": get_system_prompt(workspace),
        })

        # 2. Conversation history
        if history:
            max_window = config.max_window if config else None
            history.trim_to_window(max_window)
            messages.extend(history.messages)

        # 3. Current user input
        messages.append({
            "role": "user",
            "content": current_user_input,
        })

        return messages
