"""Context management for the Clawbot agent."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawbot.agent.config import AgentRuntimeConfig
    from clawbot.storage.session import SessionStorage


_WORKSPACE_PROMPT_FILE_ORDER = (
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
)


def _load_workspace_prompt_files(
    workspace: str,
    include_bootstrap: bool,
) -> list[tuple[str, str]]:
    """Load workspace markdown files in a stable order for prompt injection."""
    root = Path(workspace).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    try:
        # Enforce exact filename matching even on case-insensitive filesystems.
        existing_file_names = {
            p.name for p in root.iterdir() if p.is_file()
        }
    except OSError:
        return []

    loaded: list[tuple[str, str]] = []
    for file_name in _WORKSPACE_PROMPT_FILE_ORDER:
        if file_name == "BOOTSTRAP.md" and not include_bootstrap:
            continue

        if file_name not in existing_file_names:
            continue

        file_path = root / file_name
        if not file_path.exists() or not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue

        if not content:
            continue

        loaded.append((file_name, content))

    return loaded


def _format_workspace_injected_section(files: list[tuple[str, str]]) -> str:
    """Format loaded workspace files into a markdown section."""
    if not files:
        return ""

    parts = [
        "## Project Context",
        "## Workspace Files (injected)",
    ]
    for file_name, content in files:
        parts.append(f"### {file_name}")
        parts.append(content)

    return "\n\n".join(parts)


def get_system_prompt(
    workspace: str | None = None,
    include_bootstrap: bool = True,
) -> str:
    """Generate system prompt with optional workspace context."""
    base_prompt = """You are a helpful AI assistant."""

    if workspace:
        base_prompt += f"""

You have access to the user's workspace at: {workspace}
You can read, write, and analyze files in this directory.
"""

        injected = _format_workspace_injected_section(
            _load_workspace_prompt_files(workspace, include_bootstrap=include_bootstrap)
        )
        if injected:
            base_prompt += f"\n\n{injected}"

    return base_prompt


class ConversationHistory:
    """Conversation history bound to a single session_id."""

    def __init__(self, session_id: str, storage: "SessionStorage"):
        self.session_id = session_id
        self.storage = storage
        self.messages: list[dict[str, Any]] = []
        self._loaded = False
        self.is_new_session = False
        self._bootstrap_injected = False

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
            if self.storage.get_session_meta(self.session_id) is None:
                self.storage.create_session(self.session_id)
                self.is_new_session = True
            else:
                self.is_new_session = False
            self.messages = self.storage.load_session(self.session_id)
            self._loaded = True

    def consume_bootstrap_flag(self) -> bool:
        """Return True only once for brand new sessions."""
        if self.is_new_session and not self._bootstrap_injected:
            self._bootstrap_injected = True
            return True
        return False

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
        include_bootstrap = history.consume_bootstrap_flag() if history else False
        messages.append({
            "role": "system",
            "content": get_system_prompt(
                workspace,
                include_bootstrap=include_bootstrap,
            ),
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
