"""Tests for agent context management module."""

from clawbot.agent.context import (
    ContextBuilder,
    ConversationHistory,
    get_system_prompt,
)
from clawbot.storage.session import SessionConfig, SessionStorage


class TestGetSystemPrompt:
    """Tests for get_system_prompt function."""

    def test_default_prompt_without_workspace(self) -> None:
        """Test system prompt without workspace context."""
        prompt = get_system_prompt()
        assert prompt == "You are a helpful AI assistant."

    def test_prompt_with_workspace(self) -> None:
        """Test system prompt with workspace context."""
        workspace = "/home/user/project"
        prompt = get_system_prompt(workspace)
        assert "You are a helpful AI assistant." in prompt
        assert workspace in prompt
        assert "You can read, write, and analyze files in this directory." in prompt


class TestConversationHistoryInit:
    """Tests for ConversationHistory initialization."""

    def test_init_with_session_and_storage(self, tmp_path) -> None:
        """Test initialization with session_id and storage."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session-123"

        history = ConversationHistory(session_id, storage)
        assert history.session_id == session_id
        assert history.storage is storage
        assert history.messages == []
        assert history._loaded is False


class TestConversationHistoryAppend:
    """Tests for ConversationHistory append methods."""

    def test_append_single_message(self, tmp_path) -> None:
        """Test appending a single message."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append({"role": "user", "content": "Hello"})
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"

    def test_append_multiple_messages(self, tmp_path) -> None:
        """Test appending multiple messages."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append({"role": "user", "content": "Hello"})
        history.append({"role": "assistant", "content": "Hi there!"})
        assert len(history.messages) == 2

    def test_extend_messages(self, tmp_path) -> None:
        """Test extending with multiple messages at once."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        history.extend(messages)
        assert len(history.messages) == 2

    def test_clear_history(self, tmp_path) -> None:
        """Test clearing history."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append({"role": "user", "content": "Hello"})
        history.clear()
        assert len(history.messages) == 0


class TestConversationHistoryTrim:
    """Tests for ConversationHistory trim_to_window method."""

    def test_trim_no_op_when_under_limit(self, tmp_path) -> None:
        """Test trim does nothing when under limit."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append({"role": "user", "content": "Hello"})
        history.trim_to_window(100)
        assert len(history.messages) == 1

    def test_trim_keeps_latest_messages(self, tmp_path) -> None:
        """Test trim keeps only the latest messages."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        for i in range(5):
            history.append({"role": "user", "content": f"Message {i}"})
        history.trim_to_window(3)
        assert len(history.messages) == 3
        # Should keep the last 3 messages
        assert history.messages[0]["content"] == "Message 2"
        assert history.messages[2]["content"] == "Message 4"

    def test_trim_with_custom_window(self, tmp_path) -> None:
        """Test trim with custom window size."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        for i in range(10):
            history.append({"role": "user", "content": f"Message {i}"})
        history.trim_to_window(5)
        assert len(history.messages) == 5
        assert history.messages[0]["content"] == "Message 5"


class TestConversationHistoryStorage:
    """Tests for ConversationHistory storage methods."""

    def test_load(self, tmp_path) -> None:
        """Test loading from storage."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session-123"

        # First save a message
        storage.append_message(session_id, {"role": "user", "content": "Hello"})

        # Then load
        history = ConversationHistory(session_id, storage)
        history.load()

        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history._loaded is True

    def test_load_only_once(self, tmp_path) -> None:
        """Test that load() only loads once."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session-123"

        storage.append_message(session_id, {"role": "user", "content": "Hello"})

        history = ConversationHistory(session_id, storage)
        history.load()
        history.load()  # Second call should be no-op

        assert history._loaded is True
        assert len(history.messages) == 1

    def test_save_appends_and_persists(self, tmp_path) -> None:
        """Test save appends to memory and persists to storage."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session-123"

        history = ConversationHistory(session_id, storage)
        history.save({"role": "user", "content": "Hello"})

        assert len(history.messages) == 1
        assert history.messages[0]["content"] == "Hello"

        # Verify persisted to storage
        loaded = storage.load_session(session_id)
        assert len(loaded) == 1
        assert loaded[0]["content"] == "Hello"

    def test_append_assistant_response(self, tmp_path) -> None:
        """Test appending assistant response."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append_assistant_response(content="Hello!")
        assert len(history.messages) == 1
        assert history.messages[0] == {"role": "assistant", "content": "Hello!"}

    def test_append_assistant_response_with_tool_calls(self, tmp_path) -> None:
        """Test appending assistant response with tool calls."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        history.append_assistant_response(
            content="Let me check",
            tool_calls=tool_calls,
        )
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "assistant"
        assert history.messages[0]["content"] == "Let me check"
        assert history.messages[0]["tool_calls"] == tool_calls

    def test_append_tool_response(self, tmp_path) -> None:
        """Test appending tool response."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append_tool_response(
            tool_call_id="call_1",
            result="Weather is sunny",
        )
        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg["role"] == "tool"
        assert msg["content"] == "Weather is sunny"
        assert msg["tool_call_id"] == "call_1"

    def test_append_tool_response_with_error(self, tmp_path) -> None:
        """Test appending tool response with error."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        history = ConversationHistory("test-session", storage)

        history.append_tool_response(
            tool_call_id="call_1",
            result=None,
            error="API timeout",
        )
        assert history.messages[0]["content"] == "Error: API timeout"


class TestContextBuilderInit:
    """Tests for ContextBuilder initialization."""

    def test_init_requires_storage(self, tmp_path) -> None:
        """Test initialization requires storage parameter."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)

        builder = ContextBuilder(storage=storage)
        assert builder.storage is storage
        assert builder.default_workspace is None

    def test_init_with_default_workspace(self, tmp_path) -> None:
        """Test initialization with default workspace."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)

        builder = ContextBuilder(storage=storage, default_workspace="/tmp/workspace")
        assert builder.default_workspace == "/tmp/workspace"


class TestContextBuilderCreateHistory:
    """Tests for ContextBuilder history creation methods."""

    def test_create_history(self, tmp_path) -> None:
        """Test creating new empty history."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        history = builder.create_history("test-session")
        assert isinstance(history, ConversationHistory)
        assert history.session_id == "test-session"
        assert len(history.messages) == 0

    def test_load_history_via_builder(self, tmp_path) -> None:
        """Test loading history via ContextBuilder."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        session_id = "test-session"
        storage.append_message(session_id, {"role": "user", "content": "Hello"})

        history = builder.create_history(session_id)
        history.load()

        assert len(history.messages) == 1


class TestContextBuilderBuild:
    """Tests for ContextBuilder build method."""

    def test_build_with_only_user_input(self, tmp_path) -> None:
        """Test building messages with only user input."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        messages = builder.build(
            session_id="test-session",
            user_input="Hello",
        )
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_with_history(self, tmp_path) -> None:
        """Test building messages with conversation history."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        history = ConversationHistory("test-session", storage)
        history.append({"role": "user", "content": "Previous"})
        history.append({"role": "assistant", "content": "Response"})

        messages = builder.build(
            session_id="test-session",
            user_input="New message",
            history=history,
        )
        assert len(messages) == 4  # system + 2 history + user
        assert messages[1]["content"] == "Previous"
        assert messages[2]["content"] == "Response"

    def test_build_with_agent_config(self, tmp_path) -> None:
        """Test building messages with AgentRuntimeConfig."""
        from clawbot.agent.config import AgentRuntimeConfig

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        agent_config = AgentRuntimeConfig(
            name="test",
            workspace=str(tmp_path),
            model="zhipu/glm-4.7",
            max_tokens=4096,
            temperature=0.1,
            max_tool_iterations=10,
            memory_window=50,
        )

        messages = builder.build(
            session_id="test-session",
            user_input="Hello",
            agent_config=agent_config,
        )
        # Check workspace from agent_config is used in system prompt
        assert str(tmp_path) in messages[0]["content"]

    def test_build_trims_history_based_on_agent_config(self, tmp_path) -> None:
        """Test that history is trimmed based on agent config."""
        from clawbot.agent.config import AgentRuntimeConfig

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        agent_config = AgentRuntimeConfig(
            name="test",
            workspace=str(tmp_path),
            model="zhipu/glm-4.7",
            max_tokens=4096,
            temperature=0.1,
            max_tool_iterations=10,
            memory_window=2,
        )

        history = ConversationHistory("test-session", storage)
        for i in range(5):
            history.append({"role": "user", "content": f"Message {i}"})

        messages = builder.build(
            session_id="test-session",
            user_input="Final",
            agent_config=agent_config,
            history=history,
        )
        # History should be trimmed to 2 messages + system + user input
        assert len(messages) == 4  # system + 2 history + user

    def test_build_system_prompt_includes_agent_workspace(self, tmp_path) -> None:
        """Test that system prompt includes workspace from agent config."""
        from clawbot.agent.config import AgentRuntimeConfig

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage, default_workspace="/default")

        agent_config = AgentRuntimeConfig(
            name="test",
            workspace=str(tmp_path),
            model="zhipu/glm-4.7",
            max_tokens=4096,
            temperature=0.1,
            max_tool_iterations=10,
            memory_window=100,
        )

        messages = builder.build(
            session_id="test-session",
            user_input="Hello",
            agent_config=agent_config,
        )
        system_prompt = messages[0]["content"]
        assert str(tmp_path) in system_prompt
