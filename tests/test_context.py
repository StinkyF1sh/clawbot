"""Tests for agent context management module."""

from clawbot.agent.context import (
    ConversationHistory,
    ContextBuilder,
    get_system_prompt,
)


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

    def test_default_max_window(self) -> None:
        """Test default max window size."""
        history = ConversationHistory()
        assert history.max_window == 100
        assert history.messages == []

    def test_custom_max_window(self) -> None:
        """Test custom max window size."""
        history = ConversationHistory(max_window=50)
        assert history.max_window == 50


class TestConversationHistoryAppend:
    """Tests for ConversationHistory append methods."""

    def test_append_single_message(self) -> None:
        """Test appending a single message."""
        history = ConversationHistory()
        history.append({"role": "user", "content": "Hello"})
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"

    def test_append_multiple_messages(self) -> None:
        """Test appending multiple messages."""
        history = ConversationHistory()
        history.append({"role": "user", "content": "Hello"})
        history.append({"role": "assistant", "content": "Hi there!"})
        assert len(history.messages) == 2

    def test_extend_messages(self) -> None:
        """Test extending with multiple messages at once."""
        history = ConversationHistory()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        history.extend(messages)
        assert len(history.messages) == 2

    def test_clear_history(self) -> None:
        """Test clearing history."""
        history = ConversationHistory()
        history.append({"role": "user", "content": "Hello"})
        history.clear()
        assert len(history.messages) == 0


class TestConversationHistoryTrim:
    """Tests for ConversationHistory trim_to_window method."""

    def test_trim_no_op_when_under_limit(self) -> None:
        """Test trim does nothing when under limit."""
        history = ConversationHistory(max_window=100)
        history.append({"role": "user", "content": "Hello"})
        history.trim_to_window()
        assert len(history.messages) == 1

    def test_trim_keeps_latest_messages(self) -> None:
        """Test trim keeps only the latest messages."""
        history = ConversationHistory(max_window=3)
        for i in range(5):
            history.append({"role": "user", "content": f"Message {i}"})
        history.trim_to_window()
        assert len(history.messages) == 3
        # Should keep the last 3 messages
        assert history.messages[0]["content"] == "Message 2"
        assert history.messages[2]["content"] == "Message 4"

    def test_trim_with_custom_window(self) -> None:
        """Test trim with custom window size."""
        history = ConversationHistory(max_window=100)
        for i in range(10):
            history.append({"role": "user", "content": f"Message {i}"})
        history.trim_to_window(max_window=5)
        assert len(history.messages) == 5
        assert history.messages[0]["content"] == "Message 5"


class TestConversationHistoryStorage:
    """Tests for ConversationHistory storage methods."""

    def test_append_and_save(self, tmp_path) -> None:
        """Test appending message and saving to storage."""
        from clawbot.storage.session import SessionConfig, SessionStorage

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session-123"

        history = ConversationHistory()
        history.append_and_save(
            {"role": "user", "content": "Hello"},
            session_id,
            storage,
        )

        # Verify message was saved
        loaded = storage.load_session(session_id)
        assert len(loaded) == 1
        assert loaded[0]["role"] == "user"

    def test_append_assistant_response_without_tool_calls(self) -> None:
        """Test appending assistant response without tool calls."""
        history = ConversationHistory()
        history.append_assistant_response(content="Hello!")
        assert len(history.messages) == 1
        assert history.messages[0] == {"role": "assistant", "content": "Hello!"}

    def test_append_assistant_response_with_tool_calls(self) -> None:
        """Test appending assistant response with tool calls."""
        history = ConversationHistory()
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        history.append_assistant_response(
            content="Let me check",
            tool_calls=tool_calls,
        )
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "assistant"
        assert history.messages[0]["content"] == "Let me check"
        assert history.messages[0]["tool_calls"] == tool_calls

    def test_append_tool_response(self) -> None:
        """Test appending tool response."""
        history = ConversationHistory()
        history.append_tool_response(
            tool_call_id="call_1",
            result="Weather is sunny",
        )
        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg["role"] == "tool"
        assert msg["content"] == "Weather is sunny"
        assert msg["tool_call_id"] == "call_1"

    def test_append_tool_response_with_error(self) -> None:
        """Test appending tool response with error."""
        history = ConversationHistory()
        history.append_tool_response(
            tool_call_id="call_1",
            result=None,
            error="API timeout",
        )
        assert history.messages[0]["content"] == "Error: API timeout"


class TestContextBuilderInit:
    """Tests for ContextBuilder initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        builder = ContextBuilder()
        assert builder.workspace is None
        assert builder.max_window == 100
        assert builder.storage is None

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        builder = ContextBuilder(
            workspace="/tmp/workspace",
            max_window=50,
        )
        assert builder.workspace == "/tmp/workspace"
        assert builder.max_window == 50


class TestContextBuilderCreateHistory:
    """Tests for ContextBuilder history creation methods."""

    def test_create_history(self) -> None:
        """Test creating new empty history."""
        builder = ContextBuilder()
        history = builder.create_history()
        assert isinstance(history, ConversationHistory)
        assert len(history.messages) == 0

    def test_create_history_with_custom_window(self) -> None:
        """Test creating history with custom window."""
        builder = ContextBuilder(max_window=100)
        history = builder.create_history(max_window=50)
        assert history.max_window == 50

    def test_load_history_without_storage(self) -> None:
        """Test loading history without storage raises error."""
        builder = ContextBuilder()
        try:
            builder.load_history("session-123")
        except ValueError as e:
            assert "SessionStorage not configured" in str(e)

    def test_load_history_with_storage(self, tmp_path) -> None:
        """Test loading history with storage."""
        from clawbot.storage.session import SessionConfig, SessionStorage

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage)

        # First save a message
        session_id = "test-session"
        storage.append_message(session_id, {"role": "user", "content": "Hello"})

        # Then load
        history = builder.load_history(session_id)
        assert len(history.messages) == 1


class TestContextBuilderBuild:
    """Tests for ContextBuilder build method."""

    def test_build_with_only_user_input(self) -> None:
        """Test building messages with only user input."""
        builder = ContextBuilder()
        messages = builder.build(current_user_input="Hello")
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_with_history(self) -> None:
        """Test building messages with conversation history."""
        builder = ContextBuilder()
        history = ConversationHistory()
        history.append({"role": "user", "content": "Previous"})
        history.append({"role": "assistant", "content": "Response"})

        messages = builder.build(
            current_user_input="New message",
            history=history,
        )
        assert len(messages) == 4  # system + 2 history + user
        assert messages[1]["content"] == "Previous"
        assert messages[2]["content"] == "Response"

    def test_build_with_config(self, tmp_path) -> None:
        """Test building messages with SessionConfig."""
        from clawbot.storage.session import SessionConfig

        config = SessionConfig(
            workspace=str(tmp_path),
            max_window=50,
        )
        builder = ContextBuilder(workspace="/default/workspace")

        messages = builder.build(
            current_user_input="Hello",
            config=config,
        )
        # Check workspace from config is used in system prompt
        assert str(tmp_path) in messages[0]["content"]

    def test_build_trims_history_based_on_config(self) -> None:
        """Test that history is trimmed based on config."""
        from clawbot.storage.session import SessionConfig

        config = SessionConfig(workspace="/tmp", max_window=2)
        builder = ContextBuilder()

        history = ConversationHistory(max_window=100)
        for i in range(5):
            history.append({"role": "user", "content": f"Message {i}"})

        messages = builder.build(
            current_user_input="Final",
            history=history,
            config=config,
        )
        # History should be trimmed to 2 messages + system + user input
        assert len(messages) == 4  # system + 2 history + user

    def test_build_system_prompt_includes_workspace(self, tmp_path) -> None:
        """Test that system prompt includes workspace from config."""
        from clawbot.storage.session import SessionConfig

        config = SessionConfig(workspace=str(tmp_path))
        builder = ContextBuilder()

        messages = builder.build(
            current_user_input="Hello",
            config=config,
        )
        system_prompt = messages[0]["content"]
        assert str(tmp_path) in system_prompt
