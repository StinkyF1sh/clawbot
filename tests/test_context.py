"""Tests for agent context management module."""

from pathlib import Path

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

    def test_system_prompt_injects_workspace_files_in_order(self, tmp_path) -> None:
        """Test workspace markdown files are injected in the expected order."""
        ordered_files = [
            "AGENTS.md",
            "SOUL.md",
            "TOOLS.md",
            "IDENTITY.md",
            "USER.md",
            "HEARTBEAT.md",
            "BOOTSTRAP.md",
            "MEMORY.md",
        ]
        for i, file_name in enumerate(ordered_files):
            (tmp_path / file_name).write_text(f"{file_name} content {i}", encoding="utf-8")

        prompt = get_system_prompt(str(tmp_path))

        assert "## Project Context" in prompt
        assert "## Workspace Files (injected)" in prompt

        positions = [prompt.find(f"### {name}") for name in ordered_files]
        assert all(pos != -1 for pos in positions)
        assert positions == sorted(positions)

    def test_system_prompt_skips_bootstrap_when_flag_false(self, tmp_path) -> None:
        """Test BOOTSTRAP.md is skipped when include_bootstrap is False."""
        (tmp_path / "BOOTSTRAP.md").write_text("bootstrap content", encoding="utf-8")

        prompt = get_system_prompt(str(tmp_path), include_bootstrap=False)

        assert "### BOOTSTRAP.md" not in prompt
        assert "bootstrap content" not in prompt

    def test_system_prompt_loads_memory_uppercase(self, tmp_path) -> None:
        """Test MEMORY.md is loaded when present."""
        (tmp_path / "MEMORY.md").write_text("uppercase memory", encoding="utf-8")

        prompt = get_system_prompt(str(tmp_path))

        assert "### MEMORY.md" in prompt
        assert "uppercase memory" in prompt

    def test_system_prompt_ignores_memory_lowercase_only(self, tmp_path) -> None:
        """Test memory.md alone is ignored due to exact-case filename matching."""
        (tmp_path / "memory.md").write_text("lowercase memory", encoding="utf-8")

        prompt = get_system_prompt(str(tmp_path))

        assert "### MEMORY.md" not in prompt
        assert "lowercase memory" not in prompt

    def test_system_prompt_handles_missing_bootstrap_file(self, tmp_path) -> None:
        """Test missing BOOTSTRAP.md is handled without affecting other injections."""
        (tmp_path / "AGENTS.md").write_text("agents content", encoding="utf-8")

        prompt = get_system_prompt(str(tmp_path))

        assert "### AGENTS.md" in prompt
        assert "agents content" in prompt
        assert "### BOOTSTRAP.md" not in prompt

    def test_system_prompt_skips_unreadable_bootstrap_file(self, tmp_path, monkeypatch) -> None:
        """Test unreadable BOOTSTRAP.md is skipped safely."""
        (tmp_path / "AGENTS.md").write_text("agents content", encoding="utf-8")
        (tmp_path / "BOOTSTRAP.md").write_text("bootstrap content", encoding="utf-8")

        original_read_text = Path.read_text

        def _mock_read_text(self: Path, *args, **kwargs) -> str:
            if self.name == "BOOTSTRAP.md":
                raise OSError("denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _mock_read_text)

        prompt = get_system_prompt(str(tmp_path))

        assert "### AGENTS.md" in prompt
        assert "agents content" in prompt
        assert "### BOOTSTRAP.md" not in prompt
        assert "bootstrap content" not in prompt

    def test_system_prompt_includes_available_skills(self) -> None:
        """Test available_skills section is injected into prompt."""
        prompt = get_system_prompt(
            workspace=None,
            available_skills=[
                ("code_review", "Review code for bugs"),
                ("writer", "Draft concise text"),
            ],
        )

        assert "## Available Skills" in prompt
        assert "<available_skills>" in prompt
        assert '<skill name="code_review">Review code for bugs</skill>' in prompt
        assert '<skill name="writer">Draft concise text</skill>' in prompt
        assert "</available_skills>" in prompt


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

    def test_history_load_creates_session_when_missing(self, tmp_path) -> None:
        """Test load creates a session file when it does not exist."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "new-session"

        history = ConversationHistory(session_id, storage)
        history.load()

        assert history.is_new_session is True
        assert storage.get_session_meta(session_id) is not None

    def test_history_load_does_not_recreate_existing_session(self, tmp_path) -> None:
        """Test load does not mark existing sessions as new."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "existing-session"

        storage.create_session(session_id)
        before_meta = storage.get_session_meta(session_id)
        assert before_meta is not None

        history = ConversationHistory(session_id, storage)
        history.load()

        after_meta = storage.get_session_meta(session_id)
        assert after_meta is not None
        assert history.is_new_session is False
        assert abs(after_meta.created_at - before_meta.created_at) < 1.0

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
            memory_window=100,
        )

        messages = builder.build(
            session_id="test-session",
            user_input="Hello",
            agent_config=agent_config,
        )
        system_prompt = messages[0]["content"]
        assert str(tmp_path) in system_prompt

    def test_build_bootstrap_only_once_for_new_session(self, tmp_path) -> None:
        """Test BOOTSTRAP.md is injected only once for new sessions."""
        (tmp_path / "BOOTSTRAP.md").write_text("bootstrap content", encoding="utf-8")

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage, default_workspace=str(tmp_path))
        history = builder.create_history("new-session")
        history.load()

        first_messages = builder.build(
            session_id="new-session",
            user_input="hello",
            history=history,
        )
        second_messages = builder.build(
            session_id="new-session",
            user_input="hello again",
            history=history,
        )

        assert "### BOOTSTRAP.md" in first_messages[0]["content"]
        assert "bootstrap content" in first_messages[0]["content"]
        assert "### BOOTSTRAP.md" not in second_messages[0]["content"]

    def test_build_existing_session_never_injects_bootstrap(self, tmp_path) -> None:
        """Test existing sessions never inject BOOTSTRAP.md."""
        (tmp_path / "BOOTSTRAP.md").write_text("bootstrap content", encoding="utf-8")

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage, default_workspace=str(tmp_path))
        session_id = "existing-session"

        storage.append_message(session_id, {"role": "user", "content": "already there"})
        history = builder.create_history(session_id)
        history.load()

        messages = builder.build(
            session_id=session_id,
            user_input="new input",
            history=history,
        )

        assert "### BOOTSTRAP.md" not in messages[0]["content"]

    def test_build_without_history_never_injects_bootstrap(self, tmp_path) -> None:
        """Test build(history=None) does not inject BOOTSTRAP.md."""
        (tmp_path / "BOOTSTRAP.md").write_text("bootstrap content", encoding="utf-8")

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(storage=storage, default_workspace=str(tmp_path))

        messages = builder.build(
            session_id="new-session",
            user_input="hello",
            history=None,
        )

        assert messages[0]["role"] == "system"
        assert "### BOOTSTRAP.md" not in messages[0]["content"]

    def test_build_injects_skill_catalog_from_provider(self, tmp_path) -> None:
        """Test ContextBuilder injects catalog entries from skill provider."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        builder = ContextBuilder(
            storage=storage,
            default_workspace=str(tmp_path),
            skill_catalog_provider=lambda _: [("writer", "Draft concise text")],
        )

        messages = builder.build(
            session_id="test-session",
            user_input="hello",
        )

        system_prompt = messages[0]["content"]
        assert "<available_skills>" in system_prompt
        assert '<skill name="writer">Draft concise text</skill>' in system_prompt
