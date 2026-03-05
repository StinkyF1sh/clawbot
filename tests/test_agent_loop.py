"""Tests for SingleSessionAgentLoop conversation history management."""

from pathlib import Path
from typing import Any

import pytest

from clawbot.agent.config import AgentRuntimeConfig
from clawbot.agent.context import ContextBuilder
from clawbot.agent.loop import CliHandler, SingleSessionAgentLoop
from clawbot.config.schema import AgentsConfig, ClawbotConfig
from clawbot.provider.base import LLMResponse
from clawbot.storage.session import SessionConfig, SessionStorage
from clawbot.tools.filesystem import ReadFileTool
from clawbot.tools.registry import ToolRegistry


class MockTool:
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool",
        parameters: dict[str, Any] | None = None,
        result: str = "Mock result",
    ):
        self._name = name
        self._description = description
        self._parameters = parameters or {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        return self._result

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        return []


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        temp: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Simulate LLM chat response."""
        self.call_count += 1

        if self.responses and self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]

        return LLMResponse(
            content="Default response",
            tool_calls=None,
            finish_reason="stop",
        )


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create test tool registry."""
    registry = ToolRegistry()
    registry.register(MockTool(name="test_tool", result="Test result"))
    return registry


class TestSingleSessionAgentLoopHistory:
    """Tests for SingleSessionAgentLoop conversation history persistence."""

    @pytest.fixture
    def agent_config(self, tmp_path: Path) -> AgentRuntimeConfig:
        """Create test agent configuration."""
        return AgentRuntimeConfig(
            name="test",
            workspace=str(tmp_path),
            model="zhipu/glm-4.7",
            max_tokens=4096,
            temperature=0.1,
            max_tool_iterations=10,
            memory_window=100,
        )

    @pytest.fixture
    def storage(self, tmp_path: Path) -> SessionStorage:
        """Create test session storage."""
        config = SessionConfig(workspace=str(tmp_path))
        return SessionStorage(config=config)

    @pytest.fixture
    def context_builder(self, storage: SessionStorage, tmp_path: Path) -> ContextBuilder:
        """Create test context builder."""
        return ContextBuilder(storage=storage, default_workspace=str(tmp_path))

    def test_user_message_persisted_to_history(
        self,
        tmp_path: Path,
        agent_config: AgentRuntimeConfig,
        storage: SessionStorage,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that user messages are persisted to history, not just assistant responses."""
        session_id = "test-session-123"

        provider = MockProvider([
            LLMResponse(content="Response to: Hello", tool_calls=None, finish_reason="stop"),
            LLMResponse(content="Response to: Who are you?", tool_calls=None, finish_reason="stop"),
        ])

        loop = SingleSessionAgentLoop(
            session_id=session_id,
            agent_config=agent_config,
            storage=storage,
            context_builder=context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        # First turn
        import asyncio
        response1 = asyncio.run(loop.run_turn("Hello"))
        assert "Response to: Hello" in response1.content

        # Second turn
        response2 = asyncio.run(loop.run_turn("Who are you?"))
        assert "Response to: Who are you?" in response2.content

        # Verify both user messages and assistant responses are in storage
        saved_messages = storage.load_session(session_id)

        # Should have: user1, assistant1, user2, assistant2
        msg_count = len(saved_messages)
        assert msg_count == 4, f"Expected 4 messages, got {msg_count}: {saved_messages}"

        # Verify message order and roles
        assert saved_messages[0]["role"] == "user"
        assert saved_messages[0]["content"] == "Hello"

        assert saved_messages[1]["role"] == "assistant"
        assert "Response to: Hello" in saved_messages[1]["content"]

        assert saved_messages[2]["role"] == "user"
        assert saved_messages[2]["content"] == "Who are you?"

        assert saved_messages[3]["role"] == "assistant"
        assert "Response to: Who are you?" in saved_messages[3]["content"]

    def test_history_loaded_correctly_on_new_loop(
        self,
        tmp_path: Path,
        agent_config: AgentRuntimeConfig,
        storage: SessionStorage,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that history is correctly loaded when creating a new loop instance."""
        session_id = "test-session-456"

        provider = MockProvider([
            LLMResponse(content="OK", tool_calls=None, finish_reason="stop"),
            LLMResponse(content="OK", tool_calls=None, finish_reason="stop"),
        ])

        # First loop instance
        loop1 = SingleSessionAgentLoop(
            session_id=session_id,
            agent_config=agent_config,
            storage=storage,
            context_builder=context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        import asyncio
        asyncio.run(loop1.run_turn("First message"))
        asyncio.run(loop1.run_turn("Second message"))

        # Create a new loop instance (simulating CLI handler cache miss)
        loop2 = SingleSessionAgentLoop(
            session_id=session_id,
            agent_config=agent_config,
            storage=storage,
            context_builder=context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        # Verify the new loop can load the history
        loop2.history.load()
        saved_messages = storage.load_session(session_id)

        # Should have 4 messages: user1, assistant1, user2, assistant2
        assert len(saved_messages) == 4

        # Verify user messages are present
        user_messages = [m for m in saved_messages if m["role"] == "user"]
        assert len(user_messages) == 2
        assert user_messages[0]["content"] == "First message"
        assert user_messages[1]["content"] == "Second message"


class TestCliHandlerToolRegistryBinding:
    """Tests for session+agent bound tool registry behavior."""

    def _create_handler(
        self,
        default_workspace: Path,
        tool_registry: ToolRegistry | None = None,
        tool_registry_factory: Any | None = None,
    ) -> CliHandler:
        config = ClawbotConfig(
            agents=AgentsConfig.model_validate(
                {
                    "default": {"workspace": str(default_workspace), "model": "zhipu/glm-4.7"},
                    "code": {"workspace": str(default_workspace / "code"), "model": "zhipu/glm-4.7"},
                }
            )
        )
        storage = SessionStorage(config=SessionConfig(workspace=str(default_workspace)))
        context_builder = ContextBuilder(storage=storage, default_workspace=str(default_workspace))

        return CliHandler(
            global_config=config,
            storage=storage,
            context_builder=context_builder,
            providers={"mock": MockProvider()},
            tool_registry=tool_registry,
            tool_registry_factory=tool_registry_factory,
        )

    def test_registry_is_bound_by_session_and_agent(self, tmp_path: Path) -> None:
        """Different agents under same session should get separate registries."""
        default_workspace = tmp_path / "default"
        code_workspace = default_workspace / "code"
        default_workspace.mkdir(parents=True)
        code_workspace.mkdir(parents=True)

        calls: list[str] = []

        def registry_factory(agent_config: AgentRuntimeConfig) -> ToolRegistry:
            calls.append(agent_config.name)
            workspace = Path(agent_config.workspace)
            registry = ToolRegistry()
            registry.register(ReadFileTool(workspace=workspace, allowed_dir=workspace))
            return registry

        handler = self._create_handler(
            default_workspace=default_workspace,
            tool_registry=ToolRegistry(),
            tool_registry_factory=registry_factory,
        )

        default_loop = handler._get_or_create_loop("session-1", "default")
        code_loop = handler._get_or_create_loop("session-1", "code")

        assert default_loop is not code_loop
        assert default_loop.tool_registry is not code_loop.tool_registry
        assert calls == ["default", "code"]

        default_reader = default_loop.tool_registry.get("read_file")
        code_reader = code_loop.tool_registry.get("read_file")
        assert isinstance(default_reader, ReadFileTool)
        assert isinstance(code_reader, ReadFileTool)
        assert default_reader.workspace == default_workspace
        assert code_reader.workspace == code_workspace

    def test_loop_reuses_registry_for_same_session_agent(self, tmp_path: Path) -> None:
        """Same session+agent should reuse the cached loop and registry."""
        default_workspace = tmp_path / "default"
        (default_workspace / "code").mkdir(parents=True)

        call_count = 0

        def registry_factory(agent_config: AgentRuntimeConfig) -> ToolRegistry:
            nonlocal call_count
            call_count += 1
            workspace = Path(agent_config.workspace)
            registry = ToolRegistry()
            registry.register(ReadFileTool(workspace=workspace, allowed_dir=workspace))
            return registry

        handler = self._create_handler(
            default_workspace=default_workspace,
            tool_registry=ToolRegistry(),
            tool_registry_factory=registry_factory,
        )

        loop1 = handler._get_or_create_loop("session-2", "code")
        loop2 = handler._get_or_create_loop("session-2", "code")

        assert loop1 is loop2
        assert call_count == 1

    def test_static_registry_path_still_works_without_factory(self, tmp_path: Path) -> None:
        """Compatibility: when no factory is provided, static registry is used."""
        default_workspace = tmp_path / "default"
        (default_workspace / "code").mkdir(parents=True)

        shared_registry = ToolRegistry()
        shared_registry.register(MockTool(name="shared_tool"))

        handler = self._create_handler(
            default_workspace=default_workspace,
            tool_registry=shared_registry,
            tool_registry_factory=None,
        )

        default_loop = handler._get_or_create_loop("session-3", "default")
        code_loop = handler._get_or_create_loop("session-3", "code")

        assert default_loop.tool_registry is shared_registry
        assert code_loop.tool_registry is shared_registry
