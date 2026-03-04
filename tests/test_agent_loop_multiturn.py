"""Tests for SingleSessionAgentLoop multi-turn tool calling logic."""

from pathlib import Path
from typing import Any

import pytest

from clawbot.agent.config import AgentRuntimeConfig
from clawbot.agent.context import ContextBuilder
from clawbot.agent.loop import SingleSessionAgentLoop
from clawbot.provider.base import LLMResponse, ToolCallResult
from clawbot.storage.session import SessionConfig, SessionStorage
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
        self.last_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] | None = None

    def set_responses(self, responses: list[LLMResponse]) -> None:
        """Set the sequence of responses to return."""
        self.responses = responses
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
        self.last_messages = messages
        self.last_tools = tools
        self.call_count += 1

        if self.responses and self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]

        # Default response if no more responses configured
        return LLMResponse(
            content="Default response",
            tool_calls=None,
            finish_reason="stop",
        )


@pytest.fixture
def tmp_session_config(tmp_path: Path) -> SessionConfig:
    """Create test session config."""
    return SessionConfig(workspace=str(tmp_path))


@pytest.fixture
def tmp_storage(tmp_path: Path) -> SessionStorage:
    """Create test session storage."""
    config = SessionConfig(workspace=str(tmp_path))
    return SessionStorage(config=config)


@pytest.fixture
def tmp_context_builder(tmp_storage: SessionStorage, tmp_path: Path) -> ContextBuilder:
    """Create test context builder."""
    return ContextBuilder(storage=tmp_storage, default_workspace=str(tmp_path))


@pytest.fixture
def agent_config(tmp_path: Path) -> AgentRuntimeConfig:
    """Create test agent configuration."""
    return AgentRuntimeConfig(
        name="test",
        workspace=str(tmp_path),
        model="zhipu/glm-4.7",
        max_tokens=4096,
        temperature=0.1,
        max_tool_iterations=10,
        memory_window=100,
        max_steps=None,  # No limit by default
    )


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create test tool registry."""
    registry = ToolRegistry()
    registry.register(MockTool(name="test_tool", result="Test result"))
    return registry


class TestSingleSessionAgentLoopInit:
    """Tests for SingleSessionAgentLoop initialization."""

    def test_init_with_tool_registry(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that loop initializes with tool registry."""
        provider = MockProvider()

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        assert loop.tool_registry is tool_registry
        assert loop._step_count == 0

    def test_init_step_count_zero(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that step count starts at zero."""
        provider = MockProvider()

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        assert loop._step_count == 0


class TestShouldTerminate:
    """Tests for _should_terminate method."""

    @pytest.fixture
    def loop(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> SingleSessionAgentLoop:
        """Create test loop instance."""
        provider = MockProvider()
        return SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

    def test_terminate_when_tools_disabled(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test termination when tools are disabled."""
        response = LLMResponse(
            content="Response",
            tool_calls=[ToolCallResult(id="1", name="tool", arguments={})],
            finish_reason="tool_calls",
        )

        # Should terminate when can_use_tools is False
        assert loop._should_terminate(response, can_use_tools=False) is True

    def test_terminate_on_stop_finish_reason(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test termination on 'stop' finish reason."""
        response = LLMResponse(
            content="Response",
            tool_calls=None,
            finish_reason="stop",
        )

        assert loop._should_terminate(response, can_use_tools=True) is True

    def test_terminate_on_length_finish_reason(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test termination on 'length' finish reason."""
        response = LLMResponse(
            content="Response",
            tool_calls=None,
            finish_reason="length",
        )

        assert loop._should_terminate(response, can_use_tools=True) is True

    def test_continue_on_tool_calls_finish_reason(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test continuation on 'tool_calls' finish reason."""
        response = LLMResponse(
            content="Response",
            tool_calls=[ToolCallResult(id="1", name="tool", arguments={})],
            finish_reason="tool_calls",
        )

        assert loop._should_terminate(response, can_use_tools=True) is False

    def test_terminate_on_unknown_with_content_no_tools(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test termination on 'unknown' with content but no tool calls."""
        response = LLMResponse(
            content="Response",
            tool_calls=None,
            finish_reason="unknown",
        )

        assert loop._should_terminate(response, can_use_tools=True) is True

    def test_continue_on_unknown_with_tools(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test continuation on 'unknown' with tool calls."""
        response = LLMResponse(
            content="Response",
            tool_calls=[ToolCallResult(id="1", name="tool", arguments={})],
            finish_reason="unknown",
        )

        assert loop._should_terminate(response, can_use_tools=True) is False

    def test_terminate_on_other_finish_reason(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test termination on other finish reasons."""
        response = LLMResponse(
            content="Response",
            tool_calls=None,
            finish_reason="content_filter",
        )

        assert loop._should_terminate(response, can_use_tools=True) is True

    def test_terminate_with_none_finish_reason(
        self,
        loop: SingleSessionAgentLoop,
    ) -> None:
        """Test termination when finish_reason is None."""
        response = LLMResponse(
            content="Response",
            tool_calls=None,
            finish_reason="stop",  # Default is "stop"
        )

        assert loop._should_terminate(response, can_use_tools=True) is True


class TestMultiTurnToolCalling:
    """Tests for multi-turn tool calling logic."""

    @pytest.mark.asyncio
    async def test_single_tool_call(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test single tool call and response."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="I'll call the tool",
                tool_calls=[
                    ToolCallResult(id="call_1", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Tool executed successfully",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        response = await loop.run_turn("Test input")

        # Should have called LLM twice
        assert provider.call_count == 2

        # Final response should be from second call
        assert response.content == "Tool executed successfully"

        # Verify tools were passed to LLM
        assert provider.last_tools is not None
        assert len(provider.last_tools) > 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_tool_calls(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test multiple sequential tool calls."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Calling first tool",
                tool_calls=[
                    ToolCallResult(id="call_1", name="test_tool", arguments={"input": "first"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Calling second tool",
                tool_calls=[
                    ToolCallResult(id="call_2", name="test_tool", arguments={"input": "second"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="All tools executed",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        response = await loop.run_turn("Test input")

        # Should have called LLM three times
        assert provider.call_count == 3

        # Final response should be from third call
        assert response.content == "All tools executed"

    @pytest.mark.asyncio
    async def test_no_tool_calls(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test conversation without tool calls."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Hello! How can I help?",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        response = await loop.run_turn("Hello")

        # Should have called LLM once
        assert provider.call_count == 1
        assert response.content == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_tools_passed_to_llm(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that tool definitions are passed to LLM."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Response",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        await loop.run_turn("Test")

        # Verify tools were passed
        assert provider.last_tools is not None
        assert len(provider.last_tools) == 1
        assert provider.last_tools[0]["function"]["name"] == "test_tool"


class TestMaxStepsLimit:
    """Tests for max_steps configuration and limit handling."""

    @pytest.mark.asyncio
    async def test_max_steps_reached(
        self,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test behavior when max_steps is reached."""
        agent_config = AgentRuntimeConfig(
            name="test",
            workspace=str(tmp_storage.session_dir),
            model="zhipu/glm-4.7",
            max_tokens=4096,
            temperature=0.1,
            max_tool_iterations=10,
            memory_window=100,
            max_steps=2,  # Limit to 2 steps
        )

        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Step 1",
                tool_calls=[
                    ToolCallResult(id="call_1", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Step 2",
                tool_calls=[
                    ToolCallResult(id="call_2", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Step 3 - should not reach",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        await loop.run_turn("Test input")

        # Should have called LLM 3 times (2 steps + 1 summary)
        assert provider.call_count == 3

        # Last call should not have tools (summary mode)
        assert provider.last_tools is None

    @pytest.mark.asyncio
    async def test_max_steps_none_no_limit(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that max_steps=None means no limit."""
        agent_config.max_steps = None

        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Step 1",
                tool_calls=[
                    ToolCallResult(id="call_1", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Step 2",
                tool_calls=[
                    ToolCallResult(id="call_2", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Step 3",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        response = await loop.run_turn("Test input")

        # Should complete all 3 steps naturally
        assert provider.call_count == 3
        assert response.content == "Step 3"

    @pytest.mark.asyncio
    async def test_summary_prompt_injected(
        self,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that summary prompt is injected when max_steps reached."""
        agent_config = AgentRuntimeConfig(
            name="test",
            workspace=str(tmp_storage.session_dir),
            model="zhipu/glm-4.7",
            max_tokens=4096,
            temperature=0.1,
            max_tool_iterations=10,
            memory_window=100,
            max_steps=1,
        )

        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Step 1",
                tool_calls=[
                    ToolCallResult(id="call_1", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Summary of work completed",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        await loop.run_turn("Test input")

        # Verify summary prompt was added to history
        loop.history.load()
        messages = loop.history.messages

        # Find system prompt with summary notice
        summary_prompts = [
            m for m in messages
            if m.get("role") == "system" and "Maximum steps reached" in m.get("content", "")
        ]

        assert len(summary_prompts) == 1


class TestToolExecution:
    """Tests for tool execution logic."""

    @pytest.mark.asyncio
    async def test_tool_result_in_history(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that tool results are saved to history."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Calling tool",
                tool_calls=[
                    ToolCallResult(id="call_1", name="test_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Done",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        await loop.run_turn("Test input")

        # Verify tool response in history
        loop.history.load()
        tool_messages = [
            m for m in loop.history.messages
            if m.get("role") == "tool"
        ]

        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_1"
        assert "Test result" in tool_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_error_in_history(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
    ) -> None:
        """Test that tool errors are saved to history."""
        # Register a tool that returns error
        error_tool = MockTool(
            name="error_tool",
            result="Error: Something went wrong",
        )
        tool_registry = ToolRegistry()
        tool_registry.register(error_tool)

        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Calling error tool",
                tool_calls=[
                    ToolCallResult(id="call_1", name="error_tool", arguments={"input": "test"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Handling error",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        await loop.run_turn("Test input")

        # Verify error in history
        loop.history.load()
        tool_messages = [
            m for m in loop.history.messages
            if m.get("role") == "tool"
        ]

        assert len(tool_messages) == 1
        assert "Error" in tool_messages[0]["content"]


class TestStepCounting:
    """Tests for step counting logic."""

    @pytest.mark.asyncio
    async def test_step_count_increment(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that step count increments correctly."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Step 1",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        assert loop._step_count == 0

        await loop.run_turn("Test")

        # Step count should be 1 after one turn
        assert loop._step_count == 1

    @pytest.mark.asyncio
    async def test_step_count_reset_on_new_turn(
        self,
        agent_config: AgentRuntimeConfig,
        tmp_context_builder: ContextBuilder,
        tmp_storage: SessionStorage,
        tool_registry: ToolRegistry,
    ) -> None:
        """Test that step count resets on new run_turn call."""
        provider = MockProvider()
        provider.set_responses([
            LLMResponse(
                content="Response 1",
                tool_calls=None,
                finish_reason="stop",
            ),
            LLMResponse(
                content="Response 2",
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        loop = SingleSessionAgentLoop(
            session_id="test-session",
            agent_config=agent_config,
            storage=tmp_storage,
            context_builder=tmp_context_builder,
            provider=provider,
            tool_registry=tool_registry,
        )

        # First turn
        await loop.run_turn("Test 1")
        count_after_first = loop._step_count

        # Second turn
        await loop.run_turn("Test 2")
        count_after_second = loop._step_count

        # Both should have completed 1 step
        assert count_after_first == 1
        assert count_after_second == 1
