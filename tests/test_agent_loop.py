"""Tests for SingleSessionAgentLoop conversation history management."""

from pathlib import Path

import pytest

from clawbot.agent.config import AgentRuntimeConfig
from clawbot.agent.context import ContextBuilder
from clawbot.agent.loop import SingleSessionAgentLoop
from clawbot.storage.session import SessionConfig, SessionStorage


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
    ) -> None:
        """Test that user messages are persisted to history, not just assistant responses."""
        session_id = "test-session-123"

        # Create a mock provider that returns fixed responses
        class MockProvider:
            async def chat(self, messages, model, max_tokens, temp):
                from clawbot.provider.base import LLMResponse
                is_user = messages[-1]["role"] == "user"
                last_content = messages[-1]["content"] if is_user else "unknown"
                return LLMResponse(
                    content=f"Response to: {last_content}",
                    tool_calls=None,
                    usage=None,
                )

        provider = MockProvider()

        loop = SingleSessionAgentLoop(
            session_id=session_id,
            agent_config=agent_config,
            storage=storage,
            context_builder=context_builder,
            provider=provider,
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
    ) -> None:
        """Test that history is correctly loaded when creating a new loop instance."""
        session_id = "test-session-456"

        class MockProvider:
            async def chat(self, messages, model, max_tokens, temp):
                from clawbot.provider.base import LLMResponse
                return LLMResponse(
                    content="OK",
                    tool_calls=None,
                    usage=None,
                )

        provider = MockProvider()

        # First loop instance
        loop1 = SingleSessionAgentLoop(
            session_id=session_id,
            agent_config=agent_config,
            storage=storage,
            context_builder=context_builder,
            provider=provider,
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
