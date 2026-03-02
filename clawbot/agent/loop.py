"""The main loop for the Clawbot agent.

Dual-loop architecture with dual-queue design:
- InputQueue: Channel receives messages -> puts in input queue -> GlobalAgentLoop fetches
- OutputQueue: GlobalAgentLoop processes -> puts in output queue -> Channel fetches and sends

GlobalAgentLoop: Outer loop, global dispatcher, fetches from InputQueue
and dispatches to inner loops
SingleSessionAgentLoop: Inner loop, bound to single session, executes conversation flow
"""

from typing import TYPE_CHECKING

from clawbot.agent.context import ContextBuilder
from clawbot.agent.session import AgentRuntimeConfig
from clawbot.queue.queue import InputMessage, TaskQueueManager
from clawbot.storage.session import SessionStorage
from clawbot.util import QUEUE_TIMEOUT

if TYPE_CHECKING:
    from clawbot.config.schema import ClawbotConfig
    from clawbot.provider.base import BaseProvider, LLMResponse
    from clawbot.queue.queue import TaskQueueManager


class BaseAgentLoopHandler:
    """Base class for Agent Loop handlers - shared logic for GlobalAgentLoop and CliHandler."""

    def __init__(
        self,
        global_config: "ClawbotConfig",
        storage: SessionStorage,
        context_builder: ContextBuilder,
        providers: dict[str, "BaseProvider"],
    ):
        self.global_config = global_config
        self.storage = storage
        self.context_builder = context_builder
        self.providers = providers

        self._session_loops: dict[tuple[str, str], "SingleSessionAgentLoop"] = {}

    def _resolve_agent_config(self, agent_name: str) -> AgentRuntimeConfig:
        cfg = self.global_config.get_agent_config(agent_name)
        return AgentRuntimeConfig.from_agent_defaults(agent_name, cfg)

    def _resolve_provider(self, model: str) -> "BaseProvider":
        provider_name = self.global_config.get_provider_name(model)

        if provider_name in self.providers:
            return self.providers[provider_name]

        if self.providers:
            return next(iter(self.providers.values()))

        raise ValueError(f"No provider available for model: {model}")

    def _get_or_create_loop(
        self,
        session_id: str,
        agent_config: AgentRuntimeConfig,
    ) -> "SingleSessionAgentLoop":
        loop_key = (session_id, agent_config.name)

        if loop_key not in self._session_loops:
            provider = self._resolve_provider(agent_config.model)
            self._session_loops[loop_key] = SingleSessionAgentLoop(
                session_id=session_id,
                agent_config=agent_config,
                storage=self.storage,
                context_builder=self.context_builder,
                provider=provider,
            )

        return self._session_loops[loop_key]


class SingleSessionAgentLoop:
    """Inner loop - handles complete conversation flow for a single session."""

    def __init__(
        self,
        session_id: str,
        agent_config: AgentRuntimeConfig,
        storage: SessionStorage,
        context_builder: ContextBuilder,
        provider: "BaseProvider",
    ):
        self.session_id = session_id
        self.agent_config = agent_config
        self.storage = storage
        self.context_builder = context_builder
        self.provider = provider

        self.history = context_builder.create_history(session_id)

    async def run_turn(self, user_input: str) -> "LLMResponse":
        """Execute a single conversation turn."""
        self.history.load()

        messages = self.context_builder.build(
            session_id=self.session_id,
            user_input=user_input,
            agent_config=self.agent_config,
            history=self.history,
        )

        response = await self.provider.chat(
            messages=messages,
            model=self.agent_config.model,
            max_tokens=self.agent_config.max_tokens,
            temp=self.agent_config.temperature,
        )

        self.history.append_assistant_response(
            content=response.content,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in (response.tool_calls or [])
            ],
        )

        return response

    async def run_tool_turn(
        self,
        tool_results: list[dict],
    ) -> "LLMResponse":
        """Execute tool response turn (subsequent turn in multi-round tool calls)."""
        for result in tool_results:
            self.history.append_tool_response(
                tool_call_id=result["tool_call_id"],
                result=result["content"],
                error=result.get("error"),
            )

        messages = self.context_builder.build(
            session_id=self.session_id,
            user_input="",
            agent_config=self.agent_config,
            history=self.history,
        )

        response = await self.provider.chat(
            messages=messages,
            model=self.agent_config.model,
            max_tokens=self.agent_config.max_tokens,
            temp=self.agent_config.temperature,
        )

        if response.content:
            self.history.append_assistant_response(
                content=response.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in (response.tool_calls or [])
                ],
            )

        return response


class GlobalAgentLoop(BaseAgentLoopHandler):
    """Outer loop - global dispatcher."""

    def __init__(
        self,
        global_config: "ClawbotConfig",
        storage: SessionStorage,
        context_builder: ContextBuilder,
        providers: dict[str, "BaseProvider"],
        queue_manager: "TaskQueueManager",
    ):
        super().__init__(
            global_config,
            storage,
            context_builder,
            providers,
        )
        self.queue_manager = queue_manager
        self.running = False

    async def dispatch(
        self,
        msg: "InputMessage",
    ) -> "LLMResponse | None":
        """Dispatch InputMessage to inner loop."""
        session_id = msg.session_id
        agent_config = self._resolve_agent_config(msg.agent_name)
        loop = self._get_or_create_loop(session_id, agent_config)
        response = await loop.run_turn(str(msg.content))
        return response

    async def run(self):
        """Main loop: fetch from InputQueue, process, send to OutputQueue."""
        self.running = True

        while self.running:
            try:
                msg = await self.queue_manager.get_input_with_timeout(QUEUE_TIMEOUT)
                if msg is None:
                    continue

                try:
                    response = await self.dispatch(msg)

                    if response:
                        await self.queue_manager.send_output(
                            session_id=msg.session_id,
                            channel_id=msg.channel_id,
                            channel_session_id=msg.channel_session_id or "",
                            content=response.content,
                            success=True,
                        )
                    else:
                        await self.queue_manager.send_output(
                            session_id=msg.session_id,
                            channel_id=msg.channel_id,
                            channel_session_id=msg.channel_session_id or "",
                            content=None,
                            success=False,
                            error="No response from agent",
                        )

                except Exception as e:
                    await self.queue_manager.send_output(
                        session_id=msg.session_id,
                        channel_id=msg.channel_id,
                        channel_session_id=msg.channel_session_id or "",
                        content=None,
                        success=False,
                        error=str(e),
                    )

                finally:
                    self.queue_manager.task_done()

            except Exception as e:
                print(f"GlobalAgentLoop error: {e}")

    def stop(self):
        """Stop main loop."""
        self.running = False


class CliHandler(BaseAgentLoopHandler):
    """CLI handler - processes CLI direct input (bypasses queues)."""

    def __init__(
        self,
        global_config: "ClawbotConfig",
        storage: SessionStorage,
        context_builder: ContextBuilder,
        providers: dict[str, "BaseProvider"],
    ):
        super().__init__(
            global_config,
            storage,
            context_builder,
            providers,
        )

    def _get_or_create_loop(
        self,
        session_id: str,
        agent_name: str,
    ) -> SingleSessionAgentLoop:
        agent_config = self._resolve_agent_config(agent_name)
        return super()._get_or_create_loop(session_id, agent_config)

    async def run_turn(
        self,
        session_id: str,
        agent_name: str,
        user_input: str,
    ) -> str:
        """Execute single CLI conversation turn."""
        loop = self._get_or_create_loop(session_id, agent_name)
        response = await loop.run_turn(user_input)
        return response.content if response else "(no response)"
