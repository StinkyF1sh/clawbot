"""The main loop for the Clawbot agent.

Dual-loop architecture with dual-queue design:
- InputQueue: Channel receives messages -> puts in input queue -> GlobalAgentLoop fetches
- OutputQueue: GlobalAgentLoop processes -> puts in output queue -> Channel fetches and sends

GlobalAgentLoop: Outer loop, global dispatcher, fetches from InputQueue
and dispatches to inner loops
SingleSessionAgentLoop: Inner loop, bound to single session, executes conversation flow
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from clawbot.agent.config import AgentRuntimeConfig
from clawbot.agent.context import ContextBuilder
from clawbot.queue.constants import QUEUE_TIMEOUT
from clawbot.queue.queue import InputMessage, TaskQueueManager
from clawbot.storage.session import SessionStorage
from clawbot.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from clawbot.config.schema import ClawbotConfig
    from clawbot.provider.base import BaseProvider, LLMResponse, ToolCallResult
    from clawbot.queue.queue import TaskQueueManager


class BaseAgentLoopHandler:
    """Base class for Agent Loop handlers - shared logic for GlobalAgentLoop and CliHandler."""

    def __init__(
        self,
        global_config: "ClawbotConfig",
        storage: SessionStorage,
        context_builder: ContextBuilder,
        providers: dict[str, "BaseProvider"],
        tool_registry: ToolRegistry | None,
        tool_registry_factory: Callable[[AgentRuntimeConfig], ToolRegistry] | None = None,
    ):
        self.global_config = global_config
        self.storage = storage
        self.context_builder = context_builder
        self.providers = providers
        self.tool_registry = tool_registry
        self.tool_registry_factory = tool_registry_factory

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

    def _resolve_tool_registry(self, agent_config: AgentRuntimeConfig) -> ToolRegistry:
        if self.tool_registry_factory:
            return self.tool_registry_factory(agent_config)
        return self.tool_registry or ToolRegistry()

    def _get_or_create_loop(
        self,
        session_id: str,
        agent_config: AgentRuntimeConfig,
    ) -> "SingleSessionAgentLoop":
        loop_key = (session_id, agent_config.name)

        if loop_key not in self._session_loops:
            provider = self._resolve_provider(agent_config.model)
            tool_registry = self._resolve_tool_registry(agent_config)
            self._session_loops[loop_key] = SingleSessionAgentLoop(
                session_id=session_id,
                agent_config=agent_config,
                storage=self.storage,
                context_builder=self.context_builder,
                provider=provider,
                tool_registry=tool_registry,
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
        tool_registry: ToolRegistry | None,
    ):
        self.session_id = session_id
        self.agent_config = agent_config
        self.storage = storage
        self.context_builder = context_builder
        self.provider = provider
        self.tool_registry = tool_registry

        self.history = context_builder.create_history(session_id)

        self._step_count = 0

    async def run_turn(self, user_input: str) -> "LLMResponse":
        """
        Execute a single conversation turn with multi-round tool calls.

        Loop terminates when:
        - finish_reason is not "tool_calls" or "unknown" AND assistant completed
        - max_steps reached (then inject summary prompt)
        """
        self.history.load()

        self.history.save({"role": "user", "content": user_input})

        can_use_tools = True
        self._step_count = 0
        last_response: "LLMResponse" | None = None

        while True:
            max_steps = self.agent_config.max_steps
            if max_steps is not None and self._step_count >= max_steps:
                can_use_tools = False
                await self._inject_system_prompt_for_summary()

            response = await self._run_step(can_use_tools=can_use_tools)

            last_response = response
            self._step_count += 1

            if self._should_terminate(response, can_use_tools):
                break

        return last_response

    def _should_terminate(
        self,
        response: "LLMResponse",
        can_use_tools: bool,
    ) -> bool:
        """
        Determine if the loop should terminate.

        Termination conditions:
        1. can_use_tools is False (already in summary mode)
        2. finish_reason is "stop" or "length" (normal completion)
        3. finish_reason is "unknown" but no tool_calls and has content
        4. finish_reason is not "tool_calls" (other completion states)
        """
        if not can_use_tools:
            return True

        finish_reason = response.finish_reason or "stop"

        if finish_reason in ("stop", "length"):
            return True

        if finish_reason == "tool_calls":
            return False

        if finish_reason == "unknown":
            if response.content and not response.tool_calls:
                return True
            return False

        return True

    async def _run_step(
        self,
        can_use_tools: bool,
    ) -> "LLMResponse":
        """Execute a single step (LLM call + optional tool execution)."""
        messages = self.context_builder.build(
            session_id=self.session_id,
            user_input="",
            agent_config=self.agent_config,
            history=self.history,
        )

        tools = None
        if can_use_tools and self.tool_registry:
            tools = self.tool_registry.get_definitions()

        response = await self.provider.chat(
            messages=messages,
            model=self.agent_config.model_name,
            max_tokens=self.agent_config.max_tokens,
            temp=self.agent_config.temperature,
            tools=tools,
        )

        if not response.tool_calls or not can_use_tools:
            if response.content:
                self.history.append_assistant_response(
                    content=response.content,
                    tool_calls=None,
                )
            return response

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
                for tc in response.tool_calls
            ],
        )

        tool_results = await self._execute_tools(response.tool_calls)

        for result in tool_results:
            self.history.append_tool_response(
                tool_call_id=result["tool_call_id"],
                result=result["content"],
                error=result.get("error"),
            )

        return response

    async def _execute_tools(
        self,
        tool_calls: list["ToolCallResult"],
    ) -> list[dict]:
        """Execute multiple tool calls and return results."""
        results = []
        for tc in tool_calls:
            result = await self.tool_registry.execute(tc.name, tc.arguments)
            results.append({
                "tool_call_id": tc.id,
                "content": result,
                "error": None if not result.startswith("Error") else "Tool execution failed",
            })
        return results

    async def _inject_system_prompt_for_summary(self) -> None:
        """
        Inject system prompt to disable tools and request text summary.

        Called when max_steps is reached.
        """
        summary_prompt = {
            "role": "system",
            "content": (
                "[SYSTEM NOTICE: Maximum steps reached] "
                "Please provide a final summary of the work completed so far. "
                "Do NOT use any tools - respond with plain text only. "
                "Summarize: (1) what has been accomplished, "
                "(2) key findings or results, "
                "(3) any remaining tasks or recommendations."
            ),
        }
        self.history.append(summary_prompt)

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

        tools = self.tool_registry.get_definitions() if self.tool_registry else None

        response = await self.provider.chat(
            messages=messages,
            model=self.agent_config.model_name,
            max_tokens=self.agent_config.max_tokens,
            temp=self.agent_config.temperature,
            tools=tools,
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
        tool_registry: ToolRegistry | None,
        tool_registry_factory: Callable[[AgentRuntimeConfig], ToolRegistry] | None = None,
    ):
        super().__init__(
            global_config,
            storage,
            context_builder,
            providers,
            tool_registry,
            tool_registry_factory=tool_registry_factory,
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
        tool_registry: ToolRegistry | None,
        tool_registry_factory: Callable[[AgentRuntimeConfig], ToolRegistry] | None = None,
    ):
        super().__init__(
            global_config,
            storage,
            context_builder,
            providers,
            tool_registry,
            tool_registry_factory=tool_registry_factory,
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
