"""Task queue management for Clawbot.

Dual-queue architecture:
- InputQueue: Channel receives external messages -> puts in queue -> GlobalAgentLoop fetches
- OutputQueue: GlobalAgentLoop processes -> puts in queue -> ChannelManager fetches
  and sends to Channel
"""

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class InputMessage:
    """Input queue message - raw message from Channel."""
    session_id: str
    content: Any
    channel_id: str
    agent_name: str = "default"
    channel_session_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OutputMessage:
    """Output queue message - response to send to Channel."""
    session_id: str
    channel_id: str
    channel_session_id: str
    content: Any
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] | None = None


class InputQueue:
    """Input queue - handles raw messages from Channel."""

    def __init__(self, max_size: int = 0):
        self.queue: asyncio.Queue[InputMessage] = asyncio.Queue(maxsize=max_size)

    async def put(self, message: InputMessage) -> None:
        await self.queue.put(message)

    def try_put(self, message: InputMessage) -> None:
        self.queue.put_nowait(message)

    async def get(self) -> InputMessage:
        return await self.queue.get()

    def try_get(self) -> InputMessage:
        return self.queue.get_nowait()

    def qsize(self) -> int:
        return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()

    async def join(self) -> None:
        await self.queue.join()

    def task_done(self) -> None:
        self.queue.task_done()


class OutputQueue:
    """Output queue - handles response messages to Channel."""

    def __init__(self, max_size: int = 0):
        self.queue: asyncio.Queue[OutputMessage] = asyncio.Queue(maxsize=max_size)

    async def put(self, message: OutputMessage) -> None:
        await self.queue.put(message)

    def try_put(self, message: OutputMessage) -> None:
        self.queue.put_nowait(message)

    async def get(self) -> OutputMessage:
        return await self.queue.get()

    def try_get(self) -> OutputMessage:
        return self.queue.get_nowait()

    def qsize(self) -> int:
        return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()


@dataclass
class QueueConfig:
    """Queue configuration."""
    input_max_size: int = 0
    output_max_size: int = 0
    default_timeout: float = 30.0


class TaskQueueManager:
    """Task queue manager - unified management of input and output queues."""

    def __init__(self, config: QueueConfig | None = None):
        self.config = config or QueueConfig()
        self.input_queue = InputQueue(max_size=self.config.input_max_size)
        self.output_queue = OutputQueue(max_size=self.config.output_max_size)

    async def send_input(
        self,
        session_id: str,
        channel_id: str,
        agent_name: str,
        content: Any,
        channel_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message = InputMessage(
            session_id=session_id,
            channel_id=channel_id,
            agent_name=agent_name,
            content=content,
            channel_session_id=channel_session_id,
            metadata=metadata,
        )
        await self.input_queue.put(message)

    async def send_output(
        self,
        session_id: str,
        channel_id: str,
        channel_session_id: str,
        content: Any,
        success: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message = OutputMessage(
            session_id=session_id,
            channel_id=channel_id,
            channel_session_id=channel_session_id,
            content=content,
            success=success,
            error=error,
            metadata=metadata,
        )
        await self.output_queue.put(message)

    async def get_output(self) -> OutputMessage:
        return await self.output_queue.get()

    async def get_output_with_timeout(self, timeout: float) -> OutputMessage | None:
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def get_input_with_timeout(self, timeout: float) -> InputMessage | None:
        try:
            return await asyncio.wait_for(self.input_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def task_done(self) -> None:
        self.input_queue.task_done()
