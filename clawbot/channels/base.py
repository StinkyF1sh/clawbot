"""Channel abstraction for Clawbot.

Channel responsibilities:
1. Receive external messages -> put in InputQueue
2. ChannelManager fetches from OutputQueue -> send back to user

Channel does not handle business logic, only message input/output.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from clawbot.agent.session import ChannelMessage
from clawbot.queue.queue import InputMessage
from clawbot.util import POLL_INTERVAL

if TYPE_CHECKING:
    from clawbot.queue.queue import InputQueue


class BaseChannel(ABC):
    """Channel abstract base class."""

    def __init__(
        self,
        channel_id: str,
        input_queue: "InputQueue | None" = None,
    ):
        self.channel_id = channel_id
        self.input_queue = input_queue
        self._running = False

    def bind_queue(self, input_queue: "InputQueue") -> None:
        self.input_queue = input_queue

    async def start(self) -> None:
        self._running = True
        await self.run_loop()

    async def stop(self) -> None:
        self._running = False

    async def run_loop(self) -> None:
        while self._running:
            try:
                if self.input_queue:
                    msg = await self.poll_message()
                    if msg:
                        input_msg = InputMessage(
                            session_id=msg.resolved_session_id,
                            channel_id=self.channel_id,
                            agent_name=msg.agent_name,
                            content=msg.content,
                            channel_session_id=msg.channel_session_id,
                            metadata=msg.metadata,
                        )
                        await self.input_queue.put(input_msg)
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print(f"Channel {self.channel_id} error: {e}")

    @abstractmethod
    async def poll_message(self) -> ChannelMessage | None:
        pass

    @abstractmethod
    async def send_response(self, channel_session_id: str, response: str) -> bool:
        pass

    def create_channel_message(
        self,
        channel_session_id: str,
        agent_name: str,
        content: str,
        metadata: dict | None = None,
    ) -> ChannelMessage:
        return ChannelMessage(
            channel_id=self.channel_id,
            channel_session_id=channel_session_id,
            agent_name=agent_name,
            content=content,
            metadata=metadata,
        )


class SimpleChannel(BaseChannel):
    """Simplified Channel implementation for non-polling scenarios like CLI."""

    async def poll_message(self) -> ChannelMessage | None:
        return None

    async def send_response(self, channel_session_id: str, response: str) -> bool:
        return False

    def receive_message(
        self,
        channel_session_id: str,
        agent_name: str,
        content: str,
        metadata: dict | None = None,
    ) -> InputMessage | None:
        msg = ChannelMessage(
            channel_id=self.channel_id,
            channel_session_id=channel_session_id,
            agent_name=agent_name,
            content=content,
            metadata=metadata,
        )
        session_id = msg.resolved_session_id

        return InputMessage(
            session_id=session_id,
            channel_id=self.channel_id,
            agent_name=agent_name,
            content=content,
            channel_session_id=channel_session_id,
            metadata=metadata,
        )
