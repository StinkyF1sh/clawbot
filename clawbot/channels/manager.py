"""Channel manager for Clawbot."""

from typing import TYPE_CHECKING

from clawbot.queue.queue import OutputMessage
from clawbot.util import QUEUE_TIMEOUT

if TYPE_CHECKING:
    from clawbot.channels.base import BaseChannel
    from clawbot.queue.queue import TaskQueueManager


class ChannelManager:
    """Channel manager - lifecycle management and response routing."""

    def __init__(
        self,
        queue_manager: "TaskQueueManager",
    ):
        self.queue_manager = queue_manager
        self.channels: dict[str, "BaseChannel"] = {}
        self._running = False

    def register_channel(self, channel: "BaseChannel") -> None:
        self.channels[channel.channel_id] = channel

    async def start_all_channels(self) -> None:
        import asyncio

        tasks = [channel.start() for channel in self.channels.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all_channels(self) -> None:
        for channel in self.channels.values():
            await channel.stop()

    async def run_output_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = await self.queue_manager.get_output_with_timeout(QUEUE_TIMEOUT)
                if msg:
                    await self._route_response(msg)
            except Exception as e:
                print(f"ChannelManager error: {e}")

    async def _route_response(self, msg: OutputMessage) -> None:
        channel = self.channels.get(msg.channel_id)
        if channel:
            await channel.send_response(
                channel_session_id=msg.channel_session_id,
                response=msg.content,
            )

    def stop(self) -> None:
        self._running = False
