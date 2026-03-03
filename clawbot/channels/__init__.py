"""Channels package for Clawbot."""

from clawbot.channels.base import BaseChannel, ChannelMessage, SimpleChannel
from clawbot.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelMessage", "SimpleChannel", "ChannelManager"]
