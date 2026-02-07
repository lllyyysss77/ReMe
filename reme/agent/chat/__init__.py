"""chat agent"""

from .simple_chat import SimpleChat
from .stream_chat import StreamChat
from ...core import R

__all__ = [
    "StreamChat",
    "SimpleChat",
]

R.ops.register(SimpleChat)
R.ops.register(StreamChat)
