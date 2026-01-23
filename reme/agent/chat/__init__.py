"""chat agent"""

from .simple_chat import SimpleChat
from .stream_chat import StreamChat
from ...core import R

__all__ = [
    "StreamChat",
    "SimpleChat",
]

R.op.register("simple_chat")(SimpleChat)
R.op.register("stream_chat")(StreamChat)
