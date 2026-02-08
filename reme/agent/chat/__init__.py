"""chat agent"""

from .fs_cli import FsCli
from .simple_chat import SimpleChat
from .stream_chat import StreamChat
from ...core import R

__all__ = [
    "FsCli",
    "StreamChat",
    "SimpleChat",
]

R.ops.register(FsCli)
R.ops.register(SimpleChat)
R.ops.register(StreamChat)
