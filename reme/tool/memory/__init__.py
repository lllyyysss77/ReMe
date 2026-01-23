"""memory tools"""

from .base_memory_tool import BaseMemoryTool
from .hands_off.hands_off import HandsOff
from .history.add_history import AddHistory
from .history.read_history import ReadHistory
from .identity.add_identity import AddIdentity
from .identity.read_identity import ReadIdentity
from .meta.add_meta_memory import AddMetaMemory
from .meta.read_meta_memory import ReadMetaMemory
from .user_profile.read_user_profile import ReadUserProfile
from .user_profile.update_user_profile import UpdateUserProfile
from .vector.add_memory import AddMemory
from .vector.update_memory import UpdateMemory
from .vector.retrieve_memory import VectorRetrieveMemory
from .vector.retrieve_recent_memory import VectorRetrieveRecentMemory
from .vector.delete_memory import DeleteMemory
from ...core import R

__all__ = [
    "BaseMemoryTool",
    "HandsOff",
    "AddHistory",
    "ReadHistory",
    "AddIdentity",
    "ReadIdentity",
    "AddMetaMemory",
    "ReadMetaMemory",
    "ReadUserProfile",
    "UpdateUserProfile",
    "AddMemory",
    "UpdateMemory",
    "VectorRetrieveMemory",
    "VectorRetrieveRecentMemory",
    "DeleteMemory",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
