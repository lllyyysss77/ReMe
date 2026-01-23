"""memory tools"""

from .base_memory_tool import BaseMemoryTool
from .history.add_history import AddHistory
from .history.read_history import ReadHistory
from .identity.add_identity import AddIdentity
from .identity.read_identity import ReadIdentity
from .meta.add_meta_memory import AddMetaMemory
from .meta.read_meta_memory import ReadMetaMemory
from .user_profile.read_user_profile import ReadUserProfile
from .user_profile.update_user_profile import UpdateUserProfile
from ...core import R

__all__ = [
    "BaseMemoryTool",
    "AddHistory",
    "ReadHistory",
    "AddIdentity",
    "ReadIdentity",
    "AddMetaMemory",
    "ReadMetaMemory",
    "ReadUserProfile",
    "UpdateUserProfile",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
