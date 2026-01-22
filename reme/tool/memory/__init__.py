"""memory tools"""

from .add_history import AddHistory
from .base_memory_tool import BaseMemoryTool
from .read_history import ReadHistory
from .read_user_profile import ReadUserProfile
from .update_user_profile import UpdateUserProfile
from ...core import R

__all__ = [
    "AddHistory",
    "BaseMemoryTool",
    "ReadHistory",
    "ReadUserProfile",
    "UpdateUserProfile",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
