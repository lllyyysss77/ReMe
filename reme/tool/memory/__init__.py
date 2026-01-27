"""memory tools"""

from .add_history import AddHistory
from .add_memory import AddMemory
from .base_memory_tool import BaseMemoryTool
from .delegate_task import DelegateTask
from .delete_memory import DeleteMemory
from .profile_handler import ProfileHandler
from .read_history import ReadHistory
from .read_profile import ReadProfile
from .retrieve_memory import RetrieveMemory
from .retrieve_recent_memory import RetrieveRecentMemory
from .update_memory import UpdateMemory
from .update_profile import UpdateProfile
from ...core import R

__all__ = [
    "AddHistory",
    "AddMemory",
    "BaseMemoryTool",
    "DelegateTask",
    "DeleteMemory",
    "ProfileHandler",
    "ReadHistory",
    "ReadProfile",
    "RetrieveMemory",
    "RetrieveRecentMemory",
    "UpdateMemory",
    "UpdateProfile",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)