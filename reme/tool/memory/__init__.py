"""memory tools"""

from .add_draft_and_read_all_profiles import AddDraftAndReadAllProfiles
from .add_draft_and_retrieve_similar_memory import AddDraftAndRetrieveSimilarMemory
from .add_history import AddHistory
from .add_memory import AddMemory
from .base_memory_tool import BaseMemoryTool
from .delegate_task import DelegateTask
from .delete_memory import DeleteMemory
from .memory_handler import MemoryHandler
from .profile_handler import ProfileHandler
from .read_all_profiles import ReadAllProfiles
from .read_history import ReadHistory
from .retrieve_memory import RetrieveMemory
from .retrieve_recent_memory import RetrieveRecentMemory
from .update_memory import UpdateMemory
from .update_memory_v2 import UpdateMemoryV2
from .update_profile import UpdateProfile
from ...core import R

__all__ = [
    "AddDraftAndReadAllProfiles",
    "AddDraftAndRetrieveSimilarMemory",
    "AddHistory",
    "AddMemory",
    "BaseMemoryTool",
    "DelegateTask",
    "DeleteMemory",
    "MemoryHandler",
    "ProfileHandler",
    "ReadAllProfiles",
    "ReadHistory",
    "RetrieveMemory",
    "RetrieveRecentMemory",
    "UpdateMemory",
    "UpdateMemoryV2",
    "UpdateProfile",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
