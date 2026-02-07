"""File system tools."""

from .base_fs_tool import BaseFsTool
from .bash_tool import BashTool
from .edit_tool import EditTool
from .find_tool import FindTool
from .fs_memory_get import FsMemoryGet
from .fs_memory_search import FsMemorySearch
from .grep_tool import GrepTool
from .ls_tool import LsTool
from .read_tool import ReadTool
from .write_tool import WriteTool
from ...core import R

__all__ = [
    "BaseFsTool",
    "BashTool",
    "EditTool",
    "FindTool",
    "FsMemoryGet",
    "FsMemorySearch",
    "GrepTool",
    "LsTool",
    "ReadTool",
    "WriteTool",
]

for name in __all__:
    tool_class = globals()[name]
    R.ops.register(tool_class)
