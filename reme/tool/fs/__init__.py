"""File system tools."""

from .bash_tool import BashTool
from .edit_tool import EditTool
from .find_tool import FindTool
from .grep_tool import GrepTool
from .ls_tool import LsTool
from .read_tool import ReadTool
from .write_tool import WriteTool
from ...core import R

__all__ = [
    "BashTool",
    "EditTool",
    "FindTool",
    "GrepTool",
    "LsTool",
    "ReadTool",
    "WriteTool",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register(tool_class)
