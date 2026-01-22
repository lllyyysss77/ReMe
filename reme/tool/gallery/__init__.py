"""execute tool"""

from .execute_code import ExecuteCode
from .execute_shell import ExecuteShell
from .think_tool import ThinkTool
from ...core import R

__all__ = [
    "ExecuteCode",
    "ExecuteShell",
    "ThinkTool",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
