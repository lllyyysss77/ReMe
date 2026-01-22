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

R.op.register()(ExecuteCode)
R.op.register()(ExecuteShell)
R.op.register()(ThinkTool)
