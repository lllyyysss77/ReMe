"""execute tool"""

from .execute_code import ExecuteCode
from .execute_shell import ExecuteShell
from ...core import R

__all__ = [
    "ExecuteCode",
    "ExecuteShell",
]

R.op.register()(ExecuteCode)
R.op.register()(ExecuteShell)
