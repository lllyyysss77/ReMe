"""search tool"""

from .dashscope_search import DashscopeSearch
from .mock_search import MockSearch
from .tavily_search import TavilySearch
from ...core import R

__all__ = [
    "DashscopeSearch",
    "MockSearch",
    "TavilySearch",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register(tool_class)
