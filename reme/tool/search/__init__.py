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

R.op.register()(DashscopeSearch)
R.op.register()(MockSearch)
R.op.register()(TavilySearch)
