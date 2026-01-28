"""Default memory agents for personal, procedural, tool and ReMe memory operations."""

from .personal_retriever import PersonalRetriever
from .personal_summarizer import PersonalSummarizer
from .procedural_retriever import ProceduralRetriever
from .procedural_summarizer import ProceduralSummarizer
from .reme_retriever import ReMeRetriever
from .reme_summarizer import ReMeSummarizer
from .tool_retriever import ToolRetriever
from .tool_summarizer import ToolSummarizer
from ....core import R

__all__ = [
    "PersonalRetriever",
    "PersonalSummarizer",
    "ProceduralRetriever",
    "ProceduralSummarizer",
    "ReMeRetriever",
    "ReMeSummarizer",
    "ToolRetriever",
    "ToolSummarizer",
]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
