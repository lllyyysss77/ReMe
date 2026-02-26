"""File-based memory operations."""

from .fs_cli import FsCli
from .fs_compactor import FsCompactor
from .fs_context_checker import FsContextChecker
from .fs_summarizer import FsSummarizer
from ...core.registry_factory import R

__all__ = [
    "FsCli",
    "FsCompactor",
    "FsContextChecker",
    "FsSummarizer",
]

for name in __all__:
    op_class = globals()[name]
    R.ops.register(op_class)
