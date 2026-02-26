"""Summarizer operators for procedural memory workflow.

This package exposes and registers summarization-related operators such as
`TrajectoryPreprocess` and `SuccessExtraction` to the global operator registry.
"""

from .success_extraction import SuccessExtraction
from .trajectory_preprocess import TrajectoryPreprocess
from ....core import R

__all__ = ["TrajectoryPreprocess", "SuccessExtraction"]

for name in __all__:
    tool_class = globals()[name]
    R.ops.register(tool_class)
