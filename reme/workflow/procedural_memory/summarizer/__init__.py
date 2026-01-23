"""Summarizer operators for procedural memory workflow.

This package exposes and registers summarization-related operators such as
`TrajectoryPreprocess` and `SuccessExtraction` to the global operator registry.
"""

from ....core import R
from .trajectory_preprocess import TrajectoryPreprocess
from .success_extraction import SuccessExtraction

__all__ = ["TrajectoryPreprocess", "SuccessExtraction"]

for name in __all__:
    tool_class = globals()[name]
    R.op.register()(tool_class)
