"""test"""

from .test_op import TestOp
from .translate_ts import TranslateTs
from ...core import R

__all__ = [
    "TestOp",
    "TranslateTs",
]

for name in __all__:
    agent_class = globals()[name]
    R.op.register(agent_class)
