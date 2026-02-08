"""ReMe"""

from . import agent
from . import config
from . import core
from . import tool
from . import workflow
from .reme import ReMe
from .reme_fs import ReMeFs

__all__ = [
    "agent",
    "config",
    "core",
    "tool",
    "workflow",
    "ReMe",
    "ReMeFs",
]

__version__ = "0.3.0.0a1"


"""
conda create -n fl_test2 python=3.10
conda activate fl_test2
conda env remove -n fl_test2
"""