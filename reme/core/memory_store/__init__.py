"""Memory store module for persistent memory management.

This module provides storage backends for memory chunks and file metadata,
including SQLite-based implementations with vector and full-text search.
"""

from .base_memory_store import BaseMemoryStore
from .sqlite_memory_store import SqliteMemoryStore
from ..context import R

__all__ = [
    "BaseMemoryStore",
    "SqliteMemoryStore",
]

R.memory_stores.register("sqlite")(SqliteMemoryStore)
