"""Base file watcher implementation.

This module provides the base class for file watcher implementations
that monitor file system changes and trigger callbacks.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, Callable

from loguru import logger
from watchfiles import awatch, Change

from ..memory_store import BaseMemoryStore


class BaseFileWatcher:
    """
    Minimal file watcher base class

    This base class provides basic file monitoring functionality that can be extended
    to implement specific file monitoring requirements.
    """

    def __init__(
        self,
        watch_paths: list[str] | str,
        suffix_filters: list[str] | None = None,
        recursive: bool = False,
        debounce: int = 500,  # Millisecond debounce
        chunk_tokens: int = 400,
        chunk_overlap: int = 80,
        memory_store: BaseMemoryStore | None = None,
        callback: Callable[[set[tuple[Change, str]]], None | Coroutine[Any, Any, None]] | None = None,
        **kwargs,
    ):
        """
        Initialize the file watcher"""
        self.watch_paths: list[str] = [watch_paths] if isinstance(watch_paths, str) else watch_paths
        self.suffix_filters: list[str] = suffix_filters or []
        self.recursive: bool = recursive
        self.debounce: int = debounce
        self.chunk_tokens: int = chunk_tokens
        self.chunk_overlap: int = chunk_overlap
        self.memory_store: BaseMemoryStore = memory_store
        self.callback = callback
        self.kwargs: dict = kwargs

        self._stop_event = asyncio.Event()
        self._watch_task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the file watcher"""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Started watching: {self.watch_paths}")

    async def close(self):
        """Stop the file watcher"""
        if not self._running:
            return

        self._stop_event.set()
        if self._watch_task:
            await self._watch_task
        self._running = False
        logger.info("Stopped watching")

    def watch_filter(self, _change: Change, path: str) -> bool:
        """Filter function for file watching."""
        # If no suffix filters are specified, watch all files
        if not self.suffix_filters:
            return True

        # Check if the file has one of the allowed suffixes
        for suffix in self.suffix_filters:
            if path.endswith("." + suffix.strip(".")):
                return True

        return False

    async def _watch_loop(self):
        """Core monitoring loop"""
        if not self.watch_paths:
            logger.warning("No watch paths specified")
            return

        async for changes in awatch(
            *self.watch_paths,
            watch_filter=self.watch_filter,
            recursive=self.recursive,
            debounce=self.debounce,
            stop_event=self._stop_event,
        ):
            if self._stop_event.is_set():
                break

            await self.on_changes(changes)

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Callback method to handle file changes"""

    async def on_changes(self, changes: set[tuple[Change, str]]):
        """Hook method to handle file changes"""
        if self.callback:
            result = self.callback(changes)
            if asyncio.iscoroutine(result):
                await result
        else:
            await self._on_changes(changes)

    def is_running(self) -> bool:
        """Check if the watcher is running"""
        return self._running

    async def add_path(self, path: str):
        """Dynamically add a path to monitor"""
        if path not in self.watch_paths:
            self.watch_paths.append(path)
            if self._running:
                await self.close()
                await self.start()

    async def remove_path(self, path: str):
        """Remove a monitored path"""
        if path in self.watch_paths:
            self.watch_paths.remove(path)
            if self._running:
                await self.close()
                await self.start()
