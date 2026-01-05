"""Common utility functions"""

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_coro_safely(coro: Coroutine[Any, Any, Any]) -> Any | asyncio.Task[Any]:
    """Run a coroutine in the current event loop or a new one if none exists."""
    try:
        # Attempt to retrieve the event loop associated with the current thread
        loop = asyncio.get_running_loop()

    except RuntimeError:
        # Start a new event loop to run the coroutine to completion
        return asyncio.run(coro)

    else:
        # Schedule the coroutine as a background task in the active loop
        return loop.create_task(coro)