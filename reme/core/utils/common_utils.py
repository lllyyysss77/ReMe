"""Common utility functions"""

import asyncio
import hashlib
from collections.abc import AsyncGenerator, Coroutine
from typing import Any, Literal

import numpy as np
from loguru import logger

from ..enumeration import ChunkEnum
from ..schema import StreamChunk


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


async def execute_stream_task(
    stream_queue: asyncio.Queue,
    task: asyncio.Task,
    task_name: str | None = None,
    output_format: Literal["str", "bytes", "chunk"] = "str",
) -> AsyncGenerator[str | bytes | StreamChunk, None]:
    """
    Core stream flow execution logic.

    Handles streaming from a queue while monitoring the task completion.
    Properly manages errors and resource cleanup.

    Args:
        stream_queue: Queue to receive StreamChunk objects from
        task: Background task executing the flow
        task_name: Optional flow name for logging purposes
        output_format: Output format control
            - "str": SSE-formatted string (default)
            - "bytes": SSE-formatted bytes for HTTP responses
            - "chunk": Raw StreamChunk objects

    Yields:
        - str: SSE-formatted data when output_format="str"
        - bytes: SSE-formatted data when output_format="bytes"
        - StreamChunk: Raw chunk objects when output_format="chunk"
    """
    is_raw_chunk = output_format == "chunk"
    is_bytes = output_format == "bytes"
    done_msg = b"data:[DONE]\n\n" if is_bytes else "data:[DONE]\n\n"

    try:
        while True:
            # Wait for next chunk or check if task failed
            get_chunk = asyncio.create_task(stream_queue.get())
            done, _ = await asyncio.wait({get_chunk, task}, return_when=asyncio.FIRST_COMPLETED)

            if get_chunk in done:
                chunk: StreamChunk = get_chunk.result()

                # Handle raw chunk mode
                if is_raw_chunk:
                    yield chunk
                    if chunk.done:
                        break
                    continue

                # Handle SSE format mode
                if chunk.done:
                    yield done_msg
                    break

                data = f"data:{chunk.model_dump_json()}\n\n"
                yield data.encode() if is_bytes else data
            else:
                # Task finished unexpectedly or raised exception
                await task
                if is_raw_chunk:
                    # Yield a DONE chunk in raw mode
                    yield StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True)
                else:
                    yield done_msg
                break

    except Exception as e:
        log_msg = f"Stream error in {task_name}: {e}" if task_name else f"Stream error: {e}"
        logger.exception(log_msg)

        err = StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e), done=True)

        if is_raw_chunk:
            yield err
        else:
            err_data = f"data:{err.model_dump_json()}\n\n"
            yield err_data.encode() if is_bytes else err_data
            yield done_msg

    finally:
        # Ensure task is cancelled if still running to avoid resource leaks
        if not task.done():
            task.cancel()


def hash_text(text: str) -> str:
    """Generate SHA-256 hash of text content.

    Args:
        text: Input text to hash

    Returns:
        Hexadecimal representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def batch_cosine_similarity(nd_array1: np.ndarray, nd_array2: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity matrix between two batches of vectors.

    Args:
        nd_array1: Matrix of shape (batch_size1, emb_size)
        nd_array2: Matrix of shape (batch_size2, emb_size)

    Returns:
        Similarity matrix of shape (batch_size1, batch_size2) where
        result[i, j] is the cosine similarity between nd_array1[i] and nd_array2[j]

    Raises:
        ValueError: If embedding dimensions don't match
    """
    if nd_array1.shape[1] != nd_array2.shape[1]:
        raise ValueError(f"Embedding dimensions must match: {nd_array1.shape[1]} != {nd_array2.shape[1]}")

    # Compute dot products: (batch_size1, emb_size) @ (emb_size, batch_size2)
    # Result shape: (batch_size1, batch_size2)
    dot_products = np.dot(nd_array1, nd_array2.T)

    # Compute L2 norms for each vector
    norms1 = np.linalg.norm(nd_array1, axis=1)  # Shape: (batch_size1,)
    norms2 = np.linalg.norm(nd_array2, axis=1)  # Shape: (batch_size2,)

    # Compute outer product of norms: (batch_size1, 1) @ (1, batch_size2)
    # Result shape: (batch_size1, batch_size2)
    norm_products = np.outer(norms1, norms2)

    # Avoid division by zero
    norm_products = np.where(norm_products == 0, 1e-10, norm_products)

    # Compute cosine similarities
    return dot_products / norm_products
