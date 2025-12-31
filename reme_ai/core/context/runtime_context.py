"""Module providing a runtime context for managing response states and asynchronous data streaming."""

import asyncio

from .base_context import BaseContext
from ..enumeration import ChunkEnum
from ..schema import Response
from ..schema import StreamChunk


class RuntimeContext(BaseContext):
    """A context class for handling execution state, including response metadata and stream queues."""

    def __init__(
        self,
        response: Response | None = None,
        stream_queue: asyncio.Queue | None = None,
        **kwargs,
    ):
        """Initialize the runtime context with optional response objects and message queues."""
        super().__init__(**kwargs)

        self.response: Response | None = response if response is not None else Response()
        self.stream_queue: asyncio.Queue | None = stream_queue

    async def add_stream_string_and_type(self, chunk: str, chunk_type: ChunkEnum):
        """Create and enqueue a stream chunk from a raw string and specific type."""
        if self.stream_queue is None:
            return self

        # Package raw data into a StreamChunk schema
        stream_chunk = StreamChunk(chunk_type=chunk_type, chunk=chunk)
        await self.stream_queue.put(stream_chunk)
        return self

    async def add_stream_chunk(self, stream_chunk: StreamChunk):
        """Directly enqueue an existing stream chunk into the stream queue."""
        if self.stream_queue is None:
            return self
        await self.stream_queue.put(stream_chunk)
        return self

    async def add_stream_done(self):
        """Enqueue a termination chunk to signal the end of the data stream."""
        if self.stream_queue is None:
            return self

        # Create a special chunk representing the completion state
        done_chunk = StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True)
        await self.stream_queue.put(done_chunk)
        return self

    def add_response_error(self, e: Exception):
        """Update the internal response object to reflect a failure state using exception details."""
        self.response.success = False
        self.response.answer = str(e.args)
