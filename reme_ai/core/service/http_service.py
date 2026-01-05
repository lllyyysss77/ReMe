"""HTTP service implementation using FastAPI."""

import asyncio
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from .base_service import BaseService
from ..context import C
from ..enumeration import ChunkEnum
from ..flow import BaseFlow
from ..schema import Response, StreamChunk


@C.register_service("http")
class HttpService(BaseService):
    """Expose flows via HTTP REST and SSE endpoints."""

    def __init__(self, **kwargs):
        """Initialize FastAPI app with CORS and health checks."""
        super().__init__(**kwargs)
        self.app = FastAPI(title=C.service_config.app_name)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.get("/health")(lambda: {"status": "healthy"})

    def _integrate_flow(self, flow: BaseFlow) -> str:
        """Register a standard flow as a POST endpoint."""
        tool_call, request_model = self._prepare_route(flow)

        async def execute_endpoint(request: request_model) -> Response:
            return await flow.call(**request.model_dump(exclude_none=True))

        self.app.post(
            path=f"/{tool_call.name}",
            response_model=Response,
            description=tool_call.description,
        )(execute_endpoint)
        return tool_call.name

    def _integrate_stream_flow(self, flow: BaseFlow) -> str:
        """Register a streaming flow as an SSE endpoint."""
        tool_call, request_model = self._prepare_route(flow)

        async def execute_stream_endpoint(request: request_model) -> StreamingResponse:
            queue = asyncio.Queue()
            # Start flow as a background task
            task = asyncio.create_task(flow.call(stream_queue=queue, **request.model_dump(exclude_none=True)))

            async def generate_stream() -> AsyncGenerator[bytes, None]:
                done_bytes = b"data:[DONE]\n\n"
                try:
                    while True:
                        # Wait for next chunk or check if task failed
                        get_chunk = asyncio.create_task(queue.get())
                        done, _ = await asyncio.wait({get_chunk, task}, return_when=asyncio.FIRST_COMPLETED)

                        if get_chunk in done:
                            chunk: StreamChunk = get_chunk.result()
                            if chunk.done:
                                yield done_bytes
                                break
                            yield f"data:{chunk.model_dump_json()}\n\n".encode()
                        else:
                            # Task finished unexpectedly or raised exception
                            await task
                            yield done_bytes
                            break

                except Exception as e:
                    logger.exception(f"Stream error in {tool_call.name}: {e}")
                    err = StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e), done=True)
                    yield f"data:{err.model_dump_json()}\n\n".encode()
                    yield done_bytes

                finally:
                    if not task.done():
                        task.cancel()

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        self.app.post(f"/{tool_call.name}")(execute_stream_endpoint)
        return tool_call.name

    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Register a flow based on its streaming configuration."""
        return self._integrate_stream_flow(flow) if flow.stream else self._integrate_flow(flow)

    def run(self) -> None:
        """Start the Uvicorn server."""
        super().run()
        cfg = C.service_config.http
        uvicorn.run(
            self.app,
            host=cfg.host,
            port=cfg.port,
            timeout_keep_alive=cfg.timeout_keep_alive,
            limit_concurrency=cfg.limit_concurrency,
            **cfg.model_extra,
        )
