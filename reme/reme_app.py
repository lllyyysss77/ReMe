"""ReMe application classes for simplified configuration and execution."""

import asyncio
import sys

from reme.core.utils import execute_stream_task
from .config import ReMeConfigParser
from .core.context import ServiceContext
from .core.flow import BaseFlow
from .core.schema import Response


class ReMeApp:
    """ReMe application with config file support and flow execution methods."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        **kwargs,
    ):
        self.service_context = ServiceContext(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ReMeConfigParser,
            config_path=None,
            enable_logo=enable_logo,
            **kwargs,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    def __enter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Async context manager exit."""
        await self.service_context.close()
        return False

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Context manager exit."""
        self.service_context.close_sync()
        return False

    async def execute_flow(self, name: str, **kwargs) -> Response:
        """Execute a flow with the given name and parameters."""
        assert name in self.service_context.flows, f"Flow {name} not found"
        flow: BaseFlow = self.service_context.flows[name]
        return await flow.call(**kwargs)

    async def execute_stream_flow(self, name: str, **kwargs):
        """Execute a stream flow with the given name and parameters."""
        assert name in self.service_context.flows, f"Flow {name} not found"
        flow: BaseFlow = self.service_context.flows[name]
        assert flow.stream is True, "non-stream flow is not supported in execute_stream_flow!"
        stream_queue = asyncio.Queue()
        task = asyncio.create_task(flow.call(stream_queue=stream_queue, **kwargs))
        async for chunk in execute_stream_task(
            stream_queue=stream_queue,
            task=task,
            task_name=name,
            as_bytes=False,
        ):
            yield chunk

    def run_service(self):
        """Run the configured service (HTTP, MCP, or CMD)."""
        self.service_context.service.run()


def main():
    """Main entry point for running ReMe application from command line."""
    with ReMeApp(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
