"""High-level entry point for configuring and running ReMe services and flows."""

import asyncio

from .context import PromptHandler, ServiceContext
from .embedding import BaseEmbeddingModel
from .flow import BaseFlow
from .llm import BaseLLM
from .schema import Response
from .token_counter import BaseTokenCounter
from .utils import execute_stream_task, PydanticConfigParser
from .vector_store import BaseVectorStore


class Application:
    """Application wrapper that wires together service context, flows, and runtimes."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        parser: type[PydanticConfigParser] | None = None,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        **kwargs,
    ):
        self.service_context = ServiceContext(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=parser,
            config_path=None,
            enable_logo=enable_logo,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )
        self.prompt_handler = PromptHandler(language=self.service_context.language)
        self._started: bool = False

    @classmethod
    async def create(
        cls,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        parser: type[PydanticConfigParser] | None = None,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        **kwargs,
    ) -> "Application":
        """Create and start an Application instance asynchronously."""
        instance = cls(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            enable_logo=enable_logo,
            parser=parser,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )
        await instance.start()
        return instance

    async def start(self):
        """Start the application."""
        if self._started:
            return self
        else:
            await self.service_context.start()
            self._started = True
            return self

    async def close(self):
        """Close the application."""
        if self._started:
            await self.service_context.close()
            self._started = False
        else:
            raise RuntimeError("Application is not started")
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Async context manager exit."""
        return await self.close()

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

    @property
    def llm(self) -> BaseLLM:
        """Get the default LLM instance."""
        return self.service_context.llms.get("default")

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Get the default embedding model instance."""
        return self.service_context.embedding_models.get("default")

    @property
    def vector_store(self) -> BaseVectorStore:
        """Get the default vector store instance."""
        return self.service_context.vector_stores.get("default")

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Get the default token counter instance."""
        return self.service_context.token_counters.get("default")

    def run_service(self):
        """Run the configured service (HTTP, MCP, or CMD)."""
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.service_context.service.run()
