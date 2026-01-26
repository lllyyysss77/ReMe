"""ReMe application classes for simplified configuration and execution."""

import asyncio
import sys

from .agent.memory.default import ReMeSummarizer, PersonalSummarizer, PersonalRetriever, ReMeRetriever
from .config import ReMeConfigParser
from .core.context import ServiceContext
from .core.embedding import BaseEmbeddingModel
from .core.flow import BaseFlow
from .core.llm import BaseLLM
from .core.schema import Response, Message
from .core.token_counter import BaseTokenCounter
from .core.utils import execute_stream_task
from .core.vector_store import BaseVectorStore
from .tool.memory import UpdateUserProfile, RetrieveMemory, AddMemory, HandsOff, ReadHistory


class ReMe:
    """ReMe application with config file support and flow execution methods."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
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
            parser=ReMeConfigParser,
            config_path=None,
            enable_logo=enable_logo,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    def __enter__(self):
        """Context manager entry."""
        return self

    async def close(self):
        """Close the application."""
        return await self.service_context.close()

    def close_sync(self):
        """Close the application synchronously."""
        self.service_context.close_sync()

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Async context manager exit."""
        await self.close()
        return False

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Context manager exit."""
        self.close_sync()
        return False

    @property
    def default_llm(self) -> BaseLLM:
        """Return the default LLM instance from the service context."""
        return self.service_context.llms["default"]

    @property
    def default_embedding_model(self) -> BaseEmbeddingModel:
        """Return the default embedding model instance from the service context."""
        return self.service_context.embedding_models["default"]

    @property
    def default_vector_store(self) -> BaseVectorStore:
        """Return the default vector store instance from the service context."""
        return self.service_context.vector_stores["default"]

    @property
    def default_token_counter(self) -> BaseTokenCounter:
        """Return the default token counter instance from the service context."""
        return self.service_context.token_counters["default"]

    async def summary(
        self,
        messages: list[Message | dict],
        description: str = "",
        user_name: str | list[str] = "",
        enable_thinking_params: bool = False,
        meta_memories: list[dict] = None,
        version: str = "default",
        **kwargs,
    ):
        """Summarize messages and store them in memory for the specified user(s)."""
        if user_name:

            if isinstance(user_name, str):
                for message in messages:
                    if isinstance(message, dict) and not message.get("name"):
                        message["name"] = user_name
                    elif isinstance(message, Message) and not message.name:
                        message.name = user_name
                user_name = [user_name]

            if not meta_memories:
                meta_memories = [
                    {
                        "memory_type": "personal",
                        "memory_target": name,
                    }
                    for name in user_name
                ]

            if version == "default":
                reme_summarizer = ReMeSummarizer(
                    meta_memories=meta_memories,
                    tools=[
                        HandsOff(
                            memory_agents=[
                                PersonalSummarizer(
                                    tools=[
                                        RetrieveMemory(enable_thinking_params=enable_thinking_params),
                                        AddMemory(enable_thinking_params=enable_thinking_params),
                                        UpdateUserProfile(enable_thinking_params=enable_thinking_params),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )

            else:
                raise NotImplementedError

            return await reme_summarizer.call(
                messages=messages,
                description=description,
                service_context=self.service_context,
                **kwargs,
            )

        else:
            raise NotImplementedError

    async def retrieve(
        self,
        query: str = "",
        top_k: int = 20,
        description: str = "",
        messages: list[dict] | None = None,
        user_name: str | list[str] = "",
        enable_thinking_params: bool = False,
        meta_memories: list[dict] = None,
        version: str = "default",
        **kwargs,
    ):
        """Retrieve relevant memories for the specified user(s) based on query or messages."""
        if user_name:
            if isinstance(user_name, str):
                if messages:
                    for message in messages:
                        if isinstance(message, dict) and not message.get("name"):
                            message["name"] = user_name
                        elif isinstance(message, Message) and not message.name:
                            message.name = user_name
                user_name = [user_name]

            if not meta_memories:
                meta_memories = [
                    {
                        "memory_type": "personal",
                        "memory_target": name,
                    }
                    for name in user_name
                ]

            if version == "default":

                reme_retriever = ReMeRetriever(
                    meta_memories=meta_memories,
                    tools=[
                        HandsOff(
                            memory_agents=[
                                PersonalRetriever(
                                    tools=[
                                        RetrieveMemory(enable_thinking_params=enable_thinking_params, top_k=top_k),
                                        ReadHistory(enable_thinking_params=enable_thinking_params),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )

            else:
                raise NotImplementedError

            return await reme_retriever.call(
                query=query,
                messages=messages,
                description=description,
                service_context=self.service_context,
                **kwargs,
            )

        else:
            raise NotImplementedError

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
    with ReMe(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
