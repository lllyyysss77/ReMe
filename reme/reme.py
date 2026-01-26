"""ReMe classes for simplified configuration and execution."""

import asyncio
import sys

from .agent.memory.default import ReMeSummarizer, PersonalSummarizer, PersonalRetriever, ReMeRetriever
from .config import ReMeConfigParser
from .core.context import PromptHandler, ServiceContext
from .core.embedding import BaseEmbeddingModel
from .core.enumeration import MemoryType
from .core.flow import BaseFlow
from .core.llm import BaseLLM
from .core.schema import Response, Message, MemoryNode, VectorNode
from .core.token_counter import BaseTokenCounter
from .core.utils import execute_stream_task, get_now_time
from .core.vector_store import BaseVectorStore
from .tool.memory import UpdateUserProfile, RetrieveMemory, AddMemory, HandsOff, ReadHistory, ReadUserProfile


class ReMe:
    """ReMe with config file support and flow execution methods."""

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

        self.prompt_handler = PromptHandler(language=self.service_context.language)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    def __enter__(self):
        """Context manager entry."""
        return self

    async def close(self):
        """Close"""
        return await self.service_context.close()

    def close_sync(self):
        """Close synchronously"""
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
    def llm(self) -> BaseLLM:
        """Return the default LLM instance from the service context."""
        return self.service_context.llms["default"]

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Return the default embedding model instance from the service context."""
        return self.service_context.embedding_models["default"]

    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the default vector store instance from the service context."""
        return self.service_context.vector_stores["default"]

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Return the default token counter instance from the service context."""
        return self.service_context.token_counters["default"]

    async def summary_memory(
        self,
        messages: list[Message | dict],
        description: str = "",
        user_name: str | list[str] = "",
        enable_thinking_params: bool = False,
        meta_memories: list[dict] = None,
        version: str = "default",
        return_dict: bool = False,
        **kwargs,
    ) -> str | dict:
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

            result = await reme_summarizer.call(
                messages=messages,
                description=description,
                service_context=self.service_context,
                **kwargs,
            )

            if return_dict:
                return result
            else:
                return result["answer"]

        else:
            raise NotImplementedError

    async def retrieve_memory(
        self,
        query: str = "",
        top_k: int = 20,
        description: str = "",
        messages: list[dict] | None = None,
        user_name: str | list[str] = "",
        enable_thinking_params: bool = False,
        meta_memories: list[dict] = None,
        version: str = "default",
        return_dict: bool = False,
        **kwargs,
    ) -> str | dict:
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

            result = await reme_retriever.call(
                query=query,
                messages=messages,
                description=description,
                service_context=self.service_context,
                **kwargs,
            )

            if return_dict:
                return result
            else:
                return result["answer"]

        else:
            raise NotImplementedError

    async def add_memory(
        self,
        memory_content: str,
        user_name: str,
        memory_type: str | MemoryType | None = None,
        memory_target: str = "",
        when_to_use: str = "",
        ref_memory_id: str = "",
        author: str = "",
        score: float = 0,
        conversation_time: str = "",
        **kwargs,
    ) -> MemoryNode:
        """Add a new memory to the vector store for the specified user."""

        if user_name:
            memory_type = MemoryType.PERSONAL
            memory_target = user_name
        else:
            memory_type = MemoryType(memory_type)
            assert memory_target, "memory_target is required"

        metadata = kwargs.copy()
        if conversation_time:
            metadata["conversation_time"] = conversation_time

        memory_node = MemoryNode(
            memory_type=memory_type,
            memory_target=memory_target,
            when_to_use=when_to_use,
            content=memory_content,
            ref_memory_id=ref_memory_id,
            author=author,
            score=score,
            metadata=metadata,
        )
        vector_node = memory_node.to_vector_node()
        await self.vector_store.delete([vector_node.vector_id])
        await self.vector_store.insert([vector_node])

        return memory_node

    async def update_memory(
        self,
        memory_id: str,
        memory_content: str,
        user_name: str,
        memory_type: str | MemoryType | None = None,
        memory_target: str = "",
        when_to_use: str = "",
        ref_memory_id: str = "",
        author: str = "",
        score: float = 0,
        conversation_time: str = "",
        **kwargs,
    ) -> MemoryNode:
        """Update an existing memory in the vector store by its ID."""

        if user_name:
            memory_type = MemoryType.PERSONAL
            memory_target = user_name
        else:
            memory_type = MemoryType(memory_type)
            assert memory_target, "memory_target is required"

        metadata = kwargs.copy()
        if conversation_time:
            metadata["conversation_time"] = conversation_time

        memory_node = MemoryNode(
            memory_type=memory_type,
            memory_target=memory_target,
            when_to_use=when_to_use,
            content=memory_content,
            ref_memory_id=ref_memory_id,
            author=author,
            score=score,
            metadata=metadata,
        )
        vector_node = memory_node.to_vector_node()
        await self.vector_store.delete([memory_id, vector_node.vector_id])
        await self.vector_store.insert([vector_node])

        return memory_node

    async def delete_memory(self, memory_id: str | list[str]):
        """Delete one or more memories from the vector store by their IDs."""
        vector_ids = [memory_id] if isinstance(memory_id, str) else memory_id
        await self.vector_store.delete(vector_ids)

    async def delete_all_memories(self):
        """Delete all memories from the vector store."""
        await self.vector_store.delete_all()

    async def get_memory(self, memory_id: str | list[str]) -> MemoryNode | list[MemoryNode]:
        """Retrieve one or more memories from the vector store by their IDs."""
        vector_ids = [memory_id] if isinstance(memory_id, str) else memory_id
        vector_nodes = await self.vector_store.get(vector_ids)
        if isinstance(vector_nodes, VectorNode):
            return vector_nodes.to_memory_node()
        else:
            return [node.to_memory_node() for node in vector_nodes]

    async def get_all_memories(self) -> list[MemoryNode]:
        """Retrieve all memories from the vector store."""
        return [node.to_memory_node() for node in await self.vector_store.list()]

    async def get_profiles(self, user_name: str | list[str]) -> str | list[str]:
        """Retrieve user profile(s) from the system for the specified user(s)."""
        read_profile = ReadUserProfile(show_id="profile")
        if isinstance(user_name, str):
            return await read_profile.call(memory_target=user_name, service_context=self.service_context)
        else:
            return [
                await read_profile.call(memory_target=name, service_context=self.service_context) for name in user_name
            ]

    async def add_profile(
        self,
        profile_key: str,
        profile_value: str,
        user_name: str,
        update_time: str | None = None,
    ) -> MemoryNode:
        """Add user profile to ReMe system."""
        update_user_profile = UpdateUserProfile()
        if update_time is None:
            update_time = get_now_time()

        await update_user_profile.call(
            profile_ids_to_delete=[],
            profiles_to_add=[
                {
                    "update_time": update_time,
                    "profile_key": profile_key,
                    "profile_value": profile_value,
                },
            ],
            memory_target=user_name,
            service_context=self.service_context,
        )
        return update_user_profile.memory_nodes[0]

    async def update_profile(
        self,
        profile_id: str,
        profile_key: str,
        profile_value: str,
        user_name: str,
        update_time: str | None = None,
    ) -> MemoryNode:
        """Add user profile to ReMe system."""
        update_user_profile = UpdateUserProfile()
        if update_time is None:
            update_time = get_now_time()

        await update_user_profile.call(
            profile_ids_to_delete=[profile_id],
            profiles_to_add=[
                {
                    "update_time": update_time,
                    "profile_key": profile_key,
                    "profile_value": profile_value,
                },
            ],
            memory_target=user_name,
            service_context=self.service_context,
        )
        return update_user_profile.memory_nodes[0]

    async def delete_all_profiles(self, user_name: str | list[str]):
        """Delete all user profiles from ReMe system."""
        if isinstance(user_name, str):
            user_name = [user_name]

        read_profile = ReadUserProfile(show_id="profile")
        update_profile = UpdateUserProfile()
        for memory_target in user_name:
            await read_profile.call(memory_target=memory_target, service_context=self.service_context)
            profile_ids = [profile.memory_id for profile in read_profile.memory_nodes]
            await update_profile.call(
                profile_ids_to_delete=profile_ids,
                memory_target=memory_target,
                service_context=self.service_context,
            )

    async def context_offload(self):
        """working memory summary"""

    async def context_reload(self):
        """working memory retrieve"""

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
    """Main entry point for running ReMe from command line."""
    with ReMe(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
