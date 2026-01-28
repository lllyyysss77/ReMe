"""ReMe classes for simplified configuration and execution."""

import asyncio
import sys
from pathlib import Path

from loguru import logger

from .core import Application
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
from .tool.memory import UpdateUserProfile, RetrieveMemory, AddMemory, DelegateTask, ReadHistory, ReadUserProfile, \
    ProfileHandler


class ReMe(Application):
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
            personal_memory_target: list[str] | None = None,
            procedural_memory_target: list[str] | None = None,
            tool_memory_target: list[str] | None = None,
            profile_path: str = "reme_profile",
            main_summary_version: str = "default",
            personal_summary_version: str = "default",
            procedural_summary_version: str = "default",
            tool_summary_version: str = "default",
            main_retrieve_version: str = "default",
            personal_retrieve_version: str = "default",
            procedural_retrieve_version: str = "default",
            tool_retrieve_version: str = "default",
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            enable_logo=enable_logo,
            parser=ReMeConfigParser,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )
        memory_target_type_mapping: dict[str, MemoryType] = {}
        if personal_memory_target:
            for name in personal_memory_target:
                assert name not in memory_target_type_mapping, f"Memory target name {name} is already used."
                memory_target_type_mapping[name] = MemoryType.PERSONAL

        if procedural_memory_target:
            for name in procedural_memory_target:
                assert name not in memory_target_type_mapping, f"Memory target name {name} is already used."
                memory_target_type_mapping[name] = MemoryType.PROCEDURAL

        if tool_memory_target:
            for name in tool_memory_target:
                assert name not in memory_target_type_mapping, f"Memory target name {name} is already used."
                memory_target_type_mapping[name] = MemoryType.TOOL
        self.service_context.memory_target_type_mapping = memory_target_type_mapping
        self.profile_path: str = profile_path

    @property
    def memory_target_type_mapping(self) -> dict[str, MemoryType]:
        mapping = {}
        if self.service_context.personal_memory_target:
            for name in self.service_context.personal_memory_target:
                assert name not in mapping, f"Memory target name {name} is already used."
                mapping[name] = MemoryType.PERSONAL

        if self.service_context.procedural_memory_target:
            for name in self.service_context.procedural_memory_target:
                assert name not in mapping, f"Memory target name {name} is already used."
                mapping[name] = MemoryType.PROCEDURAL

        if self.service_context.tool_memory_target:
            for name in self.service_context.tool_memory_target:
                assert name not in mapping, f"Memory target name {name} is already used."
                mapping[name] = MemoryType.TOOL
        return mapping

    def add_meta_memory(self, memory_type: str | MemoryType, memory_target: str):
        memory_type = MemoryType(memory_type)
        if memory_type is MemoryType.PERSONAL:
            personal_memory_target = self.service_context.personal_memory_target
            if memory_target not in personal_memory_target:
                personal_memory_target.append(memory_target)
            else:
                logger.warning(f"Memory target {memory_target} is already added.")

        elif memory_type is MemoryType.PROCEDURAL:
            procedural_memory_target = self.service_context.procedural_memory_target
            if memory_target not in procedural_memory_target:
                procedural_memory_target.append(memory_target)
            else:
                logger.warning(f"Memory target {memory_target} is already added.")

        elif memory_type is MemoryType.TOOL:
            tool_memory_target = self.service_context.tool_memory_target
            if memory_target not in tool_memory_target:
                tool_memory_target.append(memory_target)
            else:
                logger.warning(f"Memory target {memory_target} is already added.")



    async def summary_memory(
        self,
        messages: list[Message | dict],
        description: str = "",
            user_name: str = "",
            task_name: str = "",
            tool_name: str = "",
        enable_thinking_params: bool = False,
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
                        DelegateTask(
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
                        DelegateTask(
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
        await self.vector_store.delete(list(set([memory_id, vector_node.vector_id])))
        await self.vector_store.insert([vector_node])

        return memory_node

    async def delete_memory(self, memory_id: str | list[str]):
        """Delete one or more memories from the vector store by their IDs."""
        vector_ids = [memory_id] if isinstance(memory_id, str) else memory_id
        await self.vector_store.delete(list(set(vector_ids)))

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

    def get_profile_handler(self, user_name: str) -> ProfileHandler:
        """Get the profile handler for the specified user."""
        profile_path = Path(self.profile_path) / self.vector_store.collection_name
        return ProfileHandler(memory_target=user_name, profile_path=profile_path)

    async def context_offload(self):
        """working memory summary"""

    async def context_reload(self):
        """working memory retrieve"""


def main():
    """Main entry point for running ReMe from command line."""
    with ReMe(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
