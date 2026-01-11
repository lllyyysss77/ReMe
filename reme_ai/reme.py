"""ReMe classes for simplified configuration and execution."""

from .core.application import Application
from .core.config import ReMeConfigParser
from .core.context import C
from .core.embedding import BaseEmbeddingModel
from .core.enumeration import Role
from .core.llm import BaseLLM
from .core.schema import Message
from .core.utils import singleton
from .core.vector_store import BaseVectorStore
from .mem_agent.retriever import ReMeRetriever
from .mem_agent.summarizer import ReMeSummarizer, PersonalSummarizer
from .mem_tool import (
    HandsOffTool,
    ReadHistoryMemory,
    AddMetaMemory,
    AddMemory,
    AddSummaryMemory,
    DeleteMemory,
    UpdateMemory,
    VectorRetrieveMemory,
)


@singleton
class ReMe(Application):
    """Simplified ReMe application that auto-initializes the service context."""

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
        super().__init__(
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

        C.initialize_service_context()

        self.llm: BaseLLM = C.get_llm("default")
        self.vector_store: BaseVectorStore = C.get_vector_store("default")
        self.embedding_model: BaseEmbeddingModel = C.get_embedding_model("default")

    @staticmethod
    def get_llm(name: str) -> BaseLLM:
        return C.get_llm(name)

    @staticmethod
    def _prepare_messages(messages: list[dict | Message], user_id: str, assistant_id: str):
        if not messages:
            return []

        messages = [Message(**m) if isinstance(m, dict) else m for m in messages]
        for message in messages:
            if message.role is Role.USER and user_id:
                message.name = user_id
            elif message.role is Role.ASSISTANT and assistant_id:
                message.name = assistant_id
        return messages

    async def summary(
        self,
        messages: list[dict],
        description: str = "",
        user_id: str = "",
        assistant_id: str = "",
        **kwargs,
    ):
        """Summarizes messages and stores them as memory based on the specified memory mode."""

        if user_id:
            # halumem: user_id -> message.name
            # locomo: add description
            metadata_summary = {
                "year": "The `year` information associated with the memory(Optional)",
                "month": "The `month` information associated with the memory(Optional)",
                "day": "The `day` information associated with the memory(Optional)",
                # "hour": "The `hour` information associated with the memory(Optional)",
                # "year": "The year when the memory content occurred(Optional)",
                # "month": "The month when the memory content occurred(Optional)",
                # "day": "The day when the memory content occurred(Optional)",
                # "hour": "The hour when the memory content occurred(Optional)",
            }
            meta_memories = [
                {
                    "memory_type": "personal",
                    "memory_target": user_id,
                },
            ]

            messages = self._prepare_messages(messages, user_id, assistant_id)

            personal_summarizer = PersonalSummarizer(
                tools=[
                    VectorRetrieveMemory(
                        add_memory_type_target=False,
                        metadata_desc=None,
                        top_k=15,
                    ),
                    AddMemory(add_when_to_use=False, metadata_desc=metadata_summary),
                    DeleteMemory(),
                    UpdateMemory(add_when_to_use=False, metadata_desc=metadata_summary),
                ],
            )

            reme_summarizer = ReMeSummarizer(
                meta_memories=meta_memories,
                enable_identity_memory=False,
                tools=[
                    # AddMetaMemory(),
                    AddSummaryMemory(metadata_desc=metadata_summary),
                    HandsOffTool(memory_agents=[personal_summarizer]),
                ],
            )

            try:
                await reme_summarizer.call(messages=messages, description=description, **kwargs)
                return reme_summarizer.memory_nodes
            except Exception as e:
                print(f"Warning: reme_summarizer.call failed: {e}")
                return []
            

        else:
            raise NotImplementedError

    async def retrieve(
        self,
        query: str = "",
        messages: list[dict] | None = None,
        description: str = "",
        user_id: str = "",
        assistant_id: str = "",
        top_k: int = 20,
        **kwargs,
    ):
        """Retrieves relevant memories based on the query and specified memory mode."""

        if user_id:
            messages = self._prepare_messages(messages, user_id, assistant_id)

            metadata_retrieve = {
                "year": "The year to filter memories(Optional)",
                "month": "The month to filter memories(Optional)",
                "day": "The day to filter memories(Optional)",
                # "hour": "The hour to filter memories(Optional)",
            }
            meta_memories = [
                {
                    "memory_type": "personal",
                    "memory_target": user_id,
                },
            ]

            reme_retriever = ReMeRetriever(
                meta_memories=meta_memories,
                tools=[
                    VectorRetrieveMemory(
                        add_memory_type_target=True,
                        metadata_desc=metadata_retrieve,
                        top_k=top_k,
                    ),
                    ReadHistoryMemory(),
                ],
            )

            try:
                await reme_retriever.call(query=query, messages=messages, description=description, **kwargs)
                return reme_retriever.output
            except Exception as e:
                print(f"Warning: reme_retriever.call failed: {e}")
                return "error, not retrieved"
            

        else:
            raise NotImplementedError
