"""ReMe File System"""

from pathlib import Path

from .agent.fs import FsCompactor, FsSummarizer
from .config import ReMeConfigParser
from .core import Application
from .core.enumeration import MemorySource
from .core.op import BaseTool
from .core.schema import Message
from .tool.fs import (
    BashTool,
    EditTool,
    FindTool,
    FsMemoryGet,
    FsMemorySearch,
    GrepTool,
    LsTool,
    ReadTool,
    WriteTool,
)


class ReMeFs(Application):
    """ReMe File System"""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_memory_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        working_dir: str = ".reme",
        **kwargs,
    ):
        """Initialize ReMe with config."""
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            enable_logo=enable_logo,
            parser=ReMeConfigParser,
            default_llm_config=default_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_memory_store_config=default_memory_store_config,
            default_token_counter_config=default_token_counter_config,
            default_file_watcher_config=default_file_watcher_config,
            **kwargs,
        )

        self.working_dir: str = working_dir
        self.fs_tools: list[BaseTool] = [
            BashTool(cwd=self.working_dir),
            EditTool(cwd=self.working_dir),
            FindTool(cwd=self.working_dir),
            GrepTool(cwd=self.working_dir),
            LsTool(cwd=self.working_dir),
            ReadTool(cwd=self.working_dir),
            WriteTool(cwd=self.working_dir),
        ]
        self.working_path: Path = Path(self.working_dir)
        self.working_path.mkdir(parents=True, exist_ok=True)

    async def compact(
        self,
        messages: list[Message | dict],
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
    ):
        """Compact messages."""
        messages = [Message(**message) if isinstance(message, dict) else message for message in messages]
        compactor = FsCompactor(
            context_window_tokens=context_window_tokens,
            reserve_tokens=reserve_tokens,
            keep_recent_tokens=keep_recent_tokens,
        )

        return await compactor.call(messages=messages, service_context=self.service_context)

    async def summary(
        self,
        messages: list[Message | dict],
        date: str,
        version: str = "default",
        context_window_tokens: int = 128000,
        reserve_tokens: int = 32000,
        soft_threshold_tokens: int = 4000,
    ):
        """Summarize messages."""
        messages = [Message(**message) if isinstance(message, dict) else message for message in messages]
        summarizer = FsSummarizer(
            tools=self.fs_tools,
            version=version,
            context_window_tokens=context_window_tokens,
            reserve_tokens=reserve_tokens,
            soft_threshold_tokens=soft_threshold_tokens,
        )

        return await summarizer.call(messages=messages, date=date, service_context=self.service_context)

    async def memory_search(
        self,
        query: str,
        max_results: int = 20,
        min_score: float = 0.1,
        sources: list[MemorySource] | None = None,
        hybrid_enabled: bool = True,
        hybrid_vector_weight: float = 0.7,
        hybrid_text_weight: float = 0.3,
        hybrid_candidate_multiplier: float = 3.0,
    ) -> str:
        """Semantically search memory files."""
        search_tool = FsMemorySearch(
            sources=sources,
            hybrid_enabled=hybrid_enabled,
            hybrid_vector_weight=hybrid_vector_weight,
            hybrid_text_weight=hybrid_text_weight,
            hybrid_candidate_multiplier=hybrid_candidate_multiplier,
        )

        return await search_tool.call(
            query=query,
            max_results=max_results,
            min_score=min_score,
            service_context=self.service_context,
        )

    async def memory_get(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read specific snippets from memory files."""
        get_tool = FsMemoryGet(workspace_dir=self.working_dir)
        return await get_tool.call(path=path, offset=offset, limit=limit, service_context=self.service_context)
