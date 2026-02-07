"""ReMe File System"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator

from prompt_toolkit import PromptSession

from reme.core.utils import execute_stream_task
from .agent.chat import FsCli
from .agent.fs import FsCompactor, FsSummarizer
from .config import ReMeConfigParser
from .core import Application
from .core.enumeration import MemorySource, ChunkEnum
from .core.op import BaseTool
from .core.schema import Message, StreamChunk
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
        log_to_console: bool = True,
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
            log_to_console=log_to_console,
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

        self.commands = [
            "/new",
            "/exit",
        ]

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

    async def chat_with_remy(self, tool_result_max_size: int = 100):
        """Interactive CLI chat with Remy using simple streaming output."""
        fs_cli = FsCli(working_dir=self.working_dir, tools=self.fs_tools)
        session = PromptSession()

        # Print welcome banner
        print("\n========================================")
        print("  Welcome to Remy Chat!")
        print("  Type /exit to quit, /new to start fresh.")
        print("========================================\n")

        async def chat(q: str) -> AsyncGenerator[StreamChunk, None]:
            """Execute chat query and yield streaming chunks."""
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(
                fs_cli.call(query=q, stream_queue=stream_queue, service_context=self.service_context),
            )
            async for _chunk in execute_stream_task(
                stream_queue=stream_queue,
                task=task,
                task_name="cli",
                output_format="chunk",
            ):
                yield _chunk

        while True:
            try:
                # Get user input (async)
                user_input = await session.prompt_async("You: ", default="")
                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.strip() == "/exit":
                    break

                if user_input.strip() == "/new":
                    fs_cli.reset_history()
                    print("Conversation reset.\n")
                    continue

                # Stream processing state
                in_thinking = False
                in_answer = False

                try:
                    async for chunk in chat(user_input):
                        if chunk.chunk_type == ChunkEnum.THINK:
                            if not in_thinking:
                                print("\033[90mThinking: ", end="", flush=True)
                                in_thinking = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.ANSWER:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            if not in_answer:
                                print("\nRemy: ", end="", flush=True)
                                in_answer = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.TOOL:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            print(f"\033[36m  -> Tool: {chunk.chunk}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.TOOL_RESULT:
                            tool_name = chunk.metadata.get("tool_name", "unknown")
                            result = chunk.chunk
                            if len(result) > tool_result_max_size:
                                result = result[:tool_result_max_size] + f"... ({len(chunk.chunk)} chars total)"
                            print(f"\033[36m     Tool result for {tool_name}: {result.strip()}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.ERROR:
                            print(f"\n  Error: {chunk.chunk}")

                        elif chunk.chunk_type == ChunkEnum.DONE:
                            break

                except Exception as e:
                    print(f"\nStream error: {e}")

                # End current streaming line
                print("\n")
                print("----------------------------------------\n")

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()

        print("\nGoodbye!\n")


async def async_main():
    """Main function for testing the ReMeFs CLI."""
    reme = ReMeFs(*sys.argv[1:], log_to_console=False)
    await reme.start()
    await reme.chat_with_remy()


def main():
    """Main function for testing the ReMeFs CLI."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
