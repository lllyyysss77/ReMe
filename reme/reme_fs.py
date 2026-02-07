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
from .core.enumeration import ChunkEnum
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
        config_path: str = "fs",
        enable_logo: bool = True,
        log_to_console: bool = True,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_memory_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        working_dir: str = ".reme",
        compact_params: dict | None = None,
        summary_params: dict | None = None,
        search_params: dict | None = None,
        **kwargs,
    ):
        """Initialize ReMe with config."""
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            config_path=config_path,
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
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.compact_params: dict = compact_params or {}
        self.summary_params: dict = summary_params or {}
        self.search_params: dict = search_params or {}

        # Setup file system tools
        self.fs_tools: list[BaseTool] = [
            BashTool(cwd=self.working_dir),
            EditTool(cwd=self.working_dir),
            FindTool(cwd=self.working_dir),
            GrepTool(cwd=self.working_dir),
            LsTool(cwd=self.working_dir),
            ReadTool(cwd=self.working_dir),
            WriteTool(cwd=self.working_dir),
        ]

        # Commands
        self.commands = [
            "/new",
            "/compact",
            "/exit",
            "/help",
        ]

    async def compact(self, messages: list[Message | dict], previous_summary: str = ""):
        """Compact messages."""
        messages = [Message(**message) if isinstance(message, dict) else message for message in messages]
        compactor = FsCompactor(**(self.compact_params or {}))
        return await compactor.call(
            messages=messages,
            previous_summary=previous_summary,
            service_context=self.service_context,
        )

    async def summary(self, messages: list[Message | dict], date: str):
        """Summarize messages."""
        messages = [Message(**message) if isinstance(message, dict) else message for message in messages]
        summarizer = FsSummarizer(tools=self.fs_tools, **(self.summary_params or {}))
        return await summarizer.call(messages=messages, date=date, service_context=self.service_context)

    async def memory_search(self, query: str, max_results: int = 10, min_score: float = 0.3) -> str:
        """Semantically search memory files."""
        search_tool = FsMemorySearch(**(self.search_params or {}))
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
        fs_cli = FsCli(
            working_dir=self.working_dir,
            tools=self.fs_tools,
            summary_params=self.summary_params,
            search_params=self.search_params,
            compact_params=self.compact_params,
        )
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
                    result = await fs_cli.reset_history()
                    print(f"{result}\nConversation reset\n")
                    continue

                if user_input.strip() == "/compact":
                    result = await fs_cli.compact_history()
                    print(f"{result}\nHistory compacted.\n")
                    continue

                if user_input.strip() == "/help":
                    print("\nCommands:")
                    for command in self.commands:
                        print(f"  {command}")
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
