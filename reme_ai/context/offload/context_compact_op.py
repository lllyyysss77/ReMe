"""
Context compaction module for reducing token usage in conversation contexts.

This module provides functionality to compress large tool messages by storing
their full content in external files and keeping only previews in the context.
This helps manage context window limits while preserving important information.
"""

import json
from pathlib import Path
from typing import List
from uuid import uuid4

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from loguru import logger


@C.register_op()
class ContextCompactOp(BaseAsyncOp):
    """
    Context compaction operation that reduces token usage by compressing tool messages.

    When the total token count exceeds the threshold, this operation compresses large tool
    messages by truncating their content and storing the full content in external files.
    This helps manage context window limits while preserving recent tool messages.
    """

    def __init__(
        self,
        all_token_threshold: int = 20000,
        tool_token_threshold: int = 2000,
        tool_left_char_len: int = 100,
        keep_recent: int = 1,
        storage_path: str = "./",
        exclude_tools: List[str] = None,
        **kwargs,
    ):
        """
        Initialize the context compaction operation.

        Args:
            all_token_threshold: Maximum total token count before compaction is triggered.
            tool_token_threshold: Maximum token count for a single tool message before it's compressed.
            tool_left_char_len: Number of characters to keep in the compressed tool message preview.
            keep_recent: Number of recent tool messages to keep uncompressed.
            storage_path: Directory path where compressed tool message contents will be stored.
            exclude_tools: List of tool names to exclude from compaction (not currently used).
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.all_token_threshold: int = all_token_threshold
        self.tool_token_threshold: int = tool_token_threshold
        self.tool_left_char_len: int = tool_left_char_len
        self.keep_recent: int = keep_recent
        self.storage_path: Path = Path(storage_path)
        self.exclude_tools: List[str] = exclude_tools

    async def async_execute(self):
        """
        Execute the context compaction operation.

        The operation:
        1. Calculates the total token count of all messages
        2. If below threshold, returns messages unchanged
        3. Otherwise, compresses large tool messages by:
           - Keeping only a preview of the content
           - Storing full content in external files
           - Preserving recent tool messages
        """
        # Convert context messages to Message objects
        messages = [Message(**x) for x in self.context.messages]

        # Calculate total token count
        token_cnt: int = self.token_count(messages)
        logger.info(f"Context compaction check: total token count={token_cnt}, threshold={self.all_token_threshold}")

        # If token count is within threshold, no compaction needed
        if token_cnt <= self.all_token_threshold:
            self.context.response.answer = self.context.messages
            logger.info(
                f"Token count ({token_cnt}) is within threshold ({self.all_token_threshold}), no compaction needed",
            )
            return

        # Filter tool messages for processing
        tool_messages = [x for x in messages if x.role is Role.TOOL]

        # If there are too few tool messages, no compaction needed
        if len(tool_messages) <= self.keep_recent:
            self.context.response.answer = self.context.messages
            logger.info(
                f"Tool message count ({len(tool_messages)}) is less than or "
                f"equal to keep_recent ({self.keep_recent}), no compaction needed",
            )
            return

        # Exclude recent tool messages from compaction (keep them intact)
        tool_messages = tool_messages[: -self.keep_recent]
        logger.info(
            f"Processing {len(tool_messages)} tool messages for "
            f"compaction (keeping {self.keep_recent} recent messages)",
        )

        # Dictionary to store file paths and their compressed content (for potential batch writing)
        write_file_dict = {}

        # Process each tool message
        for tool_message in tool_messages:
            # Calculate token count for this specific tool message
            tool_token_cnt = self.token_count([tool_message])

            # Skip if token count is within threshold
            if tool_token_cnt <= self.tool_token_threshold:
                logger.info(
                    f"Skipping tool message (tool_call_id={tool_message.tool_call_id}): "
                    f"token count ({tool_token_cnt}) is within threshold ({self.tool_token_threshold})",
                )
                continue

            # Create compressed preview of the tool message content
            compact_result = tool_message.content[: self.tool_left_char_len] + "..."

            # Generate file name from tool_call_id or create a unique identifier
            file_name = tool_message.tool_call_id or uuid4().hex
            path = self.storage_path / f"{file_name}.txt"

            # Store the mapping for potential batch writing
            write_file_dict[str(path)] = compact_result

            # Log the compaction action
            logger.info(
                f"Compacting tool message (tool_call_id={tool_message.tool_call_id}): "
                f"token count={tool_token_cnt}, saving full content to {path}",
            )

            # Update tool message content with preview and file reference
            compact_result += f" (detailed result is stored in {path})"
            tool_message.content = compact_result

        # Return the compacted messages as JSON
        self.context.response.answer = json.dumps([x.simple_dump() for x in messages], ensure_ascii=False, indent=2)
        logger.info(f"Context compaction completed: {len(write_file_dict)} tool messages were compacted")
