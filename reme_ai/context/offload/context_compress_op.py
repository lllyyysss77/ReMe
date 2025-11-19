"""
Context compression module for reducing token usage in conversation contexts using LLM.

This module provides functionality to compress conversation history by using a language
model to generate concise summaries of older messages while preserving recent messages.
This helps manage context window limits while maintaining conversation coherence.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from uuid import uuid4

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from flowllm.core.utils import extract_content
from loguru import logger


@C.register_op()
class ContextCompressOp(BaseAsyncOp):
    """
    Context compression operation that uses LLM to reduce token usage.

    When the total token count exceeds the threshold, this operation uses a language
    model to compress older messages into a concise summary while keeping recent
    messages intact. This preserves conversation context while reducing token usage.
    """

    file_path: str = __file__

    def __init__(
        self,
        all_token_threshold: int = 20000,
        keep_recent: int = 5,
        storage_path: str = "./compressed_contexts",
        micro_summary_token_threshold: int = None,
        **kwargs,
    ):
        """
        Initialize the context compression operation.

        Args:
            all_token_threshold: Maximum total token count before compression is triggered.
            keep_recent: Number of recent messages to keep uncompressed.
            storage_path: Directory path where original messages will be stored for traceability.
            micro_summary_token_threshold: Token threshold for each compression group.
                If set, messages will be split into groups of this size and compressed separately.
                If None, all messages will be compressed together.
            **kwargs: Additional arguments passed to the base class.

        Note:
            System messages are NEVER compressed to preserve important system instructions.
        """
        super().__init__(**kwargs)
        self.all_token_threshold: int = all_token_threshold
        self.keep_recent: int = keep_recent
        self.storage_path: Path = Path(storage_path)
        self.micro_summary_token_threshold: int = micro_summary_token_threshold

        assert (
            micro_summary_token_threshold is None or micro_summary_token_threshold > 0
        ), "Micro summary token threshold must be greater than 0"

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _save_original_messages(self, messages: List[Message]) -> str:
        """Save original messages to file for traceability.

        Args:
            messages: List of messages to save

        Returns:
            Path to the saved file
        """
        # Generate unique filename with timestamp
        file_name = f"context_{uuid4().hex}.txt"
        file_path = self.storage_path / file_name

        # Convert messages to serializable format
        messages_data = [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content,
                "name": getattr(msg, "name", None),
                "tool_call_id": getattr(msg, "tool_call_id", None),
            }
            for msg in messages
        ]

        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(messages)} original messages to {file_path}")
        return str(file_path)

    def _split_messages_by_token_threshold(
        self,
        messages: List[Message],
        token_threshold: int,
    ) -> List[List[Message]]:
        """Split messages into groups based on token threshold.

        Args:
            messages: List of messages to split
            token_threshold: Maximum token count for each group

        Returns:
            List of message groups, each within the token threshold
        """
        if not messages:
            return []

        groups = []
        current_group = []
        current_token_count = 0

        for msg in messages:
            msg_tokens = self.token_count([msg])

            # If single message exceeds threshold, put it in its own group
            if msg_tokens > token_threshold:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_token_count = 0
                groups.append([msg])
                continue

            # If adding this message would exceed threshold, start new group
            if current_token_count + msg_tokens > token_threshold and current_group:
                groups.append(current_group)
                current_group = [msg]
                current_token_count = msg_tokens
            else:
                current_group.append(msg)
                current_token_count += msg_tokens

        # Add the last group if it has messages
        if current_group:
            groups.append(current_group)

        logger.info(
            f"Split {len(messages)} messages into {len(groups)} groups " f"with token threshold {token_threshold}",
        )
        return groups

    @staticmethod
    def _extract_xml_fragments(text: str) -> str:
        """
        Extract XML fragments from text, removing scratchpad elements.

        Scans text to extract complete and parseable top-level XML fragments,
        excluding <scratchpad> elements. If state_snapshot XML is found, returns it;
        otherwise returns the original text.

        Args:
            text: Input text potentially containing XML fragments

        Returns:
            Extracted XML content or original text
        """
        try:
            # Remove scratchpad elements
            new_text = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.S | re.I)
            # Extract balanced XML tags
            extract_xml = [m[0] for m in re.findall(r"(<(\w+)[^>]*>(?:[^<]|<(?!/\2))*</\2>)", new_text)]

            # Validate XML parsing
            valid_xml = []
            for xml_str in extract_xml:
                try:
                    ET.fromstring(xml_str)
                    valid_xml.append(xml_str)
                except ET.ParseError:
                    continue

            # Return state_snapshot if found, otherwise original text
            if valid_xml and any("<state_snapshot>" in xml for xml in valid_xml):
                return next(xml for xml in valid_xml if "<state_snapshot>" in xml)
            elif valid_xml:
                return valid_xml[0]
            else:
                return text
        except Exception as e:
            logger.warning(f"Failed to extract XML fragments: {e}. Returning original text.")
            return text

    @staticmethod
    def _format_messages_for_compression(messages: List[Message]) -> str:
        """Format messages into a readable text for compression.

        Args:
            messages: List of messages to format

        Returns:
            Formatted string representation of messages
        """
        lines = []
        for i, msg in enumerate(messages, 1):
            role_name = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            lines.append(f"[Message {i} - {role_name}]")
            lines.append(msg.content)
            lines.append("")  # Empty line between messages

        return "\n".join(lines)

    async def _compress_messages_with_llm(self, messages_to_compress: List[Message]) -> str:
        """Use LLM to compress messages into a concise summary.

        Args:
            messages_to_compress: List of messages to compress

        Returns:
            Compressed summary text
        """
        # Format messages for the prompt
        formatted_messages = self._format_messages_for_compression(messages_to_compress)

        # Create prompt for compression
        prompt = self.prompt_format(
            prompt_name="compress_context_prompt",
            messages_content=formatted_messages,
        )

        def parse_compressed_result(message: Message) -> str:
            """Parse LLM response to extract compressed content.

            Args:
                message: LLM response message

            Returns:
                Compressed content string
            """
            content = message.content.strip()
            # Try to extract content from txt code block
            compressed = extract_content(content, "txt")

            # If no code block found, use the raw content
            if not compressed:
                compressed = content

            logger.info(
                f"Compressed {len(messages_to_compress)} messages into "
                f"{len(compressed)} characters (reduction: "
                f"{len(formatted_messages)} -> {len(compressed)})",
            )
            return compressed

        # Call LLM to generate compressed summary
        result = await self.llm.achat(
            messages=[Message(role=Role.USER, content=prompt)],
            callback_fn=parse_compressed_result,
        )

        return result

    async def _compress_with_micro_groups(
        self,
        messages_to_compress: List[Message],
        system_messages: List[Message],
        recent_messages: List[Message],
    ) -> List[Message]:
        """Compress messages by splitting into groups and compressing each separately.

        Args:
            messages_to_compress: Messages to be compressed
            system_messages: System messages to preserve
            recent_messages: Recent messages to keep uncompressed

        Returns:
            List of new messages after compression
        """
        # Split messages into groups based on micro threshold
        message_groups = self._split_messages_by_token_threshold(
            messages_to_compress,
            self.micro_summary_token_threshold,
        )

        # Compress each group separately
        compressed_messages = []
        total_original_tokens = 0
        total_compressed_tokens = 0

        for group_idx, group in enumerate(message_groups, 1):
            # Calculate original token count for this group
            group_original_tokens = self.token_count(group)
            total_original_tokens += group_original_tokens

            # Save original messages for this group
            group_file_path = self._save_original_messages(group)

            # Compress this group
            logger.info(
                f"Compressing group {group_idx}/{len(message_groups)} "
                f"({len(group)} messages, {group_original_tokens} tokens)",
            )
            group_summary = await self._compress_messages_with_llm(group)
            group_summary = self._extract_xml_fragments(group_summary)

            # Create compressed message for this group
            compressed_message = Message(
                role=Role.SYSTEM,
                content=(
                    f"[Compressed conversation history - Part {group_idx}/{len(message_groups)}]\n"
                    f"{group_summary}\n\n"
                    f"(Original {len(group)} messages are stored in: {group_file_path})"
                ),
            )

            # Check if compression actually reduced tokens for this group
            compressed_tokens = self.token_count([compressed_message])

            if compressed_tokens >= group_original_tokens:
                logger.warning(
                    f"Group {group_idx} compression did not reduce tokens: "
                    f"{group_original_tokens} -> {compressed_tokens}. Using original messages.",
                )
                return None

            logger.info(
                f"Group {group_idx} compression successful: "
                f"{group_original_tokens} -> {compressed_tokens} tokens "
                f"(reduction: {group_original_tokens - compressed_tokens} tokens, "
                f"{100 * (1 - compressed_tokens / group_original_tokens):.1f}%)",
            )
            compressed_messages.append(compressed_message)
            total_compressed_tokens += compressed_tokens

        # Construct new message list: system messages + all compressed messages + recent messages
        new_messages = system_messages + compressed_messages + recent_messages

        logger.info(
            f"Context compression completed using micro-compression: "
            f"{len(messages_to_compress) + len(system_messages) + len(recent_messages)} messages -> "
            f"{len(new_messages)} messages ({len(message_groups)} compressed groups), "
            f"total tokens: {total_original_tokens} -> {total_compressed_tokens}",
        )

        return new_messages

    async def _compress_all_together(
        self,
        messages_to_compress: List[Message],
        system_messages: List[Message],
        recent_messages: List[Message],
    ) -> List[Message]:
        """Compress all messages together into a single summary.

        Args:
            messages_to_compress: Messages to be compressed
            system_messages: System messages to preserve
            recent_messages: Recent messages to keep uncompressed

        Returns:
            List of new messages after compression
        """
        # Calculate original token count
        original_tokens = self.token_count(messages_to_compress)

        # Save original messages to file for traceability
        original_file_path = self._save_original_messages(messages_to_compress)

        # Use LLM to compress messages
        logger.info(
            f"Starting LLM compression of {len(messages_to_compress)} messages "
            f"({original_tokens} tokens), keeping {len(recent_messages)} recent messages",
        )
        compressed_summary = await self._compress_messages_with_llm(messages_to_compress)
        compressed_summary = self._extract_xml_fragments(compressed_summary)

        # Create a new system message with the compressed content and file reference
        compressed_message = Message(
            role=Role.SYSTEM,
            content=(
                f"[Compressed conversation history]\n"
                f"{compressed_summary}\n\n"
                f"(Original {len(messages_to_compress)} messages are stored in: {original_file_path})"
            ),
        )

        # Check if compression actually reduced tokens
        compressed_tokens = self.token_count([compressed_message])

        if compressed_tokens >= original_tokens:
            logger.warning(
                f"Compression did not reduce tokens: {original_tokens} -> {compressed_tokens}. "
                f"Returning original messages.",
            )
            return None

        logger.info(
            f"Compression successful: {original_tokens} -> {compressed_tokens} tokens "
            f"(reduction: {original_tokens - compressed_tokens} tokens, "
            f"{100 * (1 - compressed_tokens / original_tokens):.1f}%)",
        )

        # Construct new message list: system messages + compressed message + recent messages
        new_messages = system_messages + [compressed_message] + recent_messages

        logger.info(
            f"Context compression completed: "
            f"{len(messages_to_compress) + len(system_messages) + len(recent_messages)} messages -> "
            f"{len(new_messages)} messages",
        )

        return new_messages

    async def async_execute(self):
        """
        Execute the context compression operation.

        The operation:
        1. Splits messages into system messages, messages to compress, and recent messages
        2. Calculates token count of messages to compress
        3. If below threshold, returns messages unchanged
        4. Otherwise, uses LLM to compress older messages by:
           - Saving original messages to file
           - Generating a concise summary of older messages
           - Replacing older messages with a single summary message
        """
        # Convert context messages to Message objects
        messages = [Message(**x) for x in self.context.messages]

        # Check if we have enough messages to compress
        if len(messages) <= self.keep_recent:
            self.context.response.answer = self.context.messages
            logger.info(
                f"Message count ({len(messages)}) is less than or "
                f"equal to keep_recent ({self.keep_recent}), no compression needed",
            )
            return

        # Split messages into those to compress and those to keep
        messages_to_compress = messages[: -self.keep_recent]
        recent_messages = messages[-self.keep_recent :]

        # Always filter out system messages (system messages are never compressed)
        system_messages = [m for m in messages_to_compress if m.role is Role.SYSTEM]
        messages_to_compress = [m for m in messages_to_compress if m.role is not Role.SYSTEM]
        logger.info(
            f"Excluding {len(system_messages)} system messages from compression, "
            f"{len(messages_to_compress)} messages remaining for compression check",
        )

        # If nothing to compress after filtering, return original messages
        if not messages_to_compress:
            self.context.response.answer = self.context.messages
            logger.info("No messages to compress after filtering, returning original messages")
            return

        # Calculate token count of messages to compress (only the content that will be compressed)
        compress_token_cnt: int = self.token_count(messages_to_compress)
        logger.info(
            f"Context compression check: messages_to_compress token count={compress_token_cnt}, "
            f"threshold={self.all_token_threshold}",
        )

        # If token count is within threshold, no compression needed
        if compress_token_cnt <= self.all_token_threshold:
            self.context.response.answer = self.context.messages
            logger.info(
                f"Messages to compress token count ({compress_token_cnt}) is within threshold "
                f"({self.all_token_threshold}), no compression needed",
            )
            return

        # Determine whether to use micro-compression (split into groups) or compress all together
        if self.micro_summary_token_threshold is not None and self.micro_summary_token_threshold > 0:
            new_messages = await self._compress_with_micro_groups(
                messages_to_compress,
                system_messages,
                recent_messages,
            )
        else:
            new_messages = await self._compress_all_together(
                messages_to_compress,
                system_messages,
                recent_messages,
            )

        # If compression failed (returned None), use original messages
        if new_messages is None:
            self.context.response.answer = self.context.messages
            return

        # Return the compressed messages as JSON
        self.context.response.answer = json.dumps([x.model_dump() for x in new_messages], ensure_ascii=False, indent=2)
