"""Utility functions for processing and formatting LLM-related message data."""

import json
import re

from ..enumeration import Role
from ..schema import Message


def format_messages(messages: list[Message | dict], enable_system: bool = False) -> str:
    """Formats a list of messages into a single string, optionally filtering system roles."""
    formatted_lines = []
    for message in messages:
        if isinstance(message, dict):
            message = Message(**message)
        if not enable_system and message.role is Role.SYSTEM:
            continue

        formatted_lines.append(message.format_message())
    return "\n".join(formatted_lines)


def extract_content(text: str, language_tag: str = "json", greedy: bool = False):
    """Extracts content from Markdown code blocks and parses it if the tag is JSON."""
    quantifier = ".*" if greedy else ".*?"
    pattern = rf"```\s*{re.escape(language_tag)}\s*({quantifier})\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        result = match.group(1).strip()
    else:
        result = text

    if language_tag == "json":
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = None

    return result