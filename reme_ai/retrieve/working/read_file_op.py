"""Read file operation module.

This module provides a tool operation for reading file contents.
It supports reading entire files or specific line ranges for large files.
"""

from pathlib import Path
from typing import Optional

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall


@C.register_op()
class ReadFileOp(BaseAsyncToolOp):
    """Read file operation.

    This operation reads and returns the content of a specified file.
    For text files, it can read specific line ranges using offset and limit.
    """

    file_path = __file__

    def __init__(self, **kwargs):
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "ReadFile",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "file_path": {
                    "type": "string",
                    "description": self.get_prompt("file_path"),
                    "required": True,
                },
                "offset": {
                    "type": "number",
                    "description": self.get_prompt("offset"),
                    "required": True,
                },
                "limit": {
                    "type": "number",
                    "description": self.get_prompt("limit"),
                    "required": True,
                },
            },
        }

        return ToolCall(**tool_params)

    async def async_execute(self):
        """Execute the read file operation."""
        file_path: str = self.input_dict.get("file_path", "").strip()
        offset: Optional[int] = self.input_dict.get("offset")
        limit: Optional[int] = self.input_dict.get("limit")

        # Validate file_path
        if not file_path:
            raise ValueError("The 'file_path' parameter cannot be empty.")

        # Resolve file path
        file_path_obj = Path(file_path).expanduser().resolve()

        # Check if file exists
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path_obj}")

        # Read file content
        content = file_path_obj.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Handle line range if specified
        if offset is not None or limit is not None:
            if offset is None:
                offset = 0
            if limit is None:
                limit = len(lines)

            # Validate offset and limit
            if offset < 0:
                raise ValueError("Offset must be a non-negative number")
            if limit <= 0:
                raise ValueError("Limit must be a positive number")

            total_lines = len(lines)
            start = offset
            end = min(offset + limit, total_lines)

            if start >= total_lines:
                raise ValueError(
                    f"Offset {offset} is beyond file length ({total_lines} lines)",
                )

            selected_lines = lines[start:end]
            result_content = "\n".join(selected_lines)

            # Format output with range information
            if end < total_lines:
                result = f"Showing lines {start}-{end - 1} of {total_lines} total lines.\n\n---\n\n{result_content}"
            else:
                result = result_content
        else:
            result = content

        self.set_output(result)

    async def async_default_execute(self, e: Exception = None, **_kwargs):
        """Fill outputs with a default failure message when execution fails."""
        file_path: str = self.input_dict.get("file_path", "").strip()
        error_msg = f'Failed to read file "{file_path}"'
        if e:
            error_msg += f": {str(e)}"
        self.set_output(error_msg)
