"""Memory get tool for reading specific snippets from memory files."""

import os
from pathlib import Path

from reme.core.schema import ToolCall
from .base_fs_tool import BaseFsTool


class FsMemoryGet(BaseFsTool):
    """Read specific snippets from memory files."""

    def __init__(self, workspace_dir: str | None = None, **kwargs):
        """Initialize memory get tool."""
        kwargs.setdefault("name", "memory_get")
        super().__init__(**kwargs)
        self.workspace_dir = workspace_dir or os.getcwd()

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": (
                    "Safe snippet read from MEMORY.md, memory/*.md with optional offset/limit; "
                    "use after memory_search to pull only the needed lines and keep context small."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the memory file to read (relative or absolute)",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Starting line number (1-indexed, optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of lines to read from the starting line (optional)",
                        },
                    },
                    "required": ["path"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the memory get operation."""
        raw_path: str = self.context.path.strip()
        offset: int | None = self.context.get("offset", None)
        limit: int | None = self.context.get("limit", None)

        if os.path.isabs(raw_path):
            abs_path = os.path.abspath(raw_path)
        else:
            abs_path = os.path.abspath(os.path.join(self.workspace_dir, raw_path))
        assert abs_path.lower().endswith(".md")

        # Check file exists, is not a symlink, and is a regular file
        file_path = Path(abs_path)
        assert (
            file_path.exists() and not file_path.is_symlink() and file_path.is_file()
        ), f"File not found or not a regular file: {abs_path}"

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        if offset is None and limit is None:
            return content

        lines = content.split("\n")
        total_lines = len(lines)

        # Validate and normalize offset (1-indexed)
        start = offset if offset is not None else 1
        assert start >= 1, f"offset must be >= 1, got {start}"
        assert start <= total_lines, f"offset {start} exceeds total lines {total_lines}"

        # Validate and calculate count
        if limit is not None:
            assert limit > 0, f"limit must be positive, got {limit}"
            count = limit
        else:
            # Read from start to end of file
            count = total_lines - start + 1

        # Extract slice (1-indexed to 0-indexed conversion)
        selected = lines[start - 1 : start - 1 + count]
        return "\n".join(selected)
