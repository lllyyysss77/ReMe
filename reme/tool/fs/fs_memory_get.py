"""Memory get tool for reading specific snippets from memory files."""

import os
from pathlib import Path

from reme.core.op import BaseTool
from reme.core.schema import ToolCall


class FsMemoryGet(BaseTool):
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
                    "Safe snippet read from MEMORY.md, memory/*.md with optional from/lines; "
                    "use after memory_search to pull only the needed lines and keep context small."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the memory file to read (relative or absolute)",
                        },
                        "from": {
                            "type": "integer",
                            "description": "Starting line number (1-indexed, optional)",
                        },
                        "lines": {
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
        from_param: int | None = self.context.get("from", None)
        lines_param: int | None = self.context.get("lines", None)

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

        if from_param is None and lines_param is None:
            return content

        else:
            lines = content.split("\n")
            start = max(1, from_param if from_param is not None else 1)
            count = max(1, lines_param if lines_param is not None else len(lines))

            # Extract slice (1-indexed to 0-indexed conversion)
            selected = lines[start - 1 : start - 1 + count]
            text = "\n".join(selected)
            return text
