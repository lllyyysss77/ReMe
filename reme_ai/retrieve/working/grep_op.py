"""Grep text search operation module.

This module provides a tool operation for searching text patterns in files.
It enables efficient content-based search using regular expressions, with support
for glob pattern filtering and result limiting.
"""

from pathlib import Path
from typing import List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall
from loguru import logger
from reme_ai.utils.op_utils import run_shell_command


@C.register_op()
class GrepOp(BaseAsyncToolOp):
    """Grep text search operation.

    This operation searches for text patterns in files using regular expressions.
    Supports glob pattern filtering and result limiting.
    """

    file_path = __file__

    def __init__(self, **kwargs):
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "Grep",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "file_path": {
                    "type": "string",
                    "description": self.get_prompt("file_path"),
                    "required": True,
                },
                "pattern": {
                    "type": "string",
                    "description": self.get_prompt("pattern"),
                    "required": True,
                },
                "limit": {
                    "type": "number",
                    "description": self.get_prompt("limit"),
                    "required": False,
                },
            },
        }

        return ToolCall(**tool_params)

    async def async_execute(self):
        """Execute the grep search operation."""
        pattern: str = self.input_dict.get("pattern", "").strip()
        file_path: str | None | Path = self.input_dict.get("file_path", "")
        limit: int = self.input_dict.get("limit", 50)

        # Validate pattern
        if not pattern:
            raise ValueError("The 'pattern' parameter cannot be empty.")

        # Determine search directory
        if file_path:
            search_dir = Path(file_path).expanduser().resolve()
            if not search_dir.exists():
                raise ValueError(f"Search file_path does not exist: {search_dir}")
        else:
            search_dir = Path.cwd()

        # Build grep command
        cmd: List[str] = ["grep", "-RIni"]
        if limit:
            cmd.extend(["-m", str(limit)])
        cmd.extend(["--", pattern, str(search_dir)])

        logger.info(f"Running grep command: {' '.join(cmd)}")

        # Execute grep using run_shell_command
        stdout, stderr, returncode = await run_shell_command(cmd, timeout=None)

        assert returncode in (0, 1), f"grep failed with code {returncode}: {stderr.strip()}"

        output_text = stdout.strip()

        # Return raw grep output
        if not output_text:
            search_location = f'in file_path "{file_path}"' if file_path else "in the workspace directory"
            result_msg = f'No matches found for pattern "{pattern}" {search_location}.'
        else:
            result_msg = output_text

        self.set_output(result_msg)

    async def async_default_execute(self, e: Exception = None, **_kwargs):
        """Fill outputs with a default failure message when execution fails."""
        pattern: str = self.input_dict.get("pattern", "").strip()
        error_msg = f'Failed to search for pattern "{pattern}"'
        if e:
            error_msg += f": {str(e)}"
        self.set_output(error_msg)
