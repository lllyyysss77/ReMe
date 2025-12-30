"""Data models for multi-modal conversation history and LLM interaction trajectories."""

import datetime
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tool_call import ToolCall
from ..enumeration import Role


class ContentBlock(BaseModel):
    """Individual unit of multi-modal content like text, images, or video."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="")
    content: str | dict | list = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def init_block(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Dynamically maps the type-specific key to the content field."""
        content_type = data.get("type", "")
        if content_type and content_type in data:
            data["content"] = data[content_type]
        return data

    def simple_dump(self) -> dict[str, Any]:
        """Serializes the block into an API-compatible dictionary format."""
        return {
            "type": self.type,
            self.type: self.content,
            **self.model_extra,
        }


class Message(BaseModel):
    """Data model for a single dialogue entry including roles and tool interactions."""

    name: str | None = Field(default=None)
    role: Role = Field(default=Role.USER)
    content: str | list[ContentBlock] = Field(default="")
    reasoning_content: str = Field(default="")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = Field(default="")
    time_created: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def dump_content(self) -> str | list[dict[str, Any]]:
        """Returns content as a raw string or a list of serialized blocks."""
        if isinstance(self.content, str):
            return self.content
        return [block.simple_dump() for block in self.content]

    def simple_dump(self, add_reasoning: bool = True) -> dict[str, Any]:
        """Transforms the message into a simplified dictionary for standard APIs."""
        result = {"role": self.role.value, "content": self.dump_content()}

        if add_reasoning and self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            result["tool_calls"] = [tc.simple_output_dump() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result

    def format_message(
        self,
        index: int | None = None,
        add_time: bool = False,
        use_name: bool = False,
        add_reasoning: bool = True,
        add_tools: bool = True,
    ) -> str:
        """Generates a human-readable string representation of the message."""
        prefix = f"round{index} " if index is not None else ""
        time_str = f"[{self.time_created}] " if add_time else ""
        header = f"{self.name or self.role.value if use_name else self.role.value}:\n"

        lines = [f"{prefix}{time_str}{header}"]

        if add_reasoning and self.reasoning_content:
            lines.append(f"{self.reasoning_content}\n")

        if isinstance(self.content, str):
            lines.append(self.content)
        elif isinstance(self.content, list):
            for block in self.content:
                text = (
                    block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False)
                )
                lines.append(str(text))

        if add_tools and self.tool_calls:
            for tc in self.tool_calls:
                lines.append(f" - tool_call={tc.name} params={tc.arguments}")

        return "\n".join(lines).strip()


class Trajectory(BaseModel):
    """Sequence of messages representing a full conversation session and its evaluation."""

    task_id: str = Field(default="")
    messages: list[Message] = Field(default_factory=list)
    score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
