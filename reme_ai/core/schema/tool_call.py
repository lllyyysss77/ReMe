"""Model definitions for MCP tools and LLM tool call interactions."""

import json
from typing import Dict, List, Literal, Optional, Any

from mcp.types import Tool
from pydantic import BaseModel, Field, model_validator, ConfigDict

TOOL_ATTR_TYPE = Literal["string", "array", "integer", "number", "boolean", "object"]


class ToolAttr(BaseModel):
    """Represent attributes for tool parameters in a JSON schema format."""

    type: TOOL_ATTR_TYPE = Field(default="string", description="Attribute data type")
    description: str = Field(default="", description="Attribute purpose")
    required: bool = Field(default=True, description="Whether the attribute is mandatory")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values")
    items: Dict[str, Any] = Field(default_factory=dict, description="Schema for array items")

    model_config = ConfigDict(extra="allow")

    def simple_input_dump(self) -> dict:
        """Export attribute as a standard JSON schema property dictionary."""
        res: dict = {"type": self.type, "description": self.description}
        if self.enum:
            res["enum"] = self.enum
        if self.items:
            res["items"] = self.items
        return res


class ToolCall(BaseModel):
    """Handle tool definitions and execution arguments for LLM integrations."""

    index: int = Field(default=0)
    id: str = Field(default="")
    type: str = Field(default="function")
    name: str = Field(default="")
    arguments: str = Field(default="{}", description="JSON string of execution arguments")
    description: str = Field(default="")
    input_schema: Dict[str, ToolAttr] = Field(default_factory=dict)
    output_schema: Dict[str, ToolAttr] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def init_tool_call(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw API response data to the internal ToolCall structure."""
        res = data.copy()
        t_type = res.get("type", "function")
        inner = res.get(t_type, {})

        # Extract basic function metadata
        for key in ("name", "arguments", "description"):
            if key in inner:
                res[key] = inner[key]

        # Parse JSON schema parameters into ToolAttr objects
        params = inner.get("parameters", {})
        if params:
            props = params.get("properties", {})
            reqs = params.get("required", [])
            res["input_schema"] = {k: ToolAttr(**v, required=k in reqs) for k, v in props.items()}
        return res

    @property
    def argument_dict(self) -> dict:
        """Parse the arguments string into a dictionary."""
        return json.loads(self.arguments)

    def check_argument(self) -> bool:
        """Verify if the arguments string is valid JSON."""
        try:
            _ = self.argument_dict
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def _build_schema_dict(schema: Dict[str, ToolAttr]) -> dict:
        """Construct a JSON schema object from a dictionary of ToolAttrs."""
        return {
            "type": "object",
            "properties": {k: v.simple_input_dump() for k, v in schema.items()},
            "required": [k for k, v in schema.items() if v.required],
        }

    def simple_input_dump(self) -> dict:
        """Format the tool definition for LLM provider API requests."""
        return {
            "type": self.type,
            self.type: {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_schema_dict(self.input_schema),
            },
        }

    def simple_output_dump(self) -> dict:
        """Format the tool call result for LLM provider API responses."""
        return {
            "index": self.index,
            "id": self.id,
            "type": self.type,
            self.type: {"arguments": self.arguments, "name": self.name},
        }

    @classmethod
    def from_mcp_tool(cls, tool: Tool) -> "ToolCall":
        """Create a ToolCall instance from an MCP Tool object."""
        props = tool.inputSchema.get("properties", {})
        reqs = tool.inputSchema.get("required", [])
        return cls(
            name=tool.name,
            description=tool.description or "",
            input_schema={k: ToolAttr(**v, required=k in reqs) for k, v in props.items()},
        )

    def to_mcp_tool(self) -> Tool:
        """Convert the current instance into an MCP Tool object."""
        kwargs = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self._build_schema_dict(self.input_schema),
        }
        if self.output_schema:
            kwargs["outputSchema"] = self._build_schema_dict(self.output_schema)
        return Tool(**kwargs)
