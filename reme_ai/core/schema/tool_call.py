"""
MCP Tool Schema definitions for recursive JSON Schema representation.
"""
import json
from typing import Any, Dict, List, Literal, Optional, Union

from mcp.types import Tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..enumeration.json_schema_enum import JsonSchemaEnum


class ToolAttr(BaseModel):
    """Recursive model representing JSON Schema attributes for tool parameters."""
    model_config = ConfigDict(extra="allow")

    type: Literal[
        JsonSchemaEnum.STRING.value,
        JsonSchemaEnum.NUMBER.value,
        JsonSchemaEnum.INTEGER.value,
        JsonSchemaEnum.OBJECT.value,
        JsonSchemaEnum.ARRAY.value,
        JsonSchemaEnum.BOOLEAN.value,
        JsonSchemaEnum.NULL.value,
    ] = Field(
        default=JsonSchemaEnum.STRING.value, 
        description="The data type of the attribute"
    )
    description: Optional[str] = Field(default=None, description="Description of the attribute")
    required: Optional[List[str]] = Field(default=None, description="Required property names for object types")
    properties: Optional[Dict[str, "ToolAttr"]] = Field(default=None, description="Child properties for objects")
    items: Optional[Union[Dict[str, Any], "ToolAttr"]] = Field(default=None, description="Schema for array items")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values for the attribute")


    def simple_input_dump(self) -> dict:
        """Serializes the attribute into a standard JSON Schema dictionary."""
        res: dict = {"type": self.type}
        if self.description:
            res["description"] = self.description
        if self.enum:
            res["enum"] = self.enum

        if self.type == "object" and self.properties:
            res["properties"] = {k: v.simple_input_dump() if isinstance(v, ToolAttr) else v
                                 for k, v in self.properties.items()}
            if self.required:
                res["required"] = self.required

        if self.type == "array" and self.items:
            res["items"] = self.items.simple_input_dump() if isinstance(self.items, ToolAttr) else self.items

        return res


# Enable recursive type resolution
ToolAttr.model_rebuild()


class ToolCall(BaseModel):
    """Model representing a tool definition and its call structure."""

    index: int = 0
    id: str = ""
    type: str = "function"
    name: str = ""
    arguments: str = Field(default="", description="JSON string of tool execution arguments")
    description: str = ""
    input_schema: Dict[str, ToolAttr] = Field(default_factory=dict)
    input_required: List[str] = Field(default_factory=list)
    output_schema: Dict[str, ToolAttr] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def init_tool_call(cls, data: dict) -> dict:
        """Initializes the model by parsing tool-specific body data."""
        data = data.copy()
        t_type = data.get("type", "function")
        body = data.get(t_type, {})

        data["name"] = body.get("name", data.get("name", ""))
        data["arguments"] = body.get("arguments", data.get("arguments", ""))
        data["description"] = body.get("description", data.get("description", ""))

        if "parameters" in body:
            params = body["parameters"]
            data["input_required"] = params.get("required", [])
            data["input_schema"] = {k: ToolAttr(**v) for k, v in params.get("properties", {}).items()}

        return data

    def _build_full_schema(self) -> dict:
        """Generates the top-level JSON Schema object for tool parameters."""
        return {
            "type": "object",
            "properties": {k: v.simple_input_dump() for k, v in self.input_schema.items()},
            "required": self.input_required,
        }

    def simple_input_dump(self) -> dict:
        """Returns a standardized tool definition dictionary."""
        return {
            "type": self.type,
            self.type: {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_full_schema(),
            },
        }

    @classmethod
    def from_mcp_tool(cls, tool: Tool) -> "ToolCall":
        """Creates a ToolCall instance from an MCP Tool object."""
        schema = tool.inputSchema
        return cls(
            name=tool.name,
            description=tool.description or "",
            input_schema={k: ToolAttr(**v) for k, v in schema.get("properties", {}).items()},
            input_required=schema.get("required", []),
        )

    def to_mcp_tool(self) -> Tool:
        """Converts the instance back into an MCP Tool object."""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self._build_full_schema(),
        )

    @property
    def argument_dict(self) -> dict:
        """Parse and return arguments as a dictionary."""
        return json.loads(self.arguments)

    def check_argument(self) -> bool:
        """Check if arguments can be parsed as valid JSON."""
        try:
            _ = self.argument_dict
            return True
        except Exception:
            return False

    def simple_output_dump(self) -> dict:
        """Convert ToolCall to output format dictionary for API responses."""
        return {
            "index": self.index,
            "id": self.id,
            self.type: {
                "arguments": self.arguments,
                "name": self.name,
            },
            "type": self.type,
        }
