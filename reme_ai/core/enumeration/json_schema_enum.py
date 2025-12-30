"""Defines the standard data types supported by JSON Schema."""

from enum import Enum


class JsonSchemaEnum(str, Enum):
    """Enumeration of valid JSON Schema data types."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    OBJECT = "object"
    ARRAY = "array"
    BOOLEAN = "boolean"
    NULL = "null"

    def __str__(self) -> str:
        """Returns the string representation of the enum value."""
        return self.value
