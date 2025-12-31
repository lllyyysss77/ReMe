"""Defines the data structure for processing incoming user requests and message history."""

from typing import List

from pydantic import Field, BaseModel, ConfigDict

from .message import Message


class Request(BaseModel):
    """Represents a structured request payload containing a query, message list, and metadata."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(default="")
    messages: List[Message] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
