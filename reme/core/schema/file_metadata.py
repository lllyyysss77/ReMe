"""File metadata schema."""

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """File metadata with optional extended fields for various use cases."""

    # Core fields (always required)
    hash: str = Field(default=..., description="Hash of the file content")
    mtime_ms: float = Field(default=..., description="Last modification time in milliseconds")
    size: int = Field(default=..., description="File size in bytes")

    # Extended fields for session files
    path: str | None = Field(default=None, description="Relative path to the session file")
    abs_path: str | None = Field(default=None, description="Absolute path to the session file")
    content: str | None = Field(default=None, description="Parsed content from the session file")

    # Extended fields for statistics
    chunk_count: int | None = Field(default=None, description="Number of chunks in the file")
