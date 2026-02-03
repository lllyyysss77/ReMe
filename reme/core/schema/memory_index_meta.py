"""Memory index metadata schema."""

from typing import Optional

from pydantic import BaseModel, Field


class MemoryIndexMeta(BaseModel):
    """Metadata for memory index configuration."""

    model: str = Field(..., description="Name of the embedding model")
    chunk_tokens: int = Field(..., description="Maximum tokens per chunk")
    chunk_overlap: int = Field(..., description="Number of overlapping tokens between chunks")
    vector_dims: Optional[int] = Field(default=None, description="Vector embedding dimensions")
