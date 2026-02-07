"""Base storage interface for memory manager."""

from abc import ABC, abstractmethod

from ..embedding import BaseEmbeddingModel
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    def __init__(
        self,
        store_name: str,
        embedding_model: BaseEmbeddingModel,
        fts_enabled: bool = True,
        snippet_max_chars: int = 700,
        **kwargs,
    ):
        """Initialize"""
        self.store_name: str = store_name
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.fts_enabled: bool = fts_enabled
        self.snippet_max_chars: int = snippet_max_chars
        self.kwargs: dict = kwargs

        self.vector_available = False
        self.fts_available = False

    @property
    def embedding_dim(self) -> int:
        """Get the embedding model's dimensionality."""
        return self.embedding_model.dimensions

    async def get_embedding(self, query: str, **kwargs) -> list[float]:
        """Get embedding for a single query string."""
        return await self.embedding_model.get_embedding(query, **kwargs)

    async def get_embeddings(self, queries: list[str], **kwargs) -> list[list[float]]:
        """Get embeddings for a batch of query strings."""
        return await self.embedding_model.get_embeddings(queries, **kwargs)

    async def get_chunk_embedding(self, chunk: MemoryChunk, **kwargs) -> MemoryChunk:
        """Generate and populate embedding field for a single MemoryChunk object."""
        return await self.embedding_model.get_chunk_embedding(chunk, **kwargs)

    async def get_chunk_embeddings(self, chunks: list[MemoryChunk], **kwargs) -> list[MemoryChunk]:
        """Generate and populate embedding fields for a batch of MemoryChunk objects."""
        return await self.embedding_model.get_chunk_embeddings(chunks, **kwargs)

    @abstractmethod
    async def start(self):
        """Initialize the storage backend."""

    @abstractmethod
    async def upsert_file(self, file_meta: FileMetadata, source: MemorySource, chunks: list[MemoryChunk]):
        """Insert or update a file and its chunks."""

    @abstractmethod
    async def delete_file(self, path: str, source: MemorySource):
        """Delete a file and all its chunks."""

    @abstractmethod
    async def delete_file_chunks(self, path: str, chunk_ids: list[str]):
        """Delete chunks for a file."""

    @abstractmethod
    async def upsert_chunks(self, chunks: list[MemoryChunk], source: MemorySource):
        """Insert or update specific chunks without affecting other chunks."""

    @abstractmethod
    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed file paths for a source."""

    @abstractmethod
    async def get_file_metadata(self, path: str, source: MemorySource) -> FileMetadata | None:
        """Get full file metadata with statistics."""

    @abstractmethod
    async def get_file_chunks(self, path: str, source: MemorySource) -> list[MemoryChunk]:
        """Get all chunks for a file."""

    @abstractmethod
    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search.

        Args:
            query: Query embedding vector
            limit: Maximum number of results
            sources: Optional list of sources to filter

        Returns:
            List of search results sorted by similarity
        """

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform keyword/full-text search.

        Args:
            query: Search query text
            limit: Maximum number of results
            sources: Optional list of sources to filter

        Returns:
            List of search results sorted by relevance
        """

    @abstractmethod
    async def clear_all(self):
        """Clear all indexed data."""

    @abstractmethod
    async def close(self):
        """Close storage and release resources."""
