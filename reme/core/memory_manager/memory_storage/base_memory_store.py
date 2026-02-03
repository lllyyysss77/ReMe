"""Base storage interface for memory manager."""

from abc import ABC, abstractmethod

from ...embedding import BaseEmbeddingModel
from ...enumeration import MemorySource
from ...schema import FileMetadata, MemoryIndexMeta, MemoryChunk, MemorySearchResult


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    def __init__(self, embedding_model: BaseEmbeddingModel):
        """Initialize"""
        self.embedding_model: BaseEmbeddingModel = embedding_model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding model's dimensionality."""
        return self.embedding_model.dimensions

    async def get_embedding(self, query: str, **kwargs) -> list[float]:
        """Get embedding for a single query string.

        Args:
            query: Input text to generate embedding for
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            Embedding vector as a list of floats
        """
        return await self.embedding_model.get_embedding(query, **kwargs)

    async def get_embeddings(self, queries: list[str], **kwargs) -> list[list[float]]:
        """Get embeddings for a batch of query strings.

        Args:
            queries: List of input texts to generate embeddings for
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            List of embedding vectors, each as a list of floats
        """
        return await self.embedding_model.get_embeddings(queries, **kwargs)

    async def get_chunk_embedding(self, chunk: MemoryChunk, **kwargs) -> MemoryChunk:
        """Generate and populate embedding field for a single MemoryChunk object.

        Args:
            chunk: MemoryChunk object containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same MemoryChunk object with populated embedding field
        """
        return await self.embedding_model.get_chunk_embedding(chunk, **kwargs)

    async def get_chunk_embeddings(self, chunks: list[MemoryChunk], **kwargs) -> list[MemoryChunk]:
        """Generate and populate embedding fields for a batch of MemoryChunk objects.

        Args:
            chunks: List of MemoryChunk objects containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same list of MemoryChunk objects with populated embedding fields
        """
        return await self.embedding_model.get_chunk_embeddings(chunks, **kwargs)

    def get_chunk_embedding_sync(self, chunk: MemoryChunk, **kwargs) -> MemoryChunk:
        """Synchronously generate and populate embedding field for a single MemoryChunk object.

        Args:
            chunk: MemoryChunk object containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same MemoryChunk object with populated embedding field
        """
        return self.embedding_model.get_chunk_embedding_sync(chunk, **kwargs)

    def get_chunk_embeddings_sync(self, chunks: list[MemoryChunk], **kwargs) -> list[MemoryChunk]:
        """Synchronously generate embeddings for a batch of MemoryChunk objects.

        Args:
            chunks: List of MemoryChunk objects containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same list of MemoryChunk objects with populated embedding fields
        """
        return self.embedding_model.get_chunk_embeddings_sync(chunks, **kwargs)

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
    async def get_file_hash(self, path: str, source: MemorySource) -> str | None:
        """Get the hash of an indexed file."""

    @abstractmethod
    async def get_file_metadata(self, path: str, source: MemorySource) -> FileMetadata | None:
        """Get full file metadata with statistics."""

    @abstractmethod
    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed file paths for a source."""

    @abstractmethod
    async def get_chunks(self, path: str, source: MemorySource) -> list[MemoryChunk]:
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
    async def read_meta(self, key: str) -> MemoryIndexMeta | None:
        """Read metadata value."""

    @abstractmethod
    async def write_meta(self, key: str, value: MemoryIndexMeta | dict):
        """Write metadata value."""

    @abstractmethod
    async def clear_all(self):
        """Clear all indexed data."""

    @abstractmethod
    async def close(self):
        """Close storage and release resources."""
