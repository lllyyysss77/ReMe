"""SQLite storage backend for memory index."""

import json
import sqlite3
import struct
import time
from pathlib import Path

from loguru import logger

from .base_memory_store import BaseMemoryStore
from ...embedding import BaseEmbeddingModel
from ...enumeration import MemorySource
from ...schema import FileMetadata, MemoryIndexMeta, MemoryChunk, MemorySearchResult


class SqliteMemoryStore(BaseMemoryStore):
    """SQLite memory storage with vector and full-text search.

    Inherits embedding methods from BaseMemoryStore:
    - get_chunk_embedding / get_chunk_embeddings (async)
    - get_chunk_embedding_sync / get_chunk_embeddings_sync (sync)
    - get_embedding / get_embeddings (async)

    Provides SQLite-backed persistent storage with:
    - Vector similarity search (via sqlite-vec extension)
    - Full-text search (via FTS5)
    - Efficient chunk and file metadata management
    """

    VECTOR_TABLE = "chunks_vec"
    FTS_TABLE = "chunks_fts"

    def __init__(
        self,
        db_path: str,
        embedding_model: BaseEmbeddingModel,
        vec_ext_path: str = "",
        fts_enabled: bool = True,
        snippet_max_chars: int = 700,
    ):
        super().__init__(embedding_model=embedding_model)
        self.db_path = db_path
        self.vec_ext_path = vec_ext_path
        self.fts_enabled = fts_enabled
        self.snippet_max_chars = snippet_max_chars
        self.conn: sqlite3.Connection | None = None
        self.vector_available = False
        self.fts_available = False

    @staticmethod
    def vector_to_blob(embedding: list[float]) -> bytes:
        """Convert vector to binary blob for sqlite-vec."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    async def start(self) -> None:
        """Initialize database and load extensions."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)

        # Load sqlite-vec extension
        if self.vec_ext_path:
            try:
                self.conn.load_extension(self.vec_ext_path)
                self.vector_available = True
                logger.info(f"Loaded sqlite-vec: {self.vec_ext_path}")
            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec: {e}")
        else:
            # Try common extension names
            for name in ["vec0", "sqlite_vec", "vector0"]:
                try:
                    self.conn.load_extension(name)
                    self.vector_available = True
                    logger.info(f"Loaded sqlite-vec: {name}")
                    break
                except Exception:
                    pass

        self.conn.enable_load_extension(False)
        await self._create_tables()

    async def _create_tables(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()

        # Metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """,
        )

        # Files
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT,
                source TEXT,
                hash TEXT,
                mtime REAL,
                size INTEGER,
                PRIMARY KEY (path, source)
            )
        """,
        )

        # Chunks
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                path TEXT,
                source TEXT,
                start_line INTEGER,
                end_line INTEGER,
                hash TEXT,
                text TEXT,
                embedding TEXT,
                updated_at INTEGER
            )
        """,
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_path_source
            ON chunks(path, source)
        """,
        )

        # Vector table (sqlite-vec)
        if self.vector_available:
            try:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.VECTOR_TABLE} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dim}]
                    )
                """,
                )
                logger.info(f"Created vector table (dims={self.embedding_dim})")
            except Exception as e:
                logger.warning(f"Failed to create vector table: {e}")
                self.vector_available = False

        # FTS table
        if self.fts_enabled:
            try:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.FTS_TABLE} USING fts5(
                        text,
                        id UNINDEXED,
                        path UNINDEXED,
                        source UNINDEXED,
                        start_line UNINDEXED,
                        end_line UNINDEXED
                    )
                """,
                )
                self.fts_available = True
                logger.info("Created FTS5 table")
            except Exception as e:
                logger.warning(f"Failed to create FTS table: {e}")
                self.fts_available = False

        self.conn.commit()
        cursor.close()

    async def upsert_file(self, file_meta: FileMetadata, source: MemorySource, chunks: list[MemoryChunk]):
        """Insert or update file and its chunks."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("BEGIN")
            await self._delete_file_internal(cursor, file_meta.path, source)

            # Insert file
            cursor.execute(
                """
                INSERT OR REPLACE INTO files (path, source, hash, mtime, size)
                VALUES (?, ?, ?, ?, ?)
            """,
                (file_meta.path, source.value, file_meta.hash, file_meta.mtime_ms, file_meta.size),
            )

            # Insert chunks
            now = int(time.time() * 1000)
            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        id, path, source, start_line, end_line,
                        hash, text, embedding, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.id,
                        file_meta.path,
                        source.value,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.hash,
                        chunk.text,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                        now,
                    ),
                )

                # Insert vector
                if self.vector_available and chunk.embedding:
                    try:
                        cursor.execute(
                            f"""
                            INSERT INTO {self.VECTOR_TABLE} (id, embedding)
                            VALUES (?, ?)
                        """,
                            (chunk.id, self.vector_to_blob(chunk.embedding)),
                        )
                    except Exception as e:
                        logger.debug(f"Vector insert failed: {e}")

                # Insert FTS
                if self.fts_available:
                    try:
                        cursor.execute(
                            f"""
                            INSERT INTO {self.FTS_TABLE} (
                                text, id, path, source, start_line, end_line
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                            (
                                chunk.text,
                                chunk.id,
                                file_meta.path,
                                source.value,
                                chunk.start_line,
                                chunk.end_line,
                            ),
                        )
                    except Exception as e:
                        logger.debug(f"FTS insert failed: {e}")

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def delete_file(self, path: str, source: MemorySource) -> None:
        """Delete file and all its chunks."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            await self._delete_file_internal(cursor, path, source)
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def _delete_file_internal(self, cursor: sqlite3.Cursor, path: str, source: MemorySource):
        """Internal delete helper."""
        # Get chunk IDs for vector deletion
        cursor.execute(
            "SELECT id FROM chunks WHERE path = ? AND source = ?",
            (path, source.value),
        )
        chunk_ids = [row[0] for row in cursor.fetchall()]

        # Delete vectors
        if self.vector_available and chunk_ids:
            for chunk_id in chunk_ids:
                try:
                    cursor.execute(
                        f"DELETE FROM {self.VECTOR_TABLE} WHERE id = ?",
                        (chunk_id,),
                    )
                except Exception as e:
                    logger.debug(f"Vector delete failed: {e}")

        # Delete FTS entries
        if self.fts_available:
            try:
                cursor.execute(
                    f"DELETE FROM {self.FTS_TABLE} WHERE path = ? AND source = ?",
                    (path, source.value),
                )
            except Exception as e:
                logger.debug(f"FTS delete failed: {e}")

        # Delete chunks and file
        cursor.execute(
            "DELETE FROM chunks WHERE path = ? AND source = ?",
            (path, source.value),
        )
        cursor.execute(
            "DELETE FROM files WHERE path = ? AND source = ?",
            (path, source.value),
        )

    async def get_file_hash(self, path: str, source: MemorySource) -> str | None:
        """Get file hash."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT hash FROM files WHERE path = ? AND source = ?",
            (path, source.value),
        )
        row = cursor.fetchone()
        cursor.close()
        return row[0] if row else None

    async def get_file_metadata(self, path: str, source: MemorySource) -> FileMetadata | None:
        """Get file metadata with chunk count."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT hash, mtime, size FROM files WHERE path = ? AND source = ?",
            (path, source.value),
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            return None

        hash_val, mtime, size = row
        cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE path = ? AND source = ?",
            (path, source.value),
        )
        chunk_count = cursor.fetchone()[0]
        cursor.close()

        return FileMetadata(
            hash=hash_val,
            mtime_ms=mtime,
            size=size,
            chunk_count=chunk_count,
        )

    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed files."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT path FROM files WHERE source = ?", (source.value,))
        paths = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return paths

    async def get_chunks(self, path: str, source: MemorySource) -> list[MemoryChunk]:
        """Get all chunks for a file."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, path, source, start_line, end_line, text, hash, embedding
            FROM chunks WHERE path = ? AND source = ?
            ORDER BY start_line
        """,
            (path, source.value),
        )

        chunks = []
        for row in cursor.fetchall():
            chunk_id, path_val, source_val, start, end, text, hash_val, emb_str = row
            # Parse embedding from JSON string
            embedding = None
            if emb_str:
                try:
                    embedding = json.loads(emb_str)
                except (json.JSONDecodeError, TypeError):
                    embedding = None

            chunks.append(
                MemoryChunk(
                    id=chunk_id,
                    path=path_val,
                    source=MemorySource(source_val),
                    start_line=start,
                    end_line=end,
                    text=text,
                    hash=hash_val,
                    embedding=embedding,
                ),
            )

        cursor.close()
        return chunks

    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        if not self.vector_available or not query:
            return []

        # Get query embedding
        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        cursor = self.conn.cursor()
        source_filter = ""
        params: list = []
        if sources:
            placeholders = ",".join("?" * len(sources))
            source_filter = f" AND c.source IN ({placeholders})"
            params = [s.value for s in sources]

        try:
            query_blob = self.vector_to_blob(query_embedding)
            # Correct SQLite-vec syntax for vector search with limit
            query_sql = f"""
                SELECT c.id, c.path, c.start_line, c.end_line, c.source, c.text, v.distance
                FROM {self.VECTOR_TABLE} v
                JOIN chunks c ON v.id = c.id
                WHERE v.embedding MATCH ?
            """
            query_params: list = [query_blob]

            # Add source filter if specified
            if source_filter:
                query_sql += source_filter
                query_params.extend(params)

            # Order and limit results
            query_sql += " ORDER BY v.distance LIMIT ?"
            query_params.append(str(limit))

            cursor.execute(query_sql, query_params)

            results = []
            for _, path, start, end, src, text, dist in cursor.fetchall():
                score = max(0.0, 1.0 - dist)
                snippet = text[: self.snippet_max_chars] if len(text) > self.snippet_max_chars else text
                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=snippet,
                        source=MemorySource(src),
                    ),
                )

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        finally:
            cursor.close()

    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform full-text search."""
        if not self.fts_available:
            return []

        # Build FTS5 query, escaping quotes
        cleaned = query.strip().replace('"', '""')
        if not cleaned:
            return []
        fts_query = f'"{cleaned}"'

        cursor = self.conn.cursor()
        source_filter = ""
        params: list = [fts_query]
        if sources:
            placeholders = ",".join("?" * len(sources))
            source_filter = f" AND fts.source IN ({placeholders})"
            params.extend([s.value for s in sources])
        params.append(limit)

        try:
            cursor.execute(
                f"""
                SELECT fts.id, fts.path, fts.start_line, fts.end_line,
                       fts.source, fts.text, rank
                FROM {self.FTS_TABLE} fts
                WHERE fts.text MATCH ?{source_filter}
                ORDER BY rank
                LIMIT ?
            """,
                params,
            )

            results = []
            for _, path, start, end, src, text, rank in cursor.fetchall():
                # Convert BM25 rank (negative) to 0-1 score (higher=better)
                score = max(0.0, 1.0 / (1.0 + abs(rank)))
                snippet = text[: self.snippet_max_chars] if len(text) > self.snippet_max_chars else text
                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=snippet,
                        source=MemorySource(src),
                    ),
                )

            return results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
        finally:
            cursor.close()

    async def read_meta(self, key: str) -> MemoryIndexMeta | None:
        """Read metadata value."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None

        return MemoryIndexMeta(**json.loads(row[0]))

    async def write_meta(self, key: str, value: MemoryIndexMeta | dict) -> None:
        """Write metadata value."""
        data = value.model_dump() if isinstance(value, MemoryIndexMeta) else value
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO meta (key, value)
            VALUES (?, ?)
        """,
            (key, json.dumps(data)),
        )
        self.conn.commit()
        cursor.close()

    async def clear_all(self):
        """Clear all indexed data."""
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")

        try:
            cursor.execute("DELETE FROM files")
            cursor.execute("DELETE FROM chunks")

            if self.vector_available:
                try:
                    cursor.execute(f"DELETE FROM {self.VECTOR_TABLE}")
                except Exception as e:
                    logger.debug(f"Vector clear failed: {e}")

            if self.fts_available:
                try:
                    cursor.execute(f"DELETE FROM {self.FTS_TABLE}")
                except Exception as e:
                    logger.debug(f"FTS clear failed: {e}")

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
