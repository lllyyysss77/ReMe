"""SQLite storage backend for memory index."""

import json
import sqlite3
import struct
import time
from pathlib import Path

from loguru import logger

from .base_memory_store import BaseMemoryStore
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult


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

    def __init__(self, db_path: str = ".reme/memory.db", vec_ext_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self.vec_ext_path = vec_ext_path

        self.conn: sqlite3.Connection | None = None

    @property
    def vector_table_name(self) -> str:
        """Get the name of the vector table for this store."""
        return f"chunks_vec_{self.store_name}"

    @property
    def fts_table_name(self) -> str:
        """Get the name of the FTS table for this store."""
        return f"chunks_fts_{self.store_name}"

    @property
    def chunks_table_name(self) -> str:
        """Get the name of the chunks table for this store."""
        return f"chunks_{self.store_name}"

    @property
    def files_table_name(self) -> str:
        """Get the name of the files table for this store."""
        return f"files_{self.store_name}"

    @staticmethod
    def vector_to_blob(embedding: list[float]) -> bytes:
        """Convert vector to binary blob for sqlite-vec."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    async def start(self) -> None:
        """Initialize database and load extensions."""
        if self.conn is not None:
            return

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
            try:
                import sqlite_vec

                ext_path = sqlite_vec.loadable_path()
                self.conn.load_extension(ext_path)
                self.vector_available = True
                logger.info(f"Loaded sqlite-vec from package: {ext_path}")

            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec from package: {e}")
                # Fallback: try common extension names
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

        # Files
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.files_table_name} (
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
            f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table_name} (
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

        # Vector table (sqlite-vec)
        if self.vector_available:
            cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.vector_table_name} USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}]
                )
            """,
            )
            logger.info(f"Created vector table (dims={self.embedding_dim})")

        # FTS table
        if self.fts_enabled:
            cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table_name} USING fts5(
                    text,
                    id UNINDEXED,
                    path UNINDEXED,
                    source UNINDEXED,
                    start_line UNINDEXED,
                    end_line UNINDEXED,
                    tokenize='trigram'
                )
            """,
            )
            self.fts_available = True
            logger.info("Created FTS5 table with trigram tokenizer")

        self.conn.commit()
        cursor.close()

    async def upsert_file(self, file_meta: FileMetadata, source: MemorySource, chunks: list[MemoryChunk]):
        """Insert or update file and its chunks."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("BEGIN")

            # Insert file
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.files_table_name} (path, source, hash, mtime, size)
                VALUES (?, ?, ?, ?, ?)
            """,
                (file_meta.path, source.value, file_meta.hash, file_meta.mtime_ms, file_meta.size),
            )

            # Insert chunks
            now = int(time.time() * 1000)
            for chunk in chunks:
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.chunks_table_name} (
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

                # Insert vector (vec0 doesn't support OR REPLACE, use DELETE + INSERT)
                if self.vector_available:
                    assert chunk.embedding, "Embedding is required for vector insert"
                    # Delete existing vector first
                    cursor.execute(
                        f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                        (chunk.id,),
                    )
                    # Then insert new vector
                    cursor.execute(
                        f"INSERT INTO {self.vector_table_name} (id, embedding) VALUES (?, ?)",
                        (chunk.id, self.vector_to_blob(chunk.embedding)),
                    )

                # Insert FTS
                if self.fts_available:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.fts_table_name} (
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

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def delete_file(self, path: str, source: MemorySource):
        """Delete file and all its chunks."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            # Get chunk IDs for vector deletion
            cursor.execute(
                f"SELECT id FROM {self.chunks_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            # Delete vectors
            if self.vector_available and chunk_ids:
                for chunk_id in chunk_ids:
                    try:
                        cursor.execute(
                            f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                            (chunk_id,),
                        )
                    except Exception as e:
                        logger.debug(f"Vector delete failed: {e}")

            # Delete FTS entries
            if self.fts_available:
                try:
                    cursor.execute(
                        f"DELETE FROM {self.fts_table_name} WHERE path = ? AND source = ?",
                        (path, source.value),
                    )
                except Exception as e:
                    logger.debug(f"FTS delete failed: {e}")

            # Delete chunks and file
            cursor.execute(
                f"DELETE FROM {self.chunks_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )
            cursor.execute(
                f"DELETE FROM {self.files_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def delete_file_chunks(self, path: str, chunk_ids: list[str]):
        """Delete specific chunks for a file."""
        if not chunk_ids:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            # Delete vectors
            if self.vector_available:
                for chunk_id in chunk_ids:
                    try:
                        cursor.execute(
                            f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                            (chunk_id,),
                        )
                    except Exception as e:
                        logger.debug(f"Vector delete failed for {chunk_id}: {e}")

            # Delete FTS entries
            if self.fts_available:
                placeholders = ",".join("?" * len(chunk_ids))
                try:
                    cursor.execute(
                        f"DELETE FROM {self.fts_table_name} WHERE id IN ({placeholders})",
                        chunk_ids,
                    )
                except Exception as e:
                    logger.debug(f"FTS delete failed: {e}")

            # Delete chunks
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"DELETE FROM {self.chunks_table_name} WHERE id IN ({placeholders})",
                chunk_ids,
            )

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def upsert_chunks(self, chunks: list[MemoryChunk], source: MemorySource):
        """Insert or update specific chunks without affecting other chunks."""
        if not chunks:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            now = int(time.time() * 1000)
            for chunk in chunks:
                # Insert/update chunk
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.chunks_table_name} (
                        id, path, source, start_line, end_line,
                        hash, text, embedding, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.id,
                        chunk.path,
                        source.value,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.hash,
                        chunk.text,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                        now,
                    ),
                )

                # Insert/update vector (vec0 doesn't support OR REPLACE, use DELETE + INSERT)
                if self.vector_available:
                    assert chunk.embedding, "Embedding is required for vector insert"
                    # Delete existing vector first
                    cursor.execute(
                        f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                        (chunk.id,),
                    )
                    # Then insert new vector
                    cursor.execute(
                        f"INSERT INTO {self.vector_table_name} (id, embedding) VALUES (?, ?)",
                        (chunk.id, self.vector_to_blob(chunk.embedding)),
                    )

                # Insert/update FTS
                if self.fts_available:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.fts_table_name} (
                            text, id, path, source, start_line, end_line
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            chunk.text,
                            chunk.id,
                            chunk.path,
                            source.value,
                            chunk.start_line,
                            chunk.end_line,
                        ),
                    )

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed files."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT path FROM {self.files_table_name} WHERE source = ?", (source.value,))
        paths = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return paths

    async def get_file_metadata(self, path: str, source: MemorySource) -> FileMetadata | None:
        """Get file metadata with chunk count."""
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT hash, mtime, size FROM {self.files_table_name} WHERE path = ? AND source = ?",
            (path, source.value),
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            return None

        hash_val, mtime, size = row
        cursor.execute(
            f"SELECT COUNT(*) FROM {self.chunks_table_name} WHERE path = ? AND source = ?",
            (path, source.value),
        )
        chunk_count = cursor.fetchone()[0]
        cursor.close()

        return FileMetadata(
            hash=hash_val,
            mtime_ms=mtime,
            size=size,
            path=path,
            chunk_count=chunk_count,
        )

    async def get_file_chunks(self, path: str, source: MemorySource) -> list[MemoryChunk]:
        """Get all chunks for a file."""
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT id, path, source, start_line, end_line, text, hash, embedding
            FROM {self.chunks_table_name} WHERE path = ? AND source = ?
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
            # vec0 requires 'k = ?' constraint for knn queries
            query_sql = f"""
                SELECT c.id, c.path, c.start_line, c.end_line, c.source, c.text, v.distance
                FROM {self.vector_table_name} v
                JOIN {self.chunks_table_name} c ON v.id = c.id
                WHERE v.embedding MATCH ?
                AND k = ?
            """
            query_params: list = [query_blob, limit]

            # Add source filter if specified
            if source_filter:
                query_sql += source_filter
                query_params.extend(params)

            # Order by distance (k constraint already limits results)
            query_sql += " ORDER BY v.distance"

            cursor.execute(query_sql, query_params)

            results = []
            for _, path, start, end, src, text, dist in cursor.fetchall():
                # Convert L2 distance to similarity score
                # For normalized vectors, L2 distance range is [0, 2]
                # Map to [1, 0] score range (higher score = more similar)
                score = max(0.0, 1.0 - dist / 2.0)
                snippet = text
                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=snippet,
                        source=MemorySource(src),
                        raw_metric=dist,
                    ),
                )

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        finally:
            cursor.close()

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query string for FTS5 search.

        Removes or escapes special characters that have special meaning in FTS5:
        - * (prefix match)
        - ? (not used in FTS5, but can cause issues)
        - " (phrase search, needs escaping)
        - : (column filter)
        - ^ (start of line anchor, not standard FTS5)
        - ' (single quote, causes syntax errors)
        - ` (backtick, can cause issues)
        - | (pipe, OR operator)
        - + (plus, can be used for required terms)
        - - (minus, NOT operator)
        - = (equals, can cause issues)
        - < > (angle brackets, comparison operators)
        - ! (exclamation, NOT operator variant)
        - @ # $ % & (other special chars)
        - "\"
        - / (slash, can interfere)
        - ; (semicolon, statement separator)
        - , (comma, can interfere with phrase parsing)

        Args:
            query: Raw query string

        Returns:
            Sanitized query string safe for FTS5
        """
        if not query:
            return ""

        # Remove FTS5 special characters that we don't want users to use
        # Keep only alphanumeric, spaces, periods, and underscores
        special_chars = [
            "*",
            "?",
            ":",
            "^",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "'",
            '"',
            "`",
            "|",
            "+",
            "-",
            "=",
            "<",
            ">",
            "!",
            "@",
            "#",
            "$",
            "%",
            "&",
            "\\",
            "/",
            ";",
            ",",
        ]
        cleaned = query
        for char in special_chars:
            cleaned = cleaned.replace(char, " ")

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned

    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform full-text search."""
        if not self.fts_available:
            return []

        # Sanitize and prepare query
        cleaned = self._sanitize_fts_query(query)
        if not cleaned:
            return []

        # Split into words and escape double quotes for FTS5 phrase matching
        words = cleaned.split()
        if not words:
            return []

        # Use OR operator for better recall - match any of the query words
        escaped_words = [word.replace('"', '""') for word in words]
        fts_query = " OR ".join(escaped_words)

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
                FROM {self.fts_table_name} fts
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
                snippet = text
                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=snippet,
                        source=MemorySource(src),
                        raw_metric=rank,
                    ),
                )

            return results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
        finally:
            cursor.close()

    async def clear_all(self):
        """Clear all indexed data."""
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")

        try:
            cursor.execute(f"DELETE FROM {self.files_table_name}")
            cursor.execute(f"DELETE FROM {self.chunks_table_name}")

            if self.vector_available:
                cursor.execute(f"DELETE FROM {self.vector_table_name}")

            if self.fts_available:
                cursor.execute(f"DELETE FROM {self.fts_table_name}")

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
