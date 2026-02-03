"""Memory Index Manager - Main coordination layer.

This module provides the main MemoryIndexManager class that coordinates
file watching, embedding generation, and search operations across memory files
and session transcripts.
"""

import asyncio
import json
import os
import re
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, Field
from watchfiles import awatch

from .ingestion.chunking import chunk_markdown
from .memory_storage.sqlite_memory_store import SqliteMemoryStore
from .utils.hashing import hash_text
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemorySearchResult

# Constants
SNIPPET_MAX_CHARS = 700
SESSION_DIRTY_DEBOUNCE_MS = 5000
EMBEDDING_BATCH_MAX_TOKENS = 8000
EMBEDDING_APPROX_CHARS_PER_TOKEN = 1
EMBEDDING_INDEX_CONCURRENCY = 4
EMBEDDING_RETRY_MAX_ATTEMPTS = 3
EMBEDDING_RETRY_BASE_DELAY_MS = 500
EMBEDDING_RETRY_MAX_DELAY_MS = 8000
BATCH_FAILURE_LIMIT = 2
SESSION_DELTA_READ_CHUNK_BYTES = 64 * 1024
EMBEDDING_QUERY_TIMEOUT_REMOTE_MS = 60_000
EMBEDDING_QUERY_TIMEOUT_LOCAL_MS = 5 * 60_000
EMBEDDING_BATCH_TIMEOUT_REMOTE_MS = 2 * 60_000
EMBEDDING_BATCH_TIMEOUT_LOCAL_MS = 10 * 60_000


class MemorySyncProgressUpdate(BaseModel):
    """Progress update for memory sync operations."""

    completed: int = Field(default=..., description="Number of items completed")
    total: int = Field(default=..., description="Total number of items to process")
    label: str | None = Field(default=None, description="Optional label for the progress operation")


class MemorySyncProgressState(BaseModel):
    """Internal state for tracking sync progress."""

    completed: int = Field(default=0, description="Number of items completed")
    total: int = Field(default=0, description="Total number of items to process")
    label: str | None = Field(default=None, description="Optional label for the progress operation")
    report: Callable[[MemorySyncProgressUpdate], None] | None = Field(
        default=None,
        description="Callback function to report progress updates",
    )


class SessionDelta(BaseModel):
    """Tracks incremental changes in session files."""

    last_size: int = Field(default=0, description="Last known size of the session file")
    pending_bytes: int = Field(default=0, description="Number of pending bytes to process")
    pending_messages: int = Field(default=0, description="Number of pending messages to process")


class MemorySearchConfig(BaseModel):
    """Configuration for memory search operations."""

    model: str = Field(default="default", description="Model name for embeddings")
    sources: list[MemorySource] = Field(default_factory=lambda: [MemorySource.MEMORY], description="Sources to search")
    extra_paths: list[str] = Field(default_factory=list, description="Additional paths to include in search")
    store_path: str = Field(default="memory.db", description="Path to SQLite database file")
    vector_enabled: bool = Field(default=True, description="Whether to enable vector search")
    vector_extension_path: str | None = Field(default=None, description="Path to vector extension for SQLite")
    fts_enabled: bool = Field(default=True, description="Whether to enable full-text search")
    chunk_tokens: int = Field(default=300, description="Number of tokens per chunk")
    chunk_overlap: int = Field(default=30, description="Number of overlapping tokens between chunks")
    watch_enabled: bool = Field(default=True, description="Whether to enable file watching")
    watch_debounce_ms: int = Field(default=1000, description="Debounce time for file watcher in milliseconds")
    interval_minutes: int = Field(default=0, description="Interval between automatic syncs in minutes (0 to disable)")
    sync_on_search: bool = Field(default=True, description="Whether to sync before search operations")
    sync_on_session_start: bool = Field(default=True, description="Whether to sync when a session starts")
    query_min_score: float = Field(default=0.3, description="Minimum relevance score for search results")
    query_max_results: int = Field(default=10, description="Maximum number of search results to return")
    hybrid_enabled: bool = Field(default=True, description="Whether to use hybrid vector + keyword search")
    hybrid_vector_weight: float = Field(default=0.7, description="Weight for vector search in hybrid scoring")
    hybrid_text_weight: float = Field(default=0.3, description="Weight for text search in hybrid scoring")
    hybrid_candidate_multiplier: float = Field(
        default=2.0,
        description="Multiplier for number of candidates to consider in hybrid search",
    )
    session_delta_bytes: int = Field(
        default=0,
        description="Threshold for session sync based on bytes changed (0 for any change)",
    )
    session_delta_messages: int = Field(
        default=5,
        description="Threshold for session sync based on messages changed (0 for any change)",
    )


# Global cache for manager instances
INDEX_CACHE: dict[str, "MemoryIndexManager"] = {}


class MemoryIndexManager:
    """Main memory index manager coordinating all memory operations."""

    # ============================================================================
    # Initialization and Lifecycle
    # ============================================================================

    def __init__(
        self,
        agent_id: str,
        workspace_dir: str,
        settings: MemorySearchConfig,
        store: SqliteMemoryStore,
    ):
        """Initialize the memory index manager."""

        self.agent_id = agent_id
        self.workspace_dir = workspace_dir
        self.settings = settings
        self.store = store

        # State tracking
        self.sources = set(settings.sources)
        self.closed = False
        self.dirty = MemorySource.MEMORY in self.sources
        self.sessions_dirty = False
        self.sessions_dirty_files: set[str] = set()
        self.session_pending_files: set[str] = set()
        self.session_deltas: dict[str, SessionDelta] = {}
        self.session_warm: set[str] = set()

        # Sync control
        self.syncing: asyncio.Task | None = None
        self.watch_task: asyncio.Task | None = None
        self.session_watch_task: asyncio.Task | None = None
        self.interval_task: asyncio.Task | None = None

        # Batch failure tracking
        self.batch_failure_count = 0
        self.batch_failure_last_error: str | None = None
        self.batch_failure_lock = asyncio.Lock()

    async def close(self) -> None:
        """Close the manager and release resources."""
        if self.closed:
            return

        self.closed = True

        # Cancel all background tasks
        if self.watch_task:
            self.watch_task.cancel()
        if self.session_watch_task:
            self.session_watch_task.cancel()
        if self.interval_task:
            self.interval_task.cancel()

        await self.store.close()

    # ============================================================================
    # Public API Methods
    # ============================================================================

    async def warm_session(self, session_key: str | None = None):
        """Pre-sync memory before a session starts."""
        if not self.settings.sync_on_session_start:
            return

        key = (session_key or "").strip()
        if key and key in self.session_warm:
            return

        await self.sync(reason="session-start")

        if key:
            self.session_warm.add(key)

    async def sync(
        self,
        reason: str | None = None,
        force: bool = False,
        progress: Callable[[MemorySyncProgressUpdate], None] | None = None,
    ):
        """Synchronize memory index with file system."""
        if self.syncing:
            await self.syncing
            return

        self.syncing = asyncio.create_task(self._run_sync(reason, force, progress))
        try:
            await self.syncing
        finally:
            self.syncing = None

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
        session_key: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search indexed memory with hybrid vector + keyword search.

        Args:
            query: Search query text
            max_results: Maximum number of results to return
            min_score: Minimum relevance score threshold
            session_key: Optional session key for warmup

        Returns:
            List of search results sorted by relevance
        """
        await self.warm_session(session_key)

        if self.settings.sync_on_search and (self.dirty or self.sessions_dirty):
            try:
                await self.sync(reason="search")
            except Exception as err:
                logger.warning(f"memory sync failed (search): {err}")

        cleaned = query.strip()
        if not cleaned:
            return []

        min_score = min_score if min_score is not None else self.settings.query_min_score
        max_results = max_results if max_results is not None else self.settings.query_max_results

        hybrid = self.settings.hybrid_enabled
        candidates = min(200, max(1, int(max_results * self.settings.hybrid_candidate_multiplier)))

        # Run keyword search if hybrid enabled
        keyword_results = []
        if hybrid:
            keyword_results = await self._search_keyword(cleaned, candidates)

        # Perform vector search
        vector_results = await self._search_vector(cleaned, candidates)

        if not hybrid:
            return [r for r in vector_results if r.score >= min_score][:max_results]

        merged = self._merge_hybrid_results(
            vector=vector_results,
            keyword=keyword_results,
            vector_weight=self.settings.hybrid_vector_weight,
            text_weight=self.settings.hybrid_text_weight,
        )

        return [r for r in merged if r.score >= min_score][:max_results]

    async def read_file(
        self,
        rel_path: str,
        from_line: int | None = None,
        num_lines: int | None = None,
    ) -> dict[str, str]:
        """Read a memory file with optional line range.

        Args:
            rel_path: Relative path to file
            from_line: Starting line number (1-indexed)
            num_lines: Number of lines to read

        Returns:
            Dictionary with 'text' and 'path' keys

        Raises:
            ValueError: If path is invalid or not allowed
        """
        raw_path = rel_path.strip()
        assert raw_path, "path required"
        abs_path = os.path.abspath(os.path.join(self.workspace_dir, raw_path))
        rel_path_clean = os.path.relpath(abs_path, self.workspace_dir)

        in_workspace = not rel_path_clean.startswith("..") and not os.path.isabs(rel_path_clean)
        allowed = in_workspace and self._is_memory_path(rel_path_clean)

        if not allowed and self.settings.extra_paths:
            for extra in self.settings.extra_paths:
                extra_abs = os.path.abspath(extra)
                if abs_path.startswith(extra_abs):
                    allowed = True
                    break

        if not allowed:
            raise ValueError("path required")

        if not abs_path.endswith(".md"):
            raise ValueError("path required")

        # Read file
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        if from_line is None and num_lines is None:
            return {"text": content, "path": rel_path_clean}

        lines = content.split("\n")
        start = max(1, from_line or 1)
        count = max(1, num_lines or len(lines))
        slice_lines = lines[start - 1 : start - 1 + count]

        return {"text": "\n".join(slice_lines), "path": rel_path_clean}

    # ============================================================================
    # Sync Logic
    # ============================================================================

    async def _run_sync(
        self,
        reason: str | None,
        force: bool,
        progress_callback: Callable[[MemorySyncProgressUpdate], None] | None,
    ):
        """Execute sync operation."""
        progress = MemorySyncProgressState()
        if progress_callback:
            progress.report = progress_callback

        should_sync_memory = MemorySource.MEMORY in self.sources and (force or self.dirty)
        should_sync_sessions = self._should_sync_sessions(reason, force)

        if should_sync_memory:
            await self._sync_memory_files(progress)
            self.dirty = False

        if should_sync_sessions:
            await self._sync_session_files(progress)
            self.sessions_dirty = False
            self.sessions_dirty_files.clear()
        elif len(self.sessions_dirty_files) > 0:
            self.sessions_dirty = True
        else:
            self.sessions_dirty = False

    def _should_sync_sessions(self, reason: str | None, force: bool) -> bool:
        """Check if session sync is needed."""
        if MemorySource.SESSIONS not in self.sources:
            return False

        if force:
            return True

        if reason in ("session-start", "watch"):
            return False

        return self.sessions_dirty and len(self.sessions_dirty_files) > 0

    async def _sync_memory_files(self, progress: MemorySyncProgressState):
        """Sync memory markdown files."""
        files = self._list_memory_files()
        logger.debug("memory sync: indexing memory files", files=len(files))

        active_paths = {f.path for f in files}
        if progress.report:
            progress.total += len(files)
            progress.report(
                MemorySyncProgressUpdate(
                    completed=progress.completed,
                    total=progress.total,
                    label="Indexing memory files…",
                ),
            )

        tasks = []
        for file_entry in files:
            task = self._index_memory_file(file_entry, progress)
            tasks.append(task)
        await asyncio.gather(*tasks)

        indexed = await self.store.list_files(MemorySource.MEMORY)
        for stale_path in indexed:
            if stale_path not in active_paths:
                await self.store.delete_file(stale_path, MemorySource.MEMORY)

    async def _sync_session_files(self, progress: MemorySyncProgressState):
        """Sync session transcript files."""
        files = self._list_session_files()
        logger.debug(
            "memory sync: indexing session files",
            files=len(files),
            index_all=len(self.sessions_dirty_files) == 0,
            dirty_files=len(self.sessions_dirty_files),
        )

        if progress.report:
            progress.total += len(files)
            progress.report(
                MemorySyncProgressUpdate(
                    completed=progress.completed,
                    total=progress.total,
                    label="Indexing session files...",
                ),
            )

        active_paths = set()
        tasks = []

        for abs_path in files:
            rel_path = self._session_path_for_file(abs_path)
            active_paths.add(rel_path)

            if len(self.sessions_dirty_files) == 0 or abs_path in self.sessions_dirty_files:
                task = self._index_session_file(abs_path, progress)
                tasks.append(task)
            else:
                if progress.report:
                    progress.completed += 1
                    progress.report(MemorySyncProgressUpdate(completed=progress.completed, total=progress.total))
        await asyncio.gather(*tasks)

        indexed = await self.store.list_files(MemorySource.SESSIONS)
        for stale_path in indexed:
            if stale_path not in active_paths:
                await self.store.delete_file(stale_path, MemorySource.SESSIONS)

    # ============================================================================
    # File Indexing
    # ============================================================================

    async def _index_memory_file(self, file_meta: FileMetadata, progress: MemorySyncProgressState):
        """Index a single memory file."""
        existing_meta = await self.store.get_file_metadata(file_meta.path, MemorySource.MEMORY)
        if existing_meta and existing_meta.hash == file_meta.hash:
            if progress.report:
                progress.completed += 1
                progress.report(MemorySyncProgressUpdate(completed=progress.completed, total=progress.total))
            return

        # Read and chunk file
        with open(file_meta.abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_markdown(
            content,
            file_meta.path,
            MemorySource.MEMORY,
            self.settings.chunk_tokens,
            self.settings.chunk_overlap,
        )

        chunks = [c for c in chunks if c.text.strip()]

        if chunks:
            chunks = await self.store.get_chunk_embeddings(chunks)

        await self.store.upsert_file(file_meta, MemorySource.MEMORY, chunks)
        if progress.report:
            progress.completed += 1
            progress.report(MemorySyncProgressUpdate(completed=progress.completed, total=progress.total))

    async def _index_session_file(self, abs_path: str, progress: MemorySyncProgressState):
        """Index a single session transcript file."""
        file_meta = self._build_session_file_meta(abs_path)
        if not file_meta:
            if progress.report:
                progress.completed += 1
                progress.report(MemorySyncProgressUpdate(completed=progress.completed, total=progress.total))
            return

        existing_meta = await self.store.get_file_metadata(file_meta.path, MemorySource.SESSIONS)
        if existing_meta and existing_meta.hash == file_meta.hash:
            self._reset_session_delta(abs_path, file_meta.size)
            if progress.report:
                progress.completed += 1
                progress.report(MemorySyncProgressUpdate(completed=progress.completed, total=progress.total))
            return

        chunks = chunk_markdown(
            file_meta.content,
            file_meta.path,
            MemorySource.SESSIONS,
            self.settings.chunk_tokens,
            self.settings.chunk_overlap,
        )

        chunks = [c for c in chunks if c.text.strip()]

        if chunks:
            chunks = await self.store.get_chunk_embeddings(chunks)

        await self.store.upsert_file(file_meta, MemorySource.SESSIONS, chunks)
        self._reset_session_delta(abs_path, file_meta.size)

        if progress.report:
            progress.completed += 1
            progress.report(MemorySyncProgressUpdate(completed=progress.completed, total=progress.total))

    # ============================================================================
    # File Listing and Building
    # ============================================================================

    def _list_memory_files(self) -> list[FileMetadata]:
        """List all memory markdown files."""
        files = []

        # Scan workspace
        memory_paths = [
            os.path.join(self.workspace_dir, "MEMORY.md"),
            os.path.join(self.workspace_dir, "memory.md"),
            os.path.join(self.workspace_dir, "memory"),
        ]

        for base_path in memory_paths:
            if os.path.isfile(base_path) and base_path.endswith(".md"):
                files.append(self._build_file_entry(base_path))
            elif os.path.isdir(base_path):
                for root, _, filenames in os.walk(base_path):
                    for filename in filenames:
                        if filename.endswith(".md"):
                            abs_path = os.path.join(root, filename)
                            files.append(self._build_file_entry(abs_path))

        # Extra paths
        for extra in self.settings.extra_paths:
            if os.path.isfile(extra) and extra.endswith(".md"):
                files.append(self._build_file_entry(extra))
            elif os.path.isdir(extra):
                for root, _, filenames in os.walk(extra):
                    for filename in filenames:
                        if filename.endswith(".md"):
                            abs_path = os.path.join(root, filename)
                            files.append(self._build_file_entry(abs_path))

        return files

    def _list_session_files(self) -> list[str]:
        """List all session transcript files."""
        sessions_dir = os.path.join(self.workspace_dir, "sessions", self.agent_id)
        if not os.path.exists(sessions_dir):
            return []

        files = []
        for filename in os.listdir(sessions_dir):
            if filename.endswith(".jsonl"):
                files.append(os.path.join(sessions_dir, filename))

        return files

    def _build_file_entry(self, abs_path: str) -> FileMetadata:
        """Build file entry metadata."""
        stat = os.stat(abs_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        rel_path = os.path.relpath(abs_path, self.workspace_dir)

        return FileMetadata(
            hash=hash_text(content),
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=rel_path.replace("\\", "/"),
            abs_path=abs_path,
        )

    def _build_session_file_meta(self, abs_path: str) -> FileMetadata | None:
        """Build session file entry with parsed content. TODO 修改message解析逻辑"""
        stat = os.stat(abs_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            raw = f.read()

        lines = raw.split("\n")
        collected = []

        for line in lines:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("type") != "message":
                continue

            message = record.get("message", {})
            role = message.get("role")

            if role not in ("user", "assistant"):
                continue

            text = self._extract_session_text(message.get("content"))
            if not text:
                continue

            label = "User" if role == "user" else "Assistant"
            collected.append(f"{label}: {text}")

        content = "\n".join(collected)
        rel_path = self._session_path_for_file(abs_path)

        return FileMetadata(
            hash=hash_text(content),
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=rel_path,
            abs_path=abs_path,
            content=content,
        )

    # ============================================================================
    # Session Processing Helpers
    # ============================================================================

    @staticmethod
    def _session_path_for_file(abs_path: str) -> str:
        """Convert absolute session path to relative."""
        return f"sessions/{os.path.basename(abs_path)}"

    def _extract_session_text(self, content: Any) -> str | None:
        """Extract text from session message content."""
        if isinstance(content, str):
            normalized = self._normalize_session_text(content)
            return normalized if normalized else None

        if not isinstance(content, list):
            return None

        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") != "text":
                continue

            text = block.get("text")
            if isinstance(text, str):
                normalized = self._normalize_session_text(text)
                if normalized:
                    parts.append(normalized)

        return " ".join(parts) if parts else None

    @staticmethod
    def _normalize_session_text(text: str) -> str:
        """Normalize session text by collapsing whitespace."""
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ============================================================================
    # Session Delta Tracking
    # ============================================================================

    async def _process_session_delta_batch(self) -> None:
        """Process pending session file changes."""
        if not self.session_pending_files:
            return

        pending = list(self.session_pending_files)
        self.session_pending_files.clear()

        should_sync = False
        for session_file in pending:
            delta = await self._update_session_delta(session_file)
            if not delta:
                continue

            bytes_threshold = self.settings.session_delta_bytes
            messages_threshold = self.settings.session_delta_messages

            if bytes_threshold <= 0:
                bytes_hit = delta["pending_bytes"] > 0
            else:
                bytes_hit = delta["pending_bytes"] >= bytes_threshold

            if messages_threshold <= 0:
                messages_hit = delta["pending_messages"] > 0
            else:
                messages_hit = delta["pending_messages"] >= messages_threshold

            if not bytes_hit and not messages_hit:
                continue

            self.sessions_dirty_files.add(session_file)
            self.sessions_dirty = True
            should_sync = True

        if should_sync:
            try:
                await self.sync(reason="session-delta")
            except Exception as err:
                logger.warning(f"memory sync failed (session-delta): {err}")

    async def _update_session_delta(self, session_file: str) -> dict[str, int] | None:
        """Update delta tracking for a session file."""
        try:
            stat = os.stat(session_file)
            size = stat.st_size
        except OSError:
            return None

        state = self.session_deltas.get(session_file)
        if not state:
            state = SessionDelta()
            self.session_deltas[session_file] = state

        delta_bytes = max(0, size - state.last_size)

        if delta_bytes == 0 and size == state.last_size:
            return {
                "delta_bytes": self.settings.session_delta_bytes,
                "delta_messages": self.settings.session_delta_messages,
                "pending_bytes": state.pending_bytes,
                "pending_messages": state.pending_messages,
            }

        if size < state.last_size:
            state.last_size = size
            state.pending_bytes += size
            if self.settings.session_delta_messages > 0:
                state.pending_messages += await self._count_newlines(session_file, 0, size)
        else:
            state.pending_bytes += delta_bytes
            if self.settings.session_delta_messages > 0:
                state.pending_messages += await self._count_newlines(session_file, state.last_size, size)
            state.last_size = size

        return {
            "delta_bytes": self.settings.session_delta_bytes,
            "delta_messages": self.settings.session_delta_messages,
            "pending_bytes": state.pending_bytes,
            "pending_messages": state.pending_messages,
        }

    def _reset_session_delta(self, abs_path: str, size: int) -> None:
        """Reset delta tracking for a session file."""
        state = self.session_deltas.get(abs_path)
        if state:
            state.last_size = size
            state.pending_bytes = 0
            state.pending_messages = 0

    @staticmethod
    async def _count_newlines(abs_path: str, start: int, end: int) -> int:
        """Count newlines in a file range."""
        if end <= start:
            return 0

        count = 0
        with open(abs_path, "rb") as f:
            f.seek(start)
            remaining = end - start

            while remaining > 0:
                chunk_size = min(SESSION_DELTA_READ_CHUNK_BYTES, remaining)
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                count += chunk.count(b"\n")
                remaining -= len(chunk)

        return count

    # ============================================================================
    # File Watchers
    # ============================================================================

    async def _start_watchers(self):
        """Start file watching and interval sync tasks."""
        if self.settings.watch_enabled and MemorySource.MEMORY in self.sources:
            self.watch_task = asyncio.create_task(self._watch_memory_files())

        if MemorySource.SESSIONS in self.sources:
            self.session_watch_task = asyncio.create_task(self._watch_session_files())

        if self.settings.interval_minutes > 0:
            self.interval_task = asyncio.create_task(self._interval_sync())

    async def _watch_memory_files(self) -> None:
        """Watch memory files for changes."""
        watch_paths = [
            os.path.join(self.workspace_dir, "MEMORY.md"),
            os.path.join(self.workspace_dir, "memory.md"),
            os.path.join(self.workspace_dir, "memory"),
        ]

        for extra in self.settings.extra_paths:
            watch_paths.append(extra)

        async for changes in awatch(*watch_paths, stop_event=None):
            if self.closed:
                break

            for _, path in changes:
                if path.endswith(".md"):
                    self.dirty = True
                    await asyncio.sleep(self.settings.watch_debounce_ms / 1000)
                    try:
                        await self.sync(reason="watch")
                    except Exception as e:
                        logger.exception(f"memory sync failed (watch): {e}")

    async def _watch_session_files(self):
        """Watch session files for changes."""
        sessions_dir = os.path.join(self.workspace_dir, "sessions", self.agent_id)
        if not os.path.exists(sessions_dir):
            return

        async for changes in awatch(sessions_dir, stop_event=None):
            if self.closed:
                break

            for _, path in changes:
                if path.endswith(".jsonl"):
                    self.session_pending_files.add(path)

            await asyncio.sleep(SESSION_DIRTY_DEBOUNCE_MS / 1000)
            await self._process_session_delta_batch()

    async def _interval_sync(self) -> None:
        """Periodically sync the index."""
        while not self.closed:
            await asyncio.sleep(self.settings.interval_minutes * 60)
            if not self.closed:
                try:
                    await self.sync(reason="interval")
                except Exception as err:
                    logger.warning(f"memory sync failed (interval): {err}")

    # ============================================================================
    # Search Methods
    # ============================================================================

    async def _search_vector(self, query: str, limit: int) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        return await self.store.vector_search(query, limit, sources=list(self.sources))

    async def _search_keyword(self, query: str, limit: int) -> list[MemorySearchResult]:
        """Perform keyword/FTS search."""
        if not self.settings.fts_enabled:
            return []

        return await self.store.keyword_search(query, limit, sources=list(self.sources))

    @staticmethod
    def _merge_hybrid_results(
        vector: list[MemorySearchResult],
        keyword: list[MemorySearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[MemorySearchResult]:
        """Merge vector and keyword search results."""
        merged: dict[str, MemorySearchResult] = {}

        # Process vector results
        for result in vector:
            result.score = result.score * vector_weight
            merged[result.merge_key] = result

        # Process keyword results
        for result in keyword:
            key = result.merge_key
            if key in merged:
                merged[key].score += result.score * text_weight
            else:
                result.score = result.score * text_weight
                merged[key] = result

        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ============================================================================
    # Utility Methods
    # ============================================================================

    @staticmethod
    def _is_memory_path(rel_path: str) -> bool:
        """Check if path is a valid memory path."""
        normalized = rel_path.replace("\\", "/")

        if normalized in ("MEMORY.md", "memory.md"):
            return True

        if normalized.startswith("memory/") and normalized.endswith(".md"):
            return True

        return False
