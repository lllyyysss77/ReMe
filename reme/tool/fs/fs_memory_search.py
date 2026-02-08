"""Memory search tool for semantic search in memory files."""

import json

from loguru import logger

from reme.core.enumeration import MemorySource
from reme.core.schema import MemorySearchResult, ToolCall
from .base_fs_tool import BaseFsTool


class FsMemorySearch(BaseFsTool):
    """Semantically search MEMORY.md and memory files."""

    def __init__(
        self,
        sources: list[MemorySource] | None = None,
        min_score: float = 0.1,
        max_results: int = 20,
        hybrid_enabled: bool = True,
        hybrid_vector_weight: float = 0.7,
        hybrid_text_weight: float = 0.3,
        hybrid_candidate_multiplier: float = 3.0,
        **kwargs,
    ):
        """Initialize memory search tool."""
        kwargs.setdefault("name", "memory_search")
        super().__init__(**kwargs)
        self.sources = sources or [MemorySource.MEMORY]
        self.min_score = min_score
        self.max_results = max_results
        self.hybrid_enabled = hybrid_enabled
        self.hybrid_vector_weight = hybrid_vector_weight
        self.hybrid_text_weight = hybrid_text_weight
        self.hybrid_candidate_multiplier = hybrid_candidate_multiplier

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": (
                    "Mandatory recall step: semantically search MEMORY.md + memory/*.md "
                    "(and optional session transcripts) before answering questions about "
                    "prior work, decisions, dates, people, preferences, or todos; returns "
                    "top snippets with path + lines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The semantic search query to find relevant memory snippets",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return (optional)",
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum similarity score threshold for results (optional)",
                        },
                    },
                    "required": ["query"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the memory search operation."""
        query: str = self.context.query.strip()
        min_score = self.context.get("min_score", self.min_score)
        max_results = self.context.get("max_results", self.max_results)
        candidates = min(200, max(1, int(max_results * self.hybrid_candidate_multiplier)))

        # Perform hybrid search (vector + keyword)
        if self.hybrid_enabled:
            keyword_results = []
            if self.memory_store.fts_enabled:
                keyword_results = await self._search_keyword(query, candidates)
            vector_results = await self._search_vector(query, candidates)

            # Log original vector results
            logger.debug("\n=== Vector Search Results ===")
            for i, r in enumerate(vector_results[:10], 1):
                snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                logger.debug(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

            # Log original keyword results
            logger.debug("\n=== Keyword Search Results ===")
            for i, r in enumerate(keyword_results[:10], 1):
                snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                logger.debug(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

            if not keyword_results:
                results = [r for r in vector_results if r.score >= min_score][:max_results]
            elif not vector_results:
                results = [r for r in keyword_results if r.score >= min_score][:max_results]
            else:
                merged = self._merge_hybrid_results(
                    vector=vector_results,
                    keyword=keyword_results,
                    vector_weight=self.hybrid_vector_weight,
                    text_weight=self.hybrid_text_weight,
                )

                # Log merged results
                logger.debug("\n=== Merged Hybrid Results ===")
                for i, r in enumerate(merged[:10], 1):
                    snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                    logger.debug(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

                results = [r for r in merged if r.score >= min_score][:max_results]
        else:
            vector_results = await self._search_vector(query, candidates)
            results = [r for r in vector_results if r.score >= min_score][:max_results]

        return json.dumps([result.model_dump(exclude_none=True) for result in results], indent=2, ensure_ascii=False)

    async def _search_vector(self, query: str, limit: int) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        return await self.memory_store.vector_search(query, limit, sources=self.sources)

    async def _search_keyword(self, query: str, limit: int) -> list[MemorySearchResult]:
        """Perform keyword/FTS search."""
        if not self.memory_store.fts_enabled:
            return []
        return await self.memory_store.keyword_search(query, limit, sources=self.sources)

    @staticmethod
    def _merge_hybrid_results(
        vector: list[MemorySearchResult],
        keyword: list[MemorySearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[MemorySearchResult]:
        """Merge vector and keyword search results with weighted scoring."""
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

        # Sort by score and return
        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results
