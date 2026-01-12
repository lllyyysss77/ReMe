"""Combined memory retrieval: recent + vector similarity search."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema import MemoryNode, VectorNode
from ...core.utils import deduplicate_memories


@C.register_op()
class RetrieveRecentAndSimilarMemories(BaseMemoryTool):
    """Retrieve memories using both time-based and vector similarity search.

    First retrieves recent_top_k memories sorted by modification time,
    then retrieves similar_top_k memories using vector similarity search.
    Uses memory_type and memory_target from context (self.memory_type, self.memory_target).
    """

    def __init__(
            self,
            recent_top_k: int = 20,
            similar_top_k: int = 20,
            **kwargs,
    ):
        """Initialize RetrieveRecentAndSimilarMemories.

        Args:
            recent_top_k: Max recent memories to retrieve by time.
            similar_top_k: Max similar memories to retrieve by vector search.
            **kwargs: Additional args for BaseMemoryTool.
        """
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)
        self.recent_top_k: int = recent_top_k
        self.similar_top_k: int = similar_top_k

    def _build_tool_description(self) -> str:
        """Build tool description."""
        return self.prompt_format("tool_multiple",
                                  recent_top_k=self.recent_top_k,
                                  similar_top_k=self.similar_top_k)

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple query mode.

        Returns:
            Schema with query_items array.
        """
        return {
            "type": "object",
            "properties": {
                "query_items": {
                    "type": "array",
                    "description": self.get_prompt("query_items"),
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": self.get_prompt("query"),
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            "required": ["query_items"],
        }

    async def _retrieve_recent(self) -> list[MemoryNode]:
        """Retrieve recent memories sorted by time_modified.

        Returns:
            List of recent memories sorted by modification time (newest first).
        """
        filter_dict = {
            "memory_type": [self.memory_type.value],
            "memory_target": [self.memory_target],
        }

        # Use list() with sort_key="time_modified", reverse=True (descending), and limit
        nodes: list[VectorNode] = await self.vector_store.list(
            filters=filter_dict,
            limit=self.recent_top_k,
            sort_key="time_modified",
            reverse=True,  # Most recent first (descending order)
        )

        memory_nodes: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]

        return memory_nodes

    async def _retrieve_by_query(
            self,
            query: str,
    ) -> list[MemoryNode]:
        """Retrieve memories by query using vector similarity search.

        Args:
            query: Query string for similarity search.

        Returns:
            List of matching memories.
        """
        filter_dict = {
            "memory_type": [self.memory_type.value],
            "memory_target": [self.memory_target],
        }

        nodes: list[VectorNode] = await self.vector_store.search(
            query=query, limit=self.similar_top_k, filters=filter_dict
        )

        memory_nodes: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]

        return memory_nodes

    async def execute(self):
        """Execute combined memory retrieval (recent + similar).

        First retrieves recent_top_k memories by time, then retrieves similar_top_k
        memories by vector similarity for each query in query_items.
        Uses memory_type and memory_target from context. Outputs formatted results or error message.
        """
        if not self.memory_type or not self.memory_target:
            raise RuntimeError("memory_type and memory_target are required for retrieval.")

        # Get query items
        query_items: list[dict] = self.context.get("query_items", [])
        if not query_items:
            self.output = "No query items provided for retrieval."
            return

        # Filter out items without query text
        query_items = [item for item in query_items if item.get("query")]

        if not query_items:
            self.output = "No valid query texts provided for retrieval."
            return

        # Step 1: Retrieve recent memories (once, shared across all queries)
        recent_memory_nodes: list[MemoryNode] = await self._retrieve_recent()
        logger.info(f"Retrieved {len(recent_memory_nodes)} recent memories")

        # Step 2: Retrieve similar memories by vector search for all queries
        similar_memory_nodes: list[MemoryNode] = []
        for item in query_items:
            retrieved = await self._retrieve_by_query(query=item["query"])
            similar_memory_nodes.extend(retrieved)
        # Combine and deduplicate all memories
        all_memory_nodes = recent_memory_nodes + similar_memory_nodes
        all_memory_nodes = deduplicate_memories(all_memory_nodes)

        # Build set of historical memory_ids for fast lookup
        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}

        # Filter out already retrieved memories by memory_id
        new_memory_nodes = [node for node in all_memory_nodes if node.memory_id not in retrieved_memory_ids]

        # Update retrieved_nodes in context with new memories
        self.retrieved_nodes.extend(new_memory_nodes)

        # Set output to new memories only (after deduplication)
        self.memory_nodes = new_memory_nodes

        if not new_memory_nodes:
            self.output = "No new memory_nodes found (duplicates removed)."
        else:
            self.output = "\n".join([m.format_memory() for m in new_memory_nodes])

        logger.info(
            f"Retrieved {len(all_memory_nodes)} total memories "
            f"({len(recent_memory_nodes)} recent + {len(similar_memory_nodes)} similar), "
            f"{len(new_memory_nodes)} new after deduplication"
        )
