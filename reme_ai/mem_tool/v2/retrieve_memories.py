"""Retrieve memories using vector similarity search with multiple queries."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.context import C
from ...core_old.schema import MemoryNode, VectorNode
from ...core_old.utils import deduplicate_memories


@C.register_op()
class RetrieveMemories(BaseMemoryTool):
    """Retrieve memories using vector similarity search with multiple queries.

    Always requires memory_type/memory_target in the schema.
    Only supports multiple query mode (enable_multiple=True).
    Metadata filters can be customized via `metadata_desc` parameter for pre-retrieval filtering.
    """

    def __init__(self, metadata_desc: dict[str, str] | None = None, top_k: int = 20, **kwargs):
        """Initialize RetrieveMemories.

        Args:
            metadata_desc: Dictionary defining metadata filter fields and their descriptions.
                These fields will be used as filters in vector search before similarity matching.
            top_k: Max memories to retrieve per query.
            **kwargs: Additional args for BaseMemoryTool.
        """
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)
        self.metadata_desc: dict[str, str] = metadata_desc or {}
        self.top_k: int = top_k

    def _build_query_schema(self) -> tuple[dict, list[str]]:
        """Build schema properties and required fields for query items.

        Returns:
            Tuple of (properties dict, required fields list).
        """
        properties = {
            "memory_type": {
                "type": "string",
                "description": self.get_prompt("memory_type"),
            },
            "memory_target": {
                "type": "string",
                "description": self.get_prompt("memory_target"),
            },
            "query": {
                "type": "string",
                "description": self.get_prompt("query"),
            },
        }
        required = ["memory_type", "memory_target", "query"]

        # Add metadata filter fields if metadata_desc is provided and not empty
        if self.metadata_desc:
            metadata_properties = {
                key: {"type": "string", "description": desc} for key, desc in self.metadata_desc.items()
            }
            # Generate dynamic description based on metadata_desc fields
            field_descriptions = "\n".join([f"  - {key}: {desc}" for key, desc in self.metadata_desc.items()])
            metadata_description = (
                f"Optional metadata filters for narrowing search results. Available fields:\n{field_descriptions}"
            )

            properties["metadata_filters"] = {
                "type": "object",
                "description": metadata_description,
                "properties": metadata_properties,
            }

        return properties, required

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple query mode.

        Returns:
            Schema with query_items array. Each item has memory_type/memory_target/query.
        """
        item_properties, item_required = self._build_query_schema()
        return {
            "type": "object",
            "properties": {
                "query_items": {
                    "type": "array",
                    "description": self.get_prompt("query_items"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": item_required,
                    },
                },
            },
            "required": ["query_items"],
        }

    async def _retrieve_by_query(
            self,
            memory_type: str,
            memory_target: str,
            query: str,
            metadata_filters: dict | None = None,
    ) -> list[MemoryNode]:
        """Retrieve memories by query using vector similarity search.

        Args:
            memory_type: Memory type to search.
            memory_target: Memory target to search.
            query: Query string for similarity search.
            metadata_filters: Optional metadata filters to narrow search results.

        Returns:
            List of matching memories.
        """
        filter_dict = {
            "memory_type": [memory_type],
            "memory_target": [memory_target],
        }

        # Add metadata filters if provided
        if metadata_filters:
            for key, value in metadata_filters.items():
                if value:  # Only add non-empty filter values
                    value = str(value).strip()
                    filter_dict[key] = [value] if not isinstance(value, list) else value

        nodes: list[VectorNode] = await self.vector_store.search(query=query, limit=self.top_k, filters=filter_dict)
        memory_nodes: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]
        return memory_nodes

    async def execute(self):
        """Execute memory retrieval based on multiple query items.

        Outputs formatted results or error message.
        """
        query_items: list[dict] = self.context.get("query_items", [])
        if not query_items:
            self.output = "No query items provided for retrieval."
            return

        # Filter out items without query text
        query_items = [item for item in query_items if item.get("query")]

        if not query_items:
            self.output = "No valid query texts provided for retrieval."
            return

        # Retrieve memory_nodes for all queries
        memory_nodes: list[MemoryNode] = []
        for item in query_items:
            memory_type = item.get("memory_type")
            memory_target = item.get("memory_target")
            metadata_filters = item.get("metadata_filters", {}) if self.metadata_desc else {}

            if not memory_type or not memory_target:
                logger.warning(f"Skipping query with missing memory_type or memory_target: {item}")
                continue

            retrieved = await self._retrieve_by_query(
                memory_type=memory_type,
                memory_target=memory_target,
                query=item["query"],
                metadata_filters=metadata_filters,
            )
            memory_nodes.extend(retrieved)

        # Deduplicate and format output
        memory_nodes = deduplicate_memories(memory_nodes)

        # Build set of historical memory_ids for fast lookup
        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}

        # Filter out already retrieved memories by memory_id
        new_memory_nodes = [node for node in memory_nodes if node.memory_id not in retrieved_memory_ids]

        # Update retrieved_nodes in context with new memories
        self.retrieved_nodes.extend(new_memory_nodes)

        # Set output to new memories only (after deduplication)
        self.memory_nodes = new_memory_nodes

        if not new_memory_nodes:
            self.output = "No new memories found matching the queries (duplicates removed)."
        else:
            self.output = "\n".join([m.format_memory() for m in new_memory_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memories, {len(new_memory_nodes)} new after deduplication")
