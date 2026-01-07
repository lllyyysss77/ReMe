"""Vector-based memory retrieval using semantic similarity search."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.enumeration import MemoryType
from ...core.schema import MemoryNode, VectorNode
from ...core.utils import deduplicate_memories


@C.register_op()
class VectorRetrieveMemory(BaseMemoryTool):
    """Retrieve memories using vector similarity search.

    Supports single/multiple query modes via `enable_multiple` parameter.
    When `add_memory_type_target` is False, memory_type/memory_target are from context.
    """

    def __init__(
        self,
        enable_summary_memory: bool = False,
        add_memory_type_target: bool = False,
        top_k: int = 10,
        **kwargs,
    ):
        """Initialize VectorRetrieveMemory.

        Args:
            enable_summary_memory: Include summary memories in results.
            add_memory_type_target: Include memory_type/memory_target in schema (else from context).
            top_k: Max memories to retrieve per query.
            **kwargs: Additional args for BaseMemoryTool.
        """
        super().__init__(**kwargs)
        self.enable_summary_memory: bool = enable_summary_memory
        self.add_memory_type_target: bool = add_memory_type_target
        self.top_k: int = top_k

    def _build_query_schema(self) -> tuple[dict, list[str]]:
        """Build shared schema properties and required fields for query items.

        Returns:
            Tuple of (properties dict, required fields list).
        """
        properties = {}
        required = []

        if self.add_memory_type_target:
            properties["memory_type"] = {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [
                    MemoryType.IDENTITY.value,
                    MemoryType.PERSONAL.value,
                    MemoryType.PROCEDURAL.value,
                    MemoryType.TOOL.value,
                ],
            }
            properties["memory_target"] = {
                "type": "string",
                "description": self.get_prompt("memory_target"),
            }
            required.extend(["memory_type", "memory_target"])

        properties["query"] = {
            "type": "string",
            "description": self.get_prompt("query"),
        }
        required.append("query")

        return properties, required

    def _build_parameters(self) -> dict:
        """Build input schema for single query mode.

        Returns:
            Schema with memory_type/memory_target/query (if add_memory_type_target) or query only.
        """
        properties, required = self._build_query_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple query mode.

        Returns:
            Schema with query_items array. Each item has memory_type/memory_target/query
            (if add_memory_type_target) or query only.
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
    ) -> list[MemoryNode]:
        """Retrieve memories by query using vector similarity search.

        Args:
            memory_type: Memory type to search.
            memory_target: Memory target to search.
            query: Query string for similarity search.

        Returns:
            List of matching memories.
        """
        memory_type_list = [MemoryType(memory_type)]
        if self.enable_summary_memory:
            memory_type_list.append(MemoryType.SUMMARY)

        filter_dict = {
            "memory_type": [mt.value for mt in memory_type_list],
            "memory_target": [memory_target],
        }

        nodes: list[VectorNode] = await self.vector_store.search(
            query=query,
            top_k=self.top_k,
            filter_dict=filter_dict,
        )

        memory_nodes: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]

        # Filter TOOL memories: keep only if when_to_use matches query (tool name)
        filtered_memory_nodes = [
            m for m in memory_nodes if not (m.memory_type == MemoryType.TOOL and m.when_to_use != query)
        ]

        return filtered_memory_nodes

    async def execute(self):
        """Execute memory retrieval based on query texts.

        Handles single/multiple query modes. When add_memory_type_target is False,
        memory_type/memory_target are from context. Outputs formatted results or error message.
        """
        default_memory_type: str = self.context.get("memory_type", "")
        default_memory_target: str = self.context.get("memory_target", "")

        # Normalize to list of query items
        if self.enable_multiple:
            query_items: list[dict] = self.context.get("query_items", [])
            if not query_items:
                self.output = "No query items provided for retrieval."
                return
        else:
            query = self.context.get("query", "")
            if not query:
                self.output = "No query provided for retrieval."
                return

            query_items = [
                {
                    "memory_type": default_memory_type,
                    "memory_target": default_memory_target,
                    "query": query,
                },
            ]

        # Filter out items without query text
        query_items = [item for item in query_items if item.get("query")]

        if not query_items:
            self.output = "No valid query texts provided for retrieval."
            return

        # Retrieve memories for all queries
        memories: list[MemoryNode] = []
        for item in query_items:
            memory_type = item.get("memory_type") or default_memory_type
            memory_target = item.get("memory_target") or default_memory_target

            if not memory_type or not memory_target:
                logger.warning(f"Skipping query with missing memory_type or memory_target: {item}")
                continue

            retrieved = await self._retrieve_by_query(
                memory_type=memory_type,
                memory_target=memory_target,
                query=item["query"],
            )
            memories.extend(retrieved)

        # Deduplicate and format output
        memories = deduplicate_memories(memories)

        if not memories:
            self.output = "No memories found matching the query."
        else:
            self.output = "\n".join([m.format_memory() for m in memories])

        logger.info(f"Retrieved {len(memories)} memories")
