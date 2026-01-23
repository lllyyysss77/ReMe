"""Retrieve memory from vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall, MemoryNode, VectorNode
from ....core.utils import deduplicate_memories


class VectorRetrieveMemory(BaseMemoryTool):
    """Tool to retrieve memories from vector store using similarity search"""

    def __init__(
        self,
        add_memory_type_target: bool = False,
        top_k: int = 20,
        enable_metadata: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_memory_type_target: bool = add_memory_type_target
        self.top_k: int = top_k
        self.enable_metadata: bool = enable_metadata

    def _build_query_schema(self) -> tuple[dict, list[str]]:
        """Build query schema for single/multiple retrieval"""
        properties = {}
        required = []

        if self.add_memory_type_target:
            properties["memory_type"] = {
                "type": "string",
                "description": "type of memory to search for.",
            }
            properties["memory_target"] = {
                "type": "string",
                "description": "target of memory to search within.",
            }
            required.extend(["memory_type", "memory_target"])

        properties["query"] = {
            "type": "string",
            "description": "query text for vector similarity search.",
        }
        required.append("query")

        if self.enable_metadata:
            properties["metadata"] = {
                "type": "object",
                "description": "optional metadata filters for narrowing search results.",
            }

        return properties, required

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        properties, required = self._build_query_schema()
        return ToolCall(
            **{
                "description": "retrieve memories using vector similarity search.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        item_properties, item_required = self._build_query_schema()
        return ToolCall(
            **{
                "description": "retrieve memories using multiple queries with vector similarity search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_items": {
                            "type": "array",
                            "description": "list of query items for vector similarity search.",
                            "items": {
                                "type": "object",
                                "properties": item_properties,
                                "required": item_required,
                            },
                        },
                    },
                    "required": ["query_items"],
                },
            },
        )

    async def _retrieve_by_query(
        self,
        memory_type: str,
        memory_target: str,
        query: str,
        metadata: dict | None = None,
    ) -> list[MemoryNode]:
        """Retrieve memories by query with filters"""
        filter_dict = {
            "memory_type": [memory_type],
            "memory_target": [memory_target],
        }

        if metadata:
            for key, value in metadata.items():
                if value:
                    value = str(value).strip()
                    filter_dict[key] = [value] if not isinstance(value, list) else value

        nodes: list[VectorNode] = await self.vector_store.search(query=query, limit=self.top_k, filters=filter_dict)

        memory_nodes: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]

        filtered_memory_nodes = [
            m for m in memory_nodes if not (m.memory_type == MemoryType.TOOL and m.when_to_use != query)
        ]

        return filtered_memory_nodes

    async def execute(self):
        default_memory_type: str = self.context.get("memory_type", "")
        default_memory_target: str = self.context.get("memory_target", "")

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

        query_items = [item for item in query_items if item.get("query")]

        if not query_items:
            self.output = "No valid query texts provided for retrieval."
            return

        memory_nodes: list[MemoryNode] = []
        for item in query_items:
            memory_type = item.get("memory_type") or default_memory_type
            memory_target = item.get("memory_target") or default_memory_target
            metadata = item.get("metadata", {}) if self.enable_metadata else None

            if not memory_type or not memory_target:
                logger.warning(f"Skipping query with missing memory_type or memory_target: {item}")
                continue

            retrieved = await self._retrieve_by_query(
                memory_type=memory_type,
                memory_target=memory_target,
                query=item["query"],
                metadata=metadata,
            )
            memory_nodes.extend(retrieved)

        memory_nodes = deduplicate_memories(memory_nodes)

        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}

        new_memory_nodes = [node for node in memory_nodes if node.memory_id not in retrieved_memory_ids]

        self.retrieved_nodes.extend(new_memory_nodes)

        self.memory_nodes = new_memory_nodes

        if not new_memory_nodes:
            self.output = "No new memory_nodes found matching the query (duplicates removed)."
        else:
            self.output = "\n".join([m.format_memory() for m in new_memory_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memory_nodes, {len(new_memory_nodes)} new after deduplication")
