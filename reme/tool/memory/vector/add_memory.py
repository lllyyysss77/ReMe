"""Add memory to vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode


class AddMemory(BaseMemoryTool):
    """Tool to add memories to vector store"""

    def __init__(self, **kwargs):
        kwargs['enable_multiple'] = True
        super().__init__(**kwargs)

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "add multiple memories to vector store for future retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "description": "list of memories to store.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memory_content": {
                                        "type": "string",
                                        "description": "content of the memory.",
                                    },
                                    "metadata": {
                                        "type": "object",
                                        "description": "metadata for the memory.",
                                    }
                                },
                                "required": ["memory_content"],
                            },
                        },
                    },
                    "required": ["memories"],
                },
            },
        )

    def _extract_memory_data(self, mem_dict: dict) -> tuple[str, dict]:
        """Extract memory content and metadata from dict"""
        memory_content = mem_dict.get("memory_content", "")
        raw_metadata = mem_dict.get("metadata", {})
        metadata = {key: str(value).strip() for key, value in raw_metadata.items() if value}
        return memory_content, metadata

    def _build_memory_node(self, content: str, metadata: dict = None) -> MemoryNode:
        """Build a memory node"""
        return MemoryNode(
            memory_type=self.memory_type,
            memory_target=self.memory_target,
            content=content,
            author=self.author,
            metadata=metadata or {},
        )

    async def execute(self):
        memory_nodes: list[MemoryNode] = []

        memories: list[dict] = self.context.get("memories", [])
        if not memories:
            self.output = "No memories provided for addition."
            return

        for mem in memories:
            memory_content, metadata = self._extract_memory_data(mem)
            if not memory_content:
                logger.warning("Skipping memory with empty content")
                continue

            memory_nodes.append(self._build_memory_node(memory_content, metadata=metadata))

        if not memory_nodes:
            self.output = "No valid memories provided for addition."
            return

        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        await self.vector_store.delete(vector_ids=vector_ids)
        await self.vector_store.insert(nodes=vector_nodes)
        self.memory_nodes = memory_nodes

        self.output = f"Successfully added {len(memory_nodes)} memories to vector_store."
        logger.info(self.output)
