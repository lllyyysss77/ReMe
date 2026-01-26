"""Add memory to vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode


class AddMemory(BaseMemoryTool):
    """Tool to add memories to vector store"""

    def _build_tool_call(self) -> ToolCall:
        """Build and return the single tool call schema"""
        return ToolCall(
            **{
                "description": "add a memory to vector store for future retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "conversation_time": {
                            "type": "string",
                            "description": "conversation time, e.g. '2020-01-01 00:00:00'",
                        },
                        "memory_content": {
                            "type": "string",
                            "description": "content of the memory.",
                        },
                    },
                    "required": ["conversation_time", "memory_content"],
                },
            },
        )

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
                                    "conversation_time": {
                                        "type": "string",
                                        "description": "conversation time, e.g. '2020-01-01 00:00:00'",
                                    },
                                    "memory_content": {
                                        "type": "string",
                                        "description": "content of the memory.",
                                    },
                                },
                                "required": ["conversation_time", "memory_content"],
                            },
                        },
                    },
                    "required": ["memories"],
                },
            },
        )

    def _create_memory_node(self, data: dict) -> MemoryNode:
        """Create a MemoryNode from a dictionary."""
        memory_content = data.get("memory_content", "")
        conversation_time = data.get("conversation_time", "")
        metadata: dict = {"conversation_time": conversation_time}

        try:
            metadata["time_int"] = int(conversation_time.split(" ")[0].replace("-", ""))
        except Exception:
            logger.warning(f"Invalid conversation time format. {conversation_time}")

        return MemoryNode(
            memory_type=self.memory_type,
            memory_target=self.memory_target,
            content=memory_content,
            author=self.author,
            ref_memory_id=self.history_node.memory_id,
            metadata=metadata,
        )

    async def execute(self):
        memory_nodes: list[MemoryNode] = []
        memories: list[dict] = self.context.get("memories", [])

        if not memories:
            memory_nodes.append(self._create_memory_node(self.context))
        else:
            for mem in memories:
                memory_nodes.append(self._create_memory_node(mem))

        if not memory_nodes:
            output = "No valid memories provided for addition."
            logger.info(output)
            return output

        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        await self.vector_store.delete(vector_ids=list(set(vector_ids)))
        await self.vector_store.insert(nodes=vector_nodes)
        self.memory_nodes.extend(memory_nodes)

        output = f"Successfully added {len(memory_nodes)} memories to vector_store."
        logger.info(output)
        return output
