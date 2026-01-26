"""Update memory in vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode


class UpdateMemory(BaseMemoryTool):
    """Tool to update memories in vector store"""

    def _build_tool_call(self) -> ToolCall:
        """Build and return the single tool call schema"""
        return ToolCall(
            **{
                "description": "update a memory in vector store by replacing old memory with new content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "unique identifier of memory to update.",
                        },
                        "conversation_time": {
                            "type": "string",
                            "description": "conversation time, e.g. '2020-01-01 00:00:00'",
                        },
                        "memory_content": {
                            "type": "string",
                            "description": "new content of the memory.",
                        },
                    },
                    "required": ["memory_id", "conversation_time", "memory_content"],
                },
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "update multiple memories in vector store by replacing old memories with new content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "description": "list of memory update objects.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memory_id": {
                                        "type": "string",
                                        "description": "unique identifier of memory to update.",
                                    },
                                    "conversation_time": {
                                        "type": "string",
                                        "description": "conversation time, e.g. '2020-01-01 00:00:00'",
                                    },
                                    "memory_content": {
                                        "type": "string",
                                        "description": "new content of the memory.",
                                    },
                                },
                                "required": ["memory_id", "conversation_time", "memory_content"],
                            },
                        },
                    },
                    "required": ["memories"],
                },
            },
        )

    def _create_memory_node(self, data: dict) -> tuple[str, MemoryNode]:
        """Create a MemoryNode from a dictionary."""
        memory_id = data.get("memory_id", "")
        memory_content = data.get("memory_content", "")
        conversation_time = data.get("conversation_time", "")
        metadata: dict = {"conversation_time": conversation_time}

        try:
            metadata["time_int"] = int(conversation_time.split(" ")[0].replace("-", ""))
        except Exception:
            logger.warning(f"Invalid conversation time format. {conversation_time}")

        memory_node = MemoryNode(
            memory_type=self.memory_type,
            memory_target=self.memory_target,
            content=memory_content,
            author=self.author,
            metadata=metadata,
        )

        return memory_id, memory_node

    async def execute(self):
        old_memory_ids: list[str] = []
        memory_nodes: list[MemoryNode] = []
        memories: list[dict] = self.context.get("memories", [])

        if not memories:
            old_id, node = self._create_memory_node(self.context)
            old_memory_ids.append(old_id)
            memory_nodes.append(node)
        else:
            for mem in memories:
                old_id, node = self._create_memory_node(mem)
                old_memory_ids.append(old_id)
                memory_nodes.append(node)

        if not memory_nodes:
            output = "No valid memories provided for update."
            logger.info(output)
            return output

        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        new_vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        all_ids_to_delete = list(set(old_memory_ids + new_vector_ids))
        await self.vector_store.delete(vector_ids=all_ids_to_delete)
        await self.vector_store.insert(nodes=vector_nodes)
        self.memory_nodes = memory_nodes

        output = f"Successfully updated {len(memory_nodes)} memories in vector_store."
        logger.info(output)
        return output
