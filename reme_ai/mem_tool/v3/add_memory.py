from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.schema import MemoryNode


class AddMemory(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs['enable_multiple'] = True
        super().__init__(**kwargs)

    def _build_tool_description(self) -> str:
        return "Add multiple memories to the vector store for future retrieval."

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "description": "A list of memory objects to store.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_content": {
                                "type": "string",
                                "description": "memory content",
                            },
                            "conversation_time": {
                                "type": "string",
                                "description": "conversation time, e.g. '2020-01-01 00:00:00'",
                            }
                        },
                        "required": ["memory_content", "conversation_time"],
                    },
                },
            },
            "required": ["memories"],
        }

    async def execute(self):
        memories: list[dict] = self.context.get("memories", [])
        if not memories:
            self.output = "No memories provided for addition."
            return

        memory_nodes: list[MemoryNode] = []
        for mem in memories:
            memory_content = mem.get("memory_content", "")
            conversation_time = mem.get("conversation_time", "")
            metadata: dict = {"conversation_time": conversation_time}
            try:
                metadata["time_int"] = int(conversation_time.split(" ")[0].replace("-", ""))
            except Exception:
                ...
            memory_nodes.append(self._build_memory_node(memory_content, metadata=metadata))

        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        await self.vector_store.delete(vector_ids=vector_ids)
        await self.vector_store.insert(nodes=vector_nodes)
        self.memory_nodes = memory_nodes

        self.output = f"Successfully added {len(memory_nodes)} memories to vector_store."
        logger.info(self.output)
