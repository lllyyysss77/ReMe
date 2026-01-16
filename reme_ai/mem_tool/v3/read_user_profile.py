from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.schema.memory_node import MemoryNode


class ReadUserProfile(BaseMemoryTool):

    def __init__(self, add_memory_type_target: bool = True, **kwargs):
        kwargs["enable_multiple"] = False
        self.add_memory_type_target = add_memory_type_target
        super().__init__(**kwargs)

    def _build_tool_description(self) -> str:
        return "Read personal memory profile for the current user."

    def _build_parameters(self) -> dict:
        if self.add_memory_type_target:
            return {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "memory_type",
                    },
                    "memory_target": {
                        "type": "string",
                        "description": "memory_target",
                    },
                },
                "required": ["memory_type", "memory_target"],
            }
        else:
            return {
                "type": "object",
                "properties": {},
                "required": [],
            }

    async def execute(self):
        cache_key = f"{self.memory_type}_{self.memory_target}"
        cached_data = self.meta_memory.load(cache_key, auto_clean=False)

        if not cached_data:
            self.output = f"Local memory not found: {self.memory_type}_{self.memory_target}"
            logger.info(self.output)
            return

        # Convert to MemoryNode objects and sort by conversation_time (oldest first)
        memory_nodes = [MemoryNode(**node_data) for node_data in cached_data]
        memory_nodes.sort(
            key=lambda node: node.metadata.get("conversation_time", "")
        )

        memory_formated = []
        for node in memory_nodes:
            node_formated = f"profile_id={node.memory_id} profile_content={node.content}"
            if "conversation_time" in node.metadata:
                node_formated += f" conversation_time={node.metadata['conversation_time']}"
            if node.ref_memory_id:
                node_formated += f" history_id={node.ref_memory_id}"
            memory_formated.append(node_formated.strip())

        self.output = "\n".join(memory_formated)
        logger.info(f"Read {len(memory_formated)} nodes from cache key: {cache_key}")
