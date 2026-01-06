"""Add history memory operation."""

from loguru import logger

from ...base_memory_tool import BaseMemoryTool
from ....core.context import C
from ....core.schema import MemoryNode


@C.register_op()
class AddHistoryMemory(BaseMemoryTool):
    """Add history memory from conversation messages."""

    def __init__(self, add_metadata: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.add_metadata: bool = add_metadata

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        properties = {
            "messages": {
                "type": "array",
                "description": self.get_prompt("messages"),
                "items": {"type": "object"},
            },
        }
        required = ["messages"]

        if self.add_metadata:
            properties["metadata"] = {
                "type": "object",
                "description": self.get_prompt("metadata"),
            }

        return properties, required

    def _build_parameters(self) -> dict:
        properties, required = self._build_item_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_parameters(self) -> dict:
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "histories": {
                    "type": "array",
                    "description": self.get_prompt("histories"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["histories"],
        }

    def _format_messages(self, messages: list) -> str:
        return "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages])

    def _extract_history_data(self, hist_dict: dict) -> tuple[list, dict]:
        messages = hist_dict.get("messages", [])
        metadata = hist_dict.get("metadata", {}) if self.add_metadata else {}
        return messages, metadata

    async def execute(self):
        memory_nodes: list[MemoryNode] = []

        if self.enable_multiple:
            histories: list[dict] = self.context.get("histories", [])
            if not histories:
                self.output = "No histories provided for addition."
                return

            for hist in histories:
                messages, metadata = self._extract_history_data(hist)
                if not messages:
                    logger.warning("Skipping history with empty messages")
                    continue

                memory_content = self._format_messages(messages)
                memory_nodes.append(
                    self._build_memory_node(memory_content, when_to_use="", metadata=metadata),
                )
        else:
            messages, metadata = self._extract_history_data(self.context)
            if not messages:
                self.output = "No messages provided for addition."
                return

            memory_content = self._format_messages(messages)
            memory_nodes.append(
                self._build_memory_node(memory_content, when_to_use="", metadata=metadata),
            )

        if not memory_nodes:
            self.output = "No valid histories provided for addition."
            return

        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        await self.vector_store.delete(vector_ids=vector_ids)
        await self.vector_store.insert(nodes=vector_nodes)

        self.output = f"Successfully added {len(memory_nodes)} history memories to vector_store."
        logger.info(self.output)
