from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema.memory_node import MemoryNode


@C.register_op()
class UpdateUserProfile(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "profile_ids_to_delete": {
                    "type": "array",
                    "description": self.get_prompt("profile_ids_to_delete"),
                    "items": {"type": "string"},
                },
                "profiles_to_add": {
                    "type": "array",
                    "description": self.get_prompt("profiles_to_add"),
                    "items": {
                        "type": "object",
                        "properties": {
                            "profile_content": {
                                "type": "string",
                                "description": self.get_prompt("profile_content"),
                            },
                            "timestamp": {
                                "type": "string",
                                "description": self.get_prompt("timestamp"),
                            },
                        },
                        "required": ["profile_content", "timestamp"],
                    },
                },
            },
            "required": ["profile_ids_to_delete", "profiles_to_add"],
        }

    async def execute(self):
        memory_type = "personal"
        memory_target = self.memory_target
        assert memory_target, "memory_target is not configured."

        cache_key = f"{memory_type}_{memory_target}"

        profile_ids_to_delete = self.context.get("profile_ids_to_delete", [])
        profile_ids_to_delete = [m for m in profile_ids_to_delete if m]
        profile_ids_to_delete = list(dict.fromkeys(profile_ids_to_delete))

        profiles_to_add = self.context.get("profiles_to_add", [])

        if not profile_ids_to_delete and not profiles_to_add:
            self.output = "No memories to remove or add. Operation has been done."
            return

        cached_data = self.meta_memory.load(cache_key, auto_clean=False)
        existing_memory_nodes = []
        if cached_data:
            existing_memory_nodes = [MemoryNode(**node_data) for node_data in cached_data]

        removed_count = 0
        added_count = 0

        if profile_ids_to_delete:
            profile_ids_set = set(profile_ids_to_delete)
            existing_memory_nodes = [
                node for node in existing_memory_nodes if node.memory_id not in profile_ids_set
            ]
            removed_count = len(profile_ids_to_delete)
            logger.info(f"Removed {removed_count} memories from user profile.")

        new_memory_nodes = []
        if profiles_to_add:
            for mem in profiles_to_add:
                profile_content = mem.get("profile_content", "")
                timestamp = mem.get("timestamp", "")

                if not profile_content:
                    logger.warning("Skipping memory with empty content")
                    continue

                memory_node = self._build_memory_node(
                    memory_content=profile_content,
                    when_to_use="",
                    metadata={"timestamp": timestamp}
                )
                memory_node.memory_type = MemoryNode.MemoryType.PERSONAL
                memory_node.memory_target = memory_target

                new_memory_nodes.append(memory_node)

            added_count = len(new_memory_nodes)
            logger.info(f"Added {added_count} new memories to user profile.")

        updated_memory_nodes = existing_memory_nodes + new_memory_nodes

        nodes_data = [node.model_dump(exclude_none=True) for node in updated_memory_nodes]
        self.meta_memory.save(cache_key, nodes_data)

        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old memories")
        if added_count > 0:
            operations.append(f"added {added_count} new memories")

        if operations:
            self.output = f"Successfully {' and '.join(operations)} in user profile."
        else:
            self.output = "Operation has been done."

        logger.info(self.output)
