"""Summary and hands-off tool for distributing summarized memory to appropriate agents."""

import json
from typing import TYPE_CHECKING

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.enumeration import MemoryType
from ...core.schema import MemoryNode

if TYPE_CHECKING:
    from ...mem_agent import BaseMemoryAgent


@C.register_op()
class SummaryAndHandsOff(BaseMemoryTool):
    """Distribute summarized memory task to appropriate agent based on memory_type."""

    def __init__(self, memory_agents: list["BaseMemoryAgent"], **kwargs):
        # Force enable_multiple to True since this tool only supports multiple tasks
        kwargs["enable_multiple"] = True
        kwargs["sub_ops"] = memory_agents or []
        super().__init__(**kwargs)
        from ...mem_agent import BaseMemoryAgent

        self.sub_ops: list[BaseMemoryAgent] = [a for a in self.sub_ops if isinstance(a, BaseMemoryAgent)]

    @property
    def memory_agent_dict(self) -> dict[MemoryType, "BaseMemoryAgent"]:
        """Returns a dictionary mapping memory types to their corresponding agents."""
        return {a.memory_type: a for a in self.sub_ops}

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        """Build shared schema properties and required fields for memory tasks."""
        properties = {
            "memory_type": {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [k.value for k in self.memory_agent_dict],
            },
            "memory_target": {
                "type": "string",
                "description": self.get_prompt("memory_target"),
            },
        }
        required = ["memory_type", "memory_target"]
        return properties, required

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple summary and hands-off tasks."""
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "summary_content": {
                    "type": "string",
                    "description": self.get_prompt("summary_content"),
                },
                "memory_tasks": {
                    "type": "array",
                    "description": self.get_prompt("memory_tasks"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["summary_content", "memory_tasks"],
        }

    @staticmethod
    def _parse_memory_type_target(task: dict):
        memory_type = task.get("memory_type", "")
        memory_target = task.get("memory_target", "")
        return {"memory_type": MemoryType(memory_type), "memory_target": memory_target}

    def _collect_tasks(self) -> list[dict]:
        """Collect memory tasks from context."""
        tasks: list[dict] = []
        memory_tasks: list[dict] = self.context.get("memory_tasks", [])
        for task in memory_tasks:
            tasks.append(self._parse_memory_type_target(task))
        return tasks

    async def execute(self):
        """Execute memory tasks by distributing to appropriate agents in parallel."""
        summary_content = self.context.get("summary_content", "")
        assert summary_content, "No summary content provided."

        # Build and store summary node
        summary_node = MemoryNode(
            memory_type=MemoryType.HISTORY,
            memory_target="",
            when_to_use=summary_content,
            content=self.messages_formated,
            ref_memory_id="",
            author=self.author,
            metadata={},
        )
        logger.info(f"Adding summary node: {summary_node.model_dump_json(indent=2, exclude_none=True)}")
        self.memory_nodes.append(summary_node)
        vector_node = summary_node.to_vector_node()
        await self.vector_store.delete(vector_ids=[vector_node.vector_id])
        await self.vector_store.insert([vector_node])

        # Collect tasks
        tasks = self._collect_tasks()
        if not tasks:
            self.output = "No valid memory tasks to execute."
            return

        # Submit tasks to corresponding agents
        agent_list = []
        for i, task in enumerate(tasks):
            memory_type: MemoryType = task["memory_type"]
            memory_target: str = task["memory_target"]

            if memory_type not in self.memory_agent_dict:
                logger.warning(f"No agent found for memory_type={memory_type}")
                continue

            agent = self.memory_agent_dict[memory_type].copy()
            agent_list.append([agent, memory_type, memory_target])

            logger.info(f"Task {i}: Submitting {memory_type.value} agent with summary for target={memory_target}")
            self.submit_async_task(
                agent.call,
                query=self.context.get("query", ""),
                messages=self.context.get("messages", []),
                memory_type=memory_type,
                memory_target=memory_target,
                description=self.context.get("description"),
                ref_memory_id=self.context.get("ref_memory_id", ""),
            )

        await self.join_async_tasks()

        # Collect results
        results = []
        for i, (agent, memory_type, memory_target) in enumerate(agent_list):
            result_str = str(agent.output)
            if agent.memory_nodes:
                self.memory_nodes.extend(agent.memory_nodes)

            results.append(
                {
                    "memory_type": memory_type.value,
                    "memory_target": memory_target,
                    "result": result_str[:200] + ("..." if len(result_str) > 200 else ""),
                }
            )
            logger.info(f"Task {i}: Completed {memory_type.value} agent for target={memory_target}")

        results_str = json.dumps(results, ensure_ascii=False, indent=2)
        self.output = f"Successfully executed summary and {len(results)} hands-off task(s):\n{results_str}"
