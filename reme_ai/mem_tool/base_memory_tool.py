"""Base class for memory tool"""

from abc import ABCMeta
from pathlib import Path

from loguru import logger

from ..core.enumeration import MemoryType
from ..core.op import BaseOp
from ..core.schema import ToolCall, MemoryNode
from ..core.utils import CacheHandler


class BaseMemoryTool(BaseOp, metaclass=ABCMeta):
    """Base class for memory tool"""

    def __init__(
        self,
        enable_multiple: bool = True,
        enable_thinking_params: bool = False,
        meta_memory_path: str = "./meta_memory",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_multiple: bool = enable_multiple
        self.enable_thinking_params: bool = enable_thinking_params
        self.meta_memory_path: str = meta_memory_path
        self.memory_nodes: list[MemoryNode | str] = []

    def _build_parameters(self) -> dict:
        return {}

    def _build_multiple_parameters(self) -> dict:
        return {}

    def _build_tool_description(self) -> str:
        """Build tool description."""
        return self.get_prompt("tool" + ("_multiple" if self.enable_multiple else ""))

    def _build_tool_call(self) -> ToolCall:
        tool_call_params: dict = {
            "description": self._build_tool_description(),
        }

        if self.enable_multiple:
            parameters = self._build_multiple_parameters()
        else:
            parameters = self._build_parameters()

        if parameters:
            tool_call_params["parameters"] = parameters

            if self.enable_thinking_params and "thinking" not in parameters["properties"]:
                parameters["properties"] = {
                    "thinking": {
                        "type": "string",
                        "description": "Your complete and detailed thinking process about how to fill in each parameter",
                    },
                    **parameters["properties"],
                }
                parameters["required"] = ["thinking", *parameters["required"]]

        return ToolCall(**tool_call_params)

    @property
    def meta_memory(self) -> CacheHandler:
        """Create the meta memory cache handler."""
        return CacheHandler(Path(self.meta_memory_path) / self.vector_store.collection_name)

    @property
    def memory_type(self) -> MemoryType:
        """Get the memory type from context."""
        return MemoryType(self.context.get("memory_type"))

    @property
    def memory_target(self) -> str:
        """Get the memory target from context."""
        return self.context.get("memory_target", "")

    @property
    def ref_memory_id(self) -> str:
        """Get the reference memory ID from context."""
        return self.context.get("ref_memory_id", "")

    @property
    def messages_formated(self) -> str:
        """Get the formated messages from context."""
        return self.context.get("messages_formated", "")

    @property
    def retrieved_nodes(self) -> list[MemoryNode]:
        """Get the retrieved nodes from context."""
        return self.context.get("retrieved_nodes")

    @property
    def author(self) -> str:
        """Get the author from context."""
        return self.context.get("author", "")

    def _build_memory_node(
        self,
        memory_content: str,
        memory_type: MemoryType | None = None,
        memory_target: str = "",
        ref_memory_id: str = "",
        when_to_use: str = "",
        author: str = "",
        metadata: dict | None = None,
    ) -> MemoryNode:
        """Build MemoryNode from content, when_to_use, and metadata."""
        node = MemoryNode(
            memory_type=memory_type or self.memory_type,
            memory_target=memory_target or self.memory_target,
            when_to_use=when_to_use or "",
            content=memory_content,
            ref_memory_id=ref_memory_id or self.ref_memory_id,
            author=author or self.author,
            metadata=metadata or {},
        )

        # logger.opt(depth=1).info(
        #     f"[{self.__class__.__name__}] build node={node.model_dump_json(indent=2, exclude_none=True)}",
        # )
        return node
