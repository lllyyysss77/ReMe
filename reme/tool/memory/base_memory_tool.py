"""Base class for memory tool"""

from abc import ABCMeta
from pathlib import Path

from ...core.enumeration import MemoryType
from ...core.op import BaseTool
from ...core.schema import ToolCall, MemoryNode, ToolAttr
from ...core.utils import CacheHandler


class BaseMemoryTool(BaseTool, metaclass=ABCMeta):
    """Base class for memory tool"""

    def __init__(
        self,
        enable_multiple: bool = True,
        enable_thinking_params: bool = False,
        local_memory_path: str = "./reme_local_memory",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_multiple: bool = enable_multiple
        self.enable_thinking_params: bool = enable_thinking_params
        self.local_memory_path: str = local_memory_path
        self.memory_nodes: list[MemoryNode | str] = []

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""

    @property
    def tool_call(self) -> ToolCall | None:
        """Get the tool call schema."""
        if self._tool_call is None:
            if self.enable_multiple:
                self._tool_call = self._build_multiple_tool_call()
            else:
                self._tool_call = self._build_tool_call()
            self._tool_call.name = self._tool_call.name or self.name

            # Add thinking parameter if enabled
            if self.enable_thinking_params:
                parameters = self._tool_call.parameters
                if parameters and parameters.properties is not None:
                    if "thinking" not in parameters.properties:
                        parameters.properties = {
                            "thinking": ToolAttr(
                                type="string",
                                description="Your complete and detailed thinking process "
                                "about how to fill in each parameter",
                            ),
                            **parameters.properties,
                        }
                        if parameters.required is not None:
                            parameters.required = ["thinking", *parameters.required]
                        else:
                            parameters.required = ["thinking"]
        return self._tool_call

    @property
    def local_memory(self) -> CacheHandler:
        """Create the meta memory cache handler."""
        return CacheHandler(Path(self.local_memory_path) / self.vector_store.collection_name)

    @property
    def memory_type(self) -> MemoryType:
        """Get the memory type from context."""
        return MemoryType(self.context.get("memory_type"))

    @property
    def memory_target(self) -> str:
        """Get the memory target from context."""
        return self.context.get("memory_target", "")

    @property
    def history_node(self) -> MemoryNode:
        """Get the history node from context."""
        return self.context.get("history_node")

    @property
    def retrieved_nodes(self) -> list[MemoryNode]:
        """Get the retrieved nodes from context."""
        return self.context.get("retrieved_nodes")

    @property
    def author(self) -> str:
        """Get the author from context."""
        return self.context.get("author", "")
