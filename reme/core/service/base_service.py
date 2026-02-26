"""Base service definitions for flow management."""

from abc import ABC, abstractmethod

from loguru import logger
from pydantic import BaseModel

from ..flow import BaseFlow
from ..schema import ToolCall
from ..service_context import ServiceContext
from ..utils import create_pydantic_model


class BaseService(ABC):
    """Abstract base class for services that integrate and execute flows."""

    def __init__(self, service_context: ServiceContext, **kwargs):
        """Initialize the base service."""
        self.service_context: ServiceContext = service_context
        self.service_config = self.service_context.service_config
        self.kwargs = kwargs

    @abstractmethod
    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Integrate a flow into the service and return its name if successful."""

    @staticmethod
    def _prepare_route(flow: BaseFlow) -> tuple[ToolCall, type[BaseModel]]:
        """Generate the request model and route name for a flow."""
        tool_call = flow.tool_call
        model = create_pydantic_model(tool_call.name, tool_call.parameters)
        return tool_call, model

    def run(self):
        """Initialize and integrate all flows registered in the global context."""
        flow_names: list[str] = []
        for flow in self.service_context.flows.values():
            flow_name = self.integrate_flow(flow)
            if flow_name:
                flow_names.append(flow_name)

        if flow_names:
            logger.info(f"Integrated {','.join(flow_names)}")
