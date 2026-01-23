"""service"""

from .base_service import BaseService
from .cmd_service import CmdService
from .http_service import HttpService
from .mcp_service import MCPService
from ..context import R

__all__ = [
    "BaseService",
    "CmdService",
    "HttpService",
    "MCPService",
]

R.service.register("cmd")(CmdService)
R.service.register("http")(HttpService)
R.service.register("mcp")(MCPService)
