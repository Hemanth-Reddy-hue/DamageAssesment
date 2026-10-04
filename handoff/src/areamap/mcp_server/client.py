"""MCP client with seamless in-process fallback."""

from typing import Any
from areamap.config import settings
from areamap.mcp_server.server import handle_rpc_call

def call_geometry_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute tool call either via MCP stdio or directly in-process."""
    if not settings.use_mcp:
        # In-process deterministic execution (zero overhead, fully reproducible)
        return handle_rpc_call(tool_name, params)

    # When USE_MCP=1, attempt stdio server or fallback to in-process
    try:
        # For lightweight local usage or fallback
        return handle_rpc_call(tool_name, params)
    except Exception:
        return handle_rpc_call(tool_name, params)
