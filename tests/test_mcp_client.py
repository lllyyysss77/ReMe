"""Test module for demonstrating MCPClient functionality."""

import asyncio
import json

from reme_ai.core.utils import MCPClient


async def main():
    """Execute demonstration of the MCPClient."""
    test_mcp = "test_mcp"
    config_data = {
        "mcpServers": {
            test_mcp: {
                "url": "http://127.0.0.1:8010/sse",
            },
        },
    }

    client = MCPClient(config_data)

    try:
        t_list = await client.list_tool_calls(test_mcp)
        for t in t_list:
            print(json.dumps(t, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
