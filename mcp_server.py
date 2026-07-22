"""MCP server for chirppipe."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.types import Tool, TextContent, ImageContent
from mcp.server.stdio import stdio_server
from mcp.server import Server

from mcp_registry import registry
from mcp_definitions import register_tools


async def handle_tool_call(name: str, arguments: dict) -> list[TextContent] | list[ImageContent] | TextContent:
    """Handle tool execution."""
    result = await registry.execute(name, arguments)
    return result if result else TextContent(type="text", text=f"Unknown tool: {name}")


async def main():
    server = Server("chirp-pipe")
    register_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return registry.list_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None = None) -> list[TextContent] | list[ImageContent]:
        if arguments is None:
            arguments = {}
        result = await handle_tool_call(name, arguments)
        return result if isinstance(result, list) else [result]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
