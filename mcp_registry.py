"""MCP registry for chirppipe."""

from typing import Any, Callable, Protocol
from dataclasses import dataclass
import inspect

from mcp.types import Tool


class ToolConfig(Protocol):
    """Protocol for tool configuration."""
    
    def configure(self, config: dict[str, Any]) -> None:
        """Configure the tool with parameters."""
        ...
    
    def process(self, **kwargs: Any) -> dict[str, Any]:
        """Process the tool's main logic."""
        ...


@dataclass
class ToolDefinition:
    """Definition for an MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    config_class: type[ToolConfig] | None = None


class ToolRegistry:
    """Registry for managing MCP tools."""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[Tool]:
        """List all registered tools as MCP Tool objects."""
        return [
            Tool(name=tool.name, description=tool.description, inputSchema=tool.input_schema)
            for tool in self._tools.values()
        ]
    
    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return None
        result = tool.handler(arguments)
        return await result if inspect.isawaitable(result) else result


# Global registry instance
registry = ToolRegistry()
