from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport, SSETransport, StdioTransport

from micro_graph import Node, NodeResult
from micro_graph.ai.types import ToolInfo


class MCPNode(Node):
    def __init__(self, url_or_command, mode="http", header: dict[str, str] | None = None, max_retries: int = 0):
        super().__init__(max_retries=max_retries)
        if mode == "http":
            transport = StreamableHttpTransport(url_or_command, headers=header)
        elif mode == "sse":
            transport = SSETransport(url_or_command, headers=header)
        elif mode == "stdio":
            command, *args = url_or_command.split()
            transport = StdioTransport(command, args)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        self._client = Client(transport=transport)

    async def list_tools(self) -> list[ToolInfo]:
        async with self._client:
            tools = await self._client.list_tools()
            return [
                ToolInfo(
                    name=tool.name,
                    description=tool.description or "missing description",
                    arguments=str(tool.outputSchema or "missing arguments definition")
                )
                for tool in tools
            ]

    async def run(self, shared: dict, tool_name: str = "", **kwargs) -> NodeResult:
        async with self._client:
            result =  await self._client.call_tool(name=tool_name, arguments=kwargs)
            return dict(result.__dict__)
