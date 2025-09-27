# @title 🔌 MCP helpers

from contextlib import asynccontextmanager

from fastmcp import Client
from fastmcp.client import SSETransport, StreamableHttpTransport
from train_agent.utils.settings import settings
from mcp.types import Tool
from mcp.types import Resource

@asynccontextmanager
async def mcp_session(mcp_url: str):
    """
    Connects to the remote Smithery MCP server using the full URL that includes
    your API key & profile. No OAuth provider is used.
    """
    client = Client(
        StreamableHttpTransport(
            url=mcp_url,
            headers={
                "Authorization": f"Bearer {settings.mcp_bearer_token}"
                },
        )
    )
    yield client


async def list_tools_and_resources(mcp_url: str):
    """Return (tools_result, resources_result) from the remote Smithery server."""
    async with mcp_session(mcp_url) as client:
        async with client as mcp_client:
            tools = await mcp_client.list_tools()
            resources = await mcp_client.list_resources()
        return tools, resources


async def call_mcp_tool(tool_name: str, arguments: dict, mcp_url: str):
    """Invoke a tool on the remote Smithery server and return the CallToolResult."""
    async with mcp_session(mcp_url) as session:
        return await session.call_tool(tool_name, arguments)

def convert_tools_and_resources_to_dicts(tools_result: list[Tool], resources_result: list[Resource]):
    # Convert tools to the format expected by generate_scenarios
    tools_list = []
    for tool in tools_result or []:
        tools_list.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        })

    # Convert resources to the format expected by generate_scenarios  
    resources_list = []
    for resource in resources_result or []:
        resources_list.append({
            "uri": str(resource.uri),
            "name": resource.name,
            "description": resource.description,
            "mimeType": resource.mimeType,
        })

    return tools_list, resources_list
