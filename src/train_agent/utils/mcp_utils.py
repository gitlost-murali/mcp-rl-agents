# @title 🔌 MCP helpers

from contextlib import asynccontextmanager

from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from train_agent.utils.settings import settings
from fastmcp.client.client import CallToolResult
from mcp.types import Tool
from mcp.types import Resource
from train_agent.utils.debug_utils import log
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

complete_task_tool: ChatCompletionToolParam = ChatCompletionToolParam(
    type="function",
    function={
        "name": "complete_task",
        "description": "Complete the task with a summary",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of accomplishments",
                }
            },
            "required": ["summary"],
        },
    },
)


@asynccontextmanager
async def mcp_session(mcp_url: str):
    """
    Connects to the remote Smithery MCP server using the full URL that includes
    your API key & profile. No OAuth provider is used.
    """
    client = Client(
        StreamableHttpTransport(
            url=mcp_url,
            headers={"Authorization": f"Bearer {settings.mcp_bearer_token}"},
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


async def call_mcp_tool(
    tool_name: str, arguments: dict, mcp_url: str
) -> CallToolResult:
    """Invoke a tool on the remote Smithery server and return the CallToolResult."""
    async with mcp_session(mcp_url) as session:
        async with session as mcp_client:
            return await mcp_client.call_tool(tool_name, arguments)


def convert_tools_and_resources_to_dicts(
    tools_result: list[Tool], resources_result: list[Resource]
):
    # Convert tools to the format expected by generate_scenarios
    tools_list = []
    for tool in tools_result or []:
        tools_list.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        )

    # Convert resources to the format expected by generate_scenarios
    resources_list = []
    for resource in resources_result or []:
        resources_list.append(
            {
                "uri": str(resource.uri),
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType,
            }
        )

    return tools_list, resources_list


def get_content_text(result: CallToolResult) -> str:
    return str(result)


async def get_tool_schemas_from_mcp(mcp_url: str):
    # Discover available tools from the remote server
    tools_result, _resources_result = await list_tools_and_resources(mcp_url)
    tool_names = [t.name for t in tools_result]
    log("rollout: discovered tools", count=len(tool_names), names=tool_names)

    # Convert to OpenAI tool format
    tool_schemas = []
    for tool in tools_result:
        tool_schema = ChatCompletionToolParam(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        )
        tool_schemas.append(tool_schema)

    return tool_schemas


async def get_tool_schemas_from_mcp_with_complete_task_tool(mcp_url: str):
    tool_schemas = await get_tool_schemas_from_mcp(mcp_url)
    tool_schemas.append(complete_task_tool)
    return tool_schemas
