import asyncio
from train_agent.utils.mcp_utils import list_tools_and_resources
from train_agent.utils.settings import settings

async def main():
    mcp_url = settings.mcp_url
    tools, resources = await list_tools_and_resources(mcp_url)
    print("Tools:", [t.name for t in tools])
    print(
        "Resources:",
        [getattr(r, "uri", None) for r in resources or []],
    )

def cli():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())