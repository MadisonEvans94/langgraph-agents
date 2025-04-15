# test_mcp_client.py
import asyncio
from agent_resources.mcp_integration import initialize_mcp_tools

async def test_mcp():
    tools = await initialize_mcp_tools()
    for tool in tools:
        print(f"Tool name: {tool.name}, description: {tool.description}")

if __name__ == "__main__":
    asyncio.run(test_mcp())
