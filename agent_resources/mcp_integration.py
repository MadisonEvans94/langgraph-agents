# agent_resources/mcp_integration.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configuration for your real MCP server endpoint.
# In this case, we assume a single MCP server with tools (like add and multiply)
# is running at http://localhost:8002/sse.
MCP_CONFIG = {
    "mcp": {
        "url": "http://mcp-server:8002/sse",  # <--- use container name here
        "transport": "sse",
    }
}


async def initialize_mcp_tools(config: dict = MCP_CONFIG):
    """
    Initializes the MultiServerMCPClient and loads MCP tools.
    Returns a list of MCP tools converted to LangChain-compatible tools.
    """
    async with MultiServerMCPClient(config) as mcp_client:
        tools = mcp_client.get_tools()
        return tools

# For quick testing.
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    mcp_tools = loop.run_until_complete(initialize_mcp_tools())
    print(f"Loaded MCP tools: {[t.name for t in mcp_tools]}")
