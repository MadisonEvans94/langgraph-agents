import asyncio
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    async with MultiServerMCPClient(config) as mcp_client:
        tools = mcp_client.get_tools()
        tool_names = [t.name for t in tools]
        logger.info("Loaded MCP tools: %s", tool_names)
        return tools


# For quick testing.
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    mcp_tools = loop.run_until_complete(initialize_mcp_tools())
    logger.info("Loaded MCP tools: %s", [t.name for t in mcp_tools])
