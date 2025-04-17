import asyncio
import logging
from contextlib import asynccontextmanager
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

@asynccontextmanager
async def mcp_tools_session(config: dict = MCP_CONFIG):
    async with MultiServerMCPClient(config) as mcp_client:
        yield mcp_client.get_tools()

# For quick testing.
if __name__ == "__main__":
    async def main():
        async with mcp_tools_session() as tools:
            logger.info("Loaded MCP tools: %s", [t.name for t in tools])

    asyncio.run(main())