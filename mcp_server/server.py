# mcp_server/server.py
import os
from mcp.server.fastmcp import FastMCP
import uvicorn

mcp = FastMCP("MCPServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
 