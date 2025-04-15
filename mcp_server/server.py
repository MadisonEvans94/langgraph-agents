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
    # Use SSE transport and run with uvicorn on port 8002 (or use MCP_SERVER_PORT env var)
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    # mcp.sse_app() returns an ASGI app for SSE transport
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
