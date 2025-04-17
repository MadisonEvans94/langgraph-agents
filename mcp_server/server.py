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

@mcp.tool()
def fibonacci(n: int) -> list[int]: 
    """Generate the Fibonacci sequence up to n terms"""
    if n <= 0: 
        raise ValueError("n must be a positive integer")
    seq = [0, 1]
    while len(seq) < n: 
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
 