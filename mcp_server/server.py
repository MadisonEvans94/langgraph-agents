import os
from mcp.server.fastmcp import FastMCP
import uvicorn
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables (e.g. TAVILY_API_KEY)
load_dotenv()

# === Initialize MCP server ===
mcp = FastMCP("MCPServer")

# === Instantiate Tavily search tool ===
tavily_search_tool = TavilySearchResults()

# === MCP‐registered tools ===

@mcp.tool()
def web_search(query: str):
    """
    Perform a web search if query requires external knowledge or up to date information.
    """
    results, _ = tavily_search_tool._run(query)
    return results

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def fibonacci(n: int) -> list[int]:
    """Generate the Fibonacci sequence up to n terms."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
