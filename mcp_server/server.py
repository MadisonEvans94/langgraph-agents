import os
import logging
from mcp.server.fastmcp import FastMCP
import uvicorn
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPServer")

# === Initialize MCP server ===
mcp = FastMCP("MCPServer")

# Tavily Web Search Tool (include `TAVILY_API_KEY` in .env in order to use)
# tavily_search_tool = TavilySearchResults()

# @mcp.tool()
# def web_search(query: str):
#     """
#     Perform a web search if query requires external knowledge or up to date information.
#     """
#     results, _ = tavily_search_tool._run(query)
#     return results

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    logger.info(f"Tool 'add' called with a={a}, b={b}")
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    logger.info(f"Tool 'multiply' called with a={a}, b={b}")
    return a * b

@mcp.tool()
def fibonacci(n: int) -> list[int]:
    """Generate the Fibonacci sequence up to n terms."""
    logger.info(f"Tool 'fibonacci' called with n={n}")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
