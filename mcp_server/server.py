import os
import httpx
from typing import Iterable
from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from mcp.server.fastmcp import FastMCP
import uvicorn
from langchain_community.tools.tavily_search import TavilySearchResults
import logging 

load_dotenv()

# === Setup dedicated MCPServer logger ===
logger = logging.getLogger("MCPServer")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logger.addHandler(handler)
logger.propagate = False

# === Initialize MCP server ===
mcp = FastMCP("MCPServer")

tavily_search_tool = TavilySearchResults()

@mcp.tool(description="Perform a web search if query requires external knowledge or up to date information.")
def web_search(query: str):
    results, _ = tavily_search_tool._run(query)
    return results

@mcp.tool(description="Add two integers.")
def add(a: int, b: int) -> int:
    result = a + b
    return result

@mcp.tool(description="Multiply two integers.")
def multiply(a: int, b: int) -> int:
    result = a * b
    return result

@mcp.tool(description="Generate the Fibonacci sequence up to n terms.")
def fibonacci(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

# TODO: Update the description for the extract_pdf tool 
@mcp.tool(description="Load a PDF (or .txt) and return a list of ~1500 char Document chunks.")
def extract_pdf(path: str) -> str:
    """
    Ingest a .pdf or .txt file located at *path* and return a concise
    ~300-word executive summary.  Uses map-reduce summarisation so it
    scales to large documents.
    """
    # 1. Load document(s)
    if path.lower().endswith(".pdf"):
        docs = PyPDFLoader(path).load()
    else:
        docs = TextLoader(path).load()

    # 2. Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks : Iterable[Document] = splitter.split_documents(docs)

    return chunks

@mcp.tool(
    description="Search for images via Unsplash API. Returns a list of image URLs."
)
async def image_search(
    query: str,
    per_page: int = 2,
    page: int = 1,
) -> list[str]:
    """
    Query Unsplash API and return up to `per_page` image URLs for `query`.
    """
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("Missing UNSPLASH_ACCESS_KEY")
    endpoint = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {access_key}"}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "order_by": "relevant",  
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(endpoint, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    return [photo["urls"]["regular"] for photo in data.get("results", [])]

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    uvicorn.run(
        mcp.sse_app(),
        host="0.0.0.0",
        port=port,
        # leave log_config as default so Uvicorn’s HTTP/ASGI logs remain intact
        log_level="info",
    )