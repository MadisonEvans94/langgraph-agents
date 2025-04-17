import json
import logging
import os
import uuid
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
from agent_resources.agent_factory import AgentFactory
from .models import QueryRequest
from .utils import load_llm_configs
from langgraph.checkpoint.memory import MemorySaver
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()

# === App-wide constants ===
USE_OPENAI = True
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/sse")
all_configs = load_llm_configs()
llm_configs = all_configs.get("openai" if USE_OPENAI else "vllm", {})

# === Shared memory + factory ===
shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safe default before attempting to load
    app.state.tools = []

    try:
        async with sse_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                app.state.tools = await load_mcp_tools(session)
    except Exception as e:
        logger.warning(f"Failed to load tools from MCP: {e}")

    yield  # App runs here

    # Add shutdown cleanup here if needed

app = FastAPI(title="Agent MCP Client", lifespan=lifespan)

@app.post("/ask")
async def ask(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    agent_type = request.agent_type or "mcp_agent"
    tools = getattr(app.state, "tools", [])

    agent = agent_factory.factory(
        agent_type=agent_type,
        thread_id=thread_id,
        use_openai=USE_OPENAI,
        llm_configs=llm_configs,
        tools=tools
    )

    ai_msg = await agent.ainvoke(HumanMessage(content=request.user_query))

    return JSONResponse(content={
        "response": ai_msg.content,
        "thread_id": thread_id,
        "agent_type": agent_type
    })

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
