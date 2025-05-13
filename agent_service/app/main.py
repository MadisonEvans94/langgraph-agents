import logging
import os
import uuid
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
from mcp import ClientSession
from mcp.client.sse import sse_client
from agent_resources.agent_factory import AgentFactory
from .models import QueryRequest, QueryResponse
from .utils import load_llm_configs
from langgraph.checkpoint.memory import MemorySaver
from typing import List
from agent_resources.state_types import Task
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()

#  App-wide constants
USE_LLM_PROVIDER = os.getenv("USE_LLM_PROVIDER", True)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/sse")
all_configs = load_llm_configs()
llm_configs = all_configs.get(LLM_PROVIDER if USE_LLM_PROVIDER else "vllm", {})

# Shared memory + factory 
shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Open a single longlived SSE + ClientSession to the MCP server.
    """
    # Safe default
    app.state.tools = []
    app.state.mcp_session = None

    sse_cm = sse_client(MCP_SERVER_URL)
    try:
        read, write = await sse_cm.__aenter__()
        session_cm = ClientSession(read, write)
        session = await session_cm.__aenter__()
        await session.initialize()

        # Keep handles for later use / health checks
        app.state.mcp_session = session
        app.state.tools = await load_mcp_tools(session)
        logger.info(f"✅ Loaded {len(app.state.tools)} MCP tools")

        yield 

    except Exception as e:
        logger.warning(f"Failed to connect to MCP: {e}")
        yield

    finally:
        # Graceful shutdown
        if app.state.mcp_session:
            await session_cm.__aexit__(None, None, None)
        await sse_cm.__aexit__(None, None, None)
        logger.info("MCP session closed")

app = FastAPI(title="Agent MCP Client", lifespan=lifespan)

@app.post("/invoke")
async def ask(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    agent_type = request.agent_type or "react_agent"

    # Reuse cached tool wrappers
    tools = app.state.tools

    agent = agent_factory.factory(
        agent_type=agent_type,
        thread_id=thread_id,
        use_llm_provider=USE_LLM_PROVIDER,
        llm_configs=llm_configs,
        tools=tools,
    )

    ai_msg = await agent.ainvoke(HumanMessage(content=request.user_query))

    return QueryResponse(
        response=ai_msg.content,
        thread_id=thread_id,
        agent_type=agent_type,
    )

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run_tasks")
async def run_tasks(tasks: List[Task]):
    """
    Accepts a JSON array of Task objects and returns the aggregated result.
    """
    agent = agent_factory.factory(
        agent_type="orchestrator_agent",
        thread_id=str(uuid.uuid4()),
        use_llm_provider=USE_LLM_PROVIDER,
        llm_configs=llm_configs,
        tools=app.state.tools,
    )
    ai_msg: AIMessage = await agent.process_tasks(tasks)
    return {"result": ai_msg.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))