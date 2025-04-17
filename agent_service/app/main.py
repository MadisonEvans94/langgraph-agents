import json
import logging
import os
import uuid
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
from agent_resources.agent_factory import AgentFactory
from .models import QueryRequest
from .utils import load_llm_configs
from langgraph.checkpoint.memory import MemorySaver

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

app = FastAPI(title="Agent MCP Client")

@app.post("/ask")
async def ask(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    agent_type = request.agent_type or "mcp_agent"
    use_mcp = request.use_mcp if request.use_mcp is not None else True

    # Always create a new agent with per-request thread_id + tools
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await agent_factory.factory(
                agent_type=agent_type,
                thread_id=thread_id,
                use_openai=USE_OPENAI,
                use_mcp=use_mcp,
                mcp_session=session,
                llm_configs=llm_configs,
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
