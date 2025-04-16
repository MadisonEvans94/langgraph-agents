import json
import logging
import os
import uuid
from agent_resources.agent_factory import AgentFactory
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
from .models import QueryRequest
from .utils import load_llm_configs
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()

USE_OPENAI = True
all_configs = load_llm_configs()
llm_configs = all_configs.get("openai" if USE_OPENAI else "vllm", {})

app = FastAPI(title="Agent MCP Client")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/sse")
shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory)

@app.post("/ask")
async def ask(request: QueryRequest):
    thread_id = request.thread_id or "default"

    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Create agent with persistent memory and ephemeral MCP session
            agent = await agent_factory.factory(
                agent_type="mcp_agent",
                use_openai=USE_OPENAI,
                use_mcp=True,
                mcp_session=session,
                llm_configs=llm_configs,
                thread_id=thread_id
            )

            ai_msg = await agent.run_async(message=request.user_query)

            return JSONResponse(content={"response": ai_msg.content, "thread_id": thread_id})

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
