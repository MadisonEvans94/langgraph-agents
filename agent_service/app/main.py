import json
import logging
import os
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_core.messages import AIMessageChunk
from .models import QueryRequest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()

app = FastAPI(title="Agent MCP Client")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/sse")
model = ChatOpenAI(model="gpt-3.5-turbo")

@app.post("/ask_stream")
async def ask_stream(request: QueryRequest):
    # Explicitly create an SSE transport (read, write) for MCP
    async with sse_client(MCP_SERVER_URL) as (read, write):
        # ClientSession handles MCP messaging lifecycle
        async with ClientSession(read, write) as session:
            # Initialize session (required!)
            await session.initialize()

            # Load MCP tools dynamically into LangGraph
            tools = await load_mcp_tools(session)

            # Create React-style LangGraph agent
            agent = create_react_agent(model, tools)

            # Stream agent response (correctly!)
            stream_iter = agent.astream({"messages": request.user_query})

            async def event_generator():
                async for update_dict in stream_iter:
                    if "messages" in update_dict:
                        messages = update_dict["messages"]
                        for message in messages:
                            if isinstance(message, AIMessageChunk):
                                yield json.dumps({"content": message.content}) + "\n"
                            else:
                                yield json.dumps({"content": str(message)}) + "\n"
                    else:
                        yield json.dumps({"update": str(update_dict)}) + "\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/ask")
async def ask(request: QueryRequest):
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            # Invoke the agent without streaming
            agent_response = await agent.ainvoke({"messages": request.user_query})

            # Extract the final AI message content safely
            final_response = ""
            if "messages" in agent_response:
                messages = agent_response["messages"]
                for message in reversed(messages):
                    if hasattr(message, 'content') and message.content.strip():
                        final_response = message.content
                        break

            return {"response": final_response}


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
