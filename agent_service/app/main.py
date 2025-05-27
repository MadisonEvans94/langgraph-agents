import shutil
import tempfile
from loguru import logger
import os
import uuid
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager
from mcp import ClientSession
from mcp.client.sse import sse_client
from agent_resources.agent_factory import AgentFactory
from langchain_core.messages import HumanMessage
from .models import QueryRequest, QueryResponse
from .utils import load_llm_configs
from langgraph.checkpoint.memory import MemorySaver
from pprint import pformat
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>\n",
    level="INFO",
)

load_dotenv()

USE_LLM_PROVIDER = os.getenv("USE_LLM_PROVIDER", True)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/sse")
all_configs = load_llm_configs()
llm_configs = all_configs.get(LLM_PROVIDER if USE_LLM_PROVIDER else "vllm", {})

shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tools = []
    app.state.mcp_session = None

    sse_cm = sse_client(MCP_SERVER_URL)
    try:
        read, write = await sse_cm.__aenter__()
        session_cm = ClientSession(read, write)
        session = await session_cm.__aenter__()
        await session.initialize()

        app.state.mcp_session = session
        app.state.tools = await load_mcp_tools(session)
        logger.success("✅ Loaded MCP tools")
        yield

    except Exception as e:
        logger.error(f"Failed to connect to MCP: {e}")
        yield

    finally:
        if app.state.mcp_session:
            await session_cm.__aexit__(None, None, None)
        await sse_cm.__aexit__(None, None, None)
        logger.info("MCP session closed")

app = FastAPI(title="Agent MCP Client", lifespan=lifespan)

class SupervisorResponse(BaseModel):
    summary: str
    key_points: list[str]
    domain: str
    image_query: str | None
    images: list[str]
    html: str

@app.post("/run_marketing_agent", response_model=SupervisorResponse)
async def run_supervisor(file: UploadFile = File(...)):
    # 1. Validate & save incoming PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="File must be a PDF")
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    tmp_dir = tempfile.mkdtemp(prefix="upload_")
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{suffix}")
    with open(tmp_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # 2. Load & split document
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks: List[Document] = splitter.split_documents(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # 3. Instantiate the supervisor agent
    svc_agent = agent_factory.factory(
        agent_type="supervisor_agent",
        thread_id=str(uuid.uuid4()),
        use_llm_provider=USE_LLM_PROVIDER,
        llm_configs=llm_configs,
        tools=app.state.tools,
    )

    # 4. Concatenate all chunks into full text
    full_text = "\n\n".join(chunk.page_content for chunk in chunks)
    messages = [HumanMessage(content=full_text)]
    
    # 5. Run the full PDF→analysis→image search→html generation flow
    result = await svc_agent.ainvoke(messages)

    # 6. Clean up
    try:
        os.remove(tmp_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # 7. Unpack and return
    analysis = result.get("analysis", {})
    return SupervisorResponse(
        summary=analysis.get("summary", ""),
        key_points=analysis.get("key_points", []),
        domain=analysis.get("domain", ""),
        image_query=result.get("image_query"),
        images=result.get("images", []),
        html=result.get("html", "")
    )
        
@app.post("/invoke")
async def ask(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    agent_type = request.agent_type or "react_agent"
    tools = app.state.tools

    agent = agent_factory.factory(
        agent_type=agent_type,
        thread_id=thread_id,
        use_llm_provider=USE_LLM_PROVIDER,
        llm_configs=llm_configs,
        tools=tools,
    )

    logger.info(f"🧠 Invoking agent: {agent_type}")
    ai_msg = await agent.ainvoke(request.user_query)
    return QueryResponse(
        response=ai_msg.content,
        thread_id=thread_id,
        agent_type=agent_type,
    )

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))