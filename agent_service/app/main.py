import shutil
import tempfile
from loguru import logger
import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager
from mcp import ClientSession
from mcp.client.sse import sse_client
from agent_resources.agent_factory import AgentFactory
from agent_resources.agents.marketing_agent.marketing_supervisor_agent import MarketingSupervisorAgent
from agent_resources.state_types import MarketingSupervisorState
from langchain_core.messages import HumanMessage, AIMessage
from .models import QueryRequest, QueryResponse, MarketingSupervisorResponse
from .utils import load_llm_configs
from langgraph.checkpoint.memory import MemorySaver
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

GLOBAL_THREAD_ID = "TEST"     # for testing

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
shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory)

@app.post("/run_marketing_supervisor", response_model=MarketingSupervisorResponse)
async def run_marketing_supervisor(
    file:   UploadFile = File(...),
    prompt: str = Form("Create a landing page for the attached document.")
):
    # basic PDF-handling boilerplate
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="File must be a PDF")

    suffix   = os.path.splitext(file.filename)[1] or ".pdf"
    tmp_dir  = tempfile.mkdtemp(prefix="upload_")
    pdf_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{suffix}")
    with open(pdf_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        docs     = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks   = splitter.split_documents(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    full_text = "\n\n".join(c.page_content for c in chunks)

    # build / reuse sub-agents  (all share GLOBAL_THREAD_ID)
    landing_page_agent = AgentFactory(memory=shared_memory).factory(
        agent_type = "landing_page_agent",
        thread_id = GLOBAL_THREAD_ID,
        use_llm_provider = USE_LLM_PROVIDER,
        llm_configs = llm_configs,
        tools = app.state.tools,
    )

    social_media_agent = AgentFactory(memory=shared_memory).factory(
        agent_type = "social_media_agent",
        thread_id = GLOBAL_THREAD_ID,
        use_llm_provider = USE_LLM_PROVIDER,
        llm_configs = llm_configs,
        tools = [],
    )

    # supervisor  (same GLOBAL_THREAD_ID)
    supervisor = MarketingSupervisorAgent(
        llm_configs = llm_configs,
        thread_id = GLOBAL_THREAD_ID,
        tools = app.state.tools,
        sub_agents = [landing_page_agent, social_media_agent],
        use_llm_provider = USE_LLM_PROVIDER,
    )

    # invoke supervisor – we pass only *this turn's* user request
    sup_state: MarketingSupervisorState = {
        "messages": [HumanMessage(content=prompt)],
        "document_text": full_text,
        "remaining_steps": 25,
    }

    result_state = await supervisor.state_graph.ainvoke(sup_state)

    # pick up the last assistant turn
    last_ai_msg = next(m for m in reversed(result_state["messages"])
                             if isinstance(m, AIMessage))
    last_message_text = last_ai_msg.content.strip()

    html = result_state.get("html")
    html_path = None
    image_url = result_state.get("image_url")

    if html: 
        OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/marketing_agent_outputs")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        html_path = os.path.join(OUTPUT_DIR, "marketing_brochure.html")
        with open(html_path, "w", encoding="utf-8") as f: 
            f.write(html)
        logger.success(f"Supervisor HTML saved at {html_path}")

    # clean-up tmp files
    try:
        os.remove(pdf_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return MarketingSupervisorResponse(
        last_message = last_message_text,
        html = html,
        html_path = html_path,
        image_url = image_url,
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

    logger.info(f">>> Invoking agent: {agent_type}")
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