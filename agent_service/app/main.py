import logging
import time
import uuid
import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.agent_factory import AgentFactory
from .models import QueryRequest, QueryResponse
from .utils import load_llm_configs

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Decide if we're using OpenAI or vLLM
USE_OPENAI = True

# Load entire config
all_configs = load_llm_configs()

# Extract ONLY "default_llm" and "alternate_llm" from the correct provider
llm_configs = all_configs.get("openai", {}) if USE_OPENAI else all_configs.get("vllm", {})

logger.info("Extracted LLM configs for agent:")
logger.info(json.dumps(llm_configs, indent=2))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

app = FastAPI(
    title="Conversational Agent API",
    description="An API for conversation agents powered by LangGraph, etc.",
    version="1.0.0",
)

shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory, thread_id=str(uuid.uuid4()))

agent = agent_factory.factory(
    agent_type="conversational_agent_with_routing",
    llm_configs=llm_configs,
    use_openai=USE_OPENAI,
)

@app.post("/ask_stream")
async def ask_question_stream(request: QueryRequest):
    user_query = request.user_query
    logger.info(f"/ask_stream endpoint called with user_query={user_query!r}")

    human_message = HumanMessage(content=user_query)
    try:
        stream_iterator = agent.run_stream(human_message)
        logger.info("Initialized streaming generator from agent.run_stream(...)")
    except Exception as e:
        logger.error("Error initializing streaming response", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing the query: {e}")

    def event_generator():
        for stream_mode, data in stream_iterator:
            logger.info(f"ask_stream generator: stream_mode={stream_mode!r}, data={data!r}")

            # Typically data = (message_obj, metadata_dict)
            if isinstance(data, tuple) and len(data) == 2:
                chunk_obj, metadata = data

                # 1) Skip user messages
                if isinstance(chunk_obj, HumanMessage):
                    continue

                # 2) Stream partial AI chunks
                if isinstance(chunk_obj, AIMessageChunk):
                    chunk_text = chunk_obj.content
                    logger.info(f"   => Streaming partial chunk: {chunk_text!r}")
                    yield json.dumps({"mode": stream_mode, "data": chunk_text}) + "\n"
                    continue

                # 3) Skip the final full AIMessage to avoid duplication
                if isinstance(chunk_obj, AIMessage):
                    logger.info("Skipping final AIMessage to avoid duplicated text.")
                    continue

        logger.info("Streaming completed.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
