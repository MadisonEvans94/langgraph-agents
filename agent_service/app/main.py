# agent_service/app/main.py

import logging
import time
import uuid
import os
import sys
import json
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.agent_factory import AgentFactory
from .models import QueryRequest, QueryResponse
from .utils import load_llm_configs

load_dotenv()

# Decide if we're using openai or not
USE_OPENAI = True

# Load entire config
all_configs = load_llm_configs()

# Extract ONLY "default_llm" and "alternate_llm" from the correct provider
if USE_OPENAI:
    llm_configs = all_configs.get("openai", {})
else:
    llm_configs = all_configs.get("vllm", {})

# Debugging log to confirm extracted config
print("Extracted LLM configs for agent:", json.dumps(llm_configs, indent=2))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Conversational Agent API",
    description="An API for conversation agents powered by LangGraph, etc.",
    version="1.0.0",
)

shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory, thread_id=str(uuid.uuid4()))

# Build an agent with explicit use_openai parameter
agent = agent_factory.factory(
    agent_type="conversational_agent_with_routing",
    llm_configs=llm_configs,
    use_openai=USE_OPENAI,  # <- explicitly passed here
)


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    start_time = time.perf_counter()
    user_query = request.user_query
    agent_type = request.agent_type  # Currently unused, could remove later

    human_message = HumanMessage(content=user_query)

    try:
        ai_message = agent.run(human_message)
        response_time = time.perf_counter() - start_time
        logger.info(f"Agent response: {ai_message.content} (Time: {response_time:.2f}s)")

        # Retrieve model and tools from additional_kwargs
        model_used = ai_message.additional_kwargs.get("model_used", "")
        tools_used = ai_message.additional_kwargs.get("tools_used", [])

        return QueryResponse(
            response=ai_message.content,
            model_used=model_used,
            tools_used=tools_used,
        )

    except Exception as e:
        logger.error("Error generating response", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing the query: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
