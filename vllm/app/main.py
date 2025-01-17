import os
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from agent_resources.agent_factory import AgentFactory
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI  # vLLM OpenAI-compatible client

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check vLLM server and model configuration
downstream_server = os.getenv("VLLM_DOWNSTREAM_HOST")
if not downstream_server:
    raise ValueError("VLLM_DOWNSTREAM_HOST environment variable is not set.")
openai_api_base = f"{downstream_server}/v1"

# Set up OpenAI-compatible vLLM client
openai_api_key = "EMPTY"  # OpenAI compatibility mode doesn't require a key
vllm_openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Initialize FastAPI app
app = FastAPI(
    title="Conversational Agent API",
    description="An API to interact with conversational agents powered by vLLM and LangGraph.",
    version="1.0.0",
)

# Shared memory setup
shared_memory = MemorySaver()

# AgentFactory initialized with vLLM
agent_factory = AgentFactory(llm=vllm_openai_client, memory=shared_memory)

# Input model for API requests
class QueryRequest(BaseModel):
    agent_type: str
    user_query: str

# Response model for API responses
class QueryResponse(BaseModel):
    response: str

# Endpoint to handle user queries
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    start_time = time.perf_counter()
    agent_type = request.agent_type
    user_query = request.user_query

    logger.info(f"Received query: {user_query} for agent type: {agent_type}")

    try:
        # Instantiate the agent using the AgentFactory
        agent = agent_factory.factory(agent_type)
    except ValueError as e:
        logger.error(f"Invalid agent type: {agent_type}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize the agent.")

    # Process the query using the agent
    try:
        human_message = HumanMessage(content=user_query)
        ai_message = agent.run(human_message)
        response_time = time.perf_counter() - start_time

        logger.info(f"Agent response: {ai_message.content} (Time: {response_time:.2f}s)")
        return QueryResponse(response=ai_message.content)
    except Exception as e:
        logger.error("Error generating response", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing the query: {e}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
