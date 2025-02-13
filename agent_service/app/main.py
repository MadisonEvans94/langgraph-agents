import logging
import time
import uuid
from .models import QueryRequest, QueryResponse
from fastapi import FastAPI, HTTPException
import os 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from .utils import load_llm_configs
from agent_resources.agent_factory import AgentFactory
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
llm_configs = load_llm_configs()

app = FastAPI(
    title="Conversational Agent API",
    description="An API for a conversation agent powered by vLLM and LangGraph.",
    version="1.0.0",
)

shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory, thread_id=str(uuid.uuid4()))
agent = agent_factory.factory("conversational_agent_with_routing", llm_configs=llm_configs)

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    start_time = time.perf_counter()
    user_query = request.user_query

    # Convert user query to a HumanMessage
    human_message = HumanMessage(content=user_query)

    try:
        ai_message = agent.run(human_message)
        response_time = time.perf_counter() - start_time

        logger.info(f"Agent response: {ai_message.content} (Time: {response_time:.2f}s)")
        return QueryResponse(response=ai_message.content)

    except Exception as e:
        logger.error("Error generating response", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing the query: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
