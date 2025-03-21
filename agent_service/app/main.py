import json
import logging
import sys
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.agent_factory import AgentFactory
from .models import QueryRequest
from .utils import load_llm_configs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

USE_OPENAI = True  # or decide if you want vLLM

# Load from your config.yaml
all_configs = load_llm_configs()
llm_configs = all_configs["openai"] if USE_OPENAI else all_configs["vllm"]

app = FastAPI(title="Conversational Agent API", version="1.0.0")

shared_memory = MemorySaver()
agent_factory = AgentFactory(memory=shared_memory, thread_id=str(uuid.uuid4()))

agent = agent_factory.factory(
    agent_type="conversational_agent_with_routing",
    llm_configs=llm_configs,
    use_openai=USE_OPENAI,
)

@app.post("/ask_stream")
async def ask_stream(request: QueryRequest):
    user_query = request.user_query
    logger.info(f"ask_stream received query: {user_query!r}")

    human_msg = HumanMessage(content=user_query)
    try:
        # agent.run_stream(...) returns an iterator of (stream_mode, data)
        stream_iter = agent.run_stream(human_msg)
        logger.info("Got streaming iterator from agent.")
    except Exception as e:
        logger.error("Error creating stream", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    def event_generator():
        """
        Yields SSE-like lines of JSON:
          - {"mode": "messages", "data": "... partial text ..."}
          - {"mode": "model_used", "data": "...model name..."}
        """
        for stream_mode, data in stream_iter:
            # data should be (chunk_obj, metadata_dict)
            if not (isinstance(data, tuple) and len(data) == 2):
                continue

            chunk_obj, metadata = data

            # Skip user messages
            if isinstance(chunk_obj, HumanMessage):
                continue

            # Stream partial tokens
            if isinstance(chunk_obj, AIMessageChunk):
                partial_text = chunk_obj.content
                yield json.dumps({"mode": "messages", "data": partial_text}) + "\n"
                continue

            # Final AIMessage => we do NOT repeat its text
            if isinstance(chunk_obj, AIMessage):
                # Instead, we just yield the model name
                final_model = chunk_obj.additional_kwargs.get("model_used", "unknown")
                yield json.dumps({"mode": "model_used", "data": final_model}) + "\n"
                break  # We are done streaming

        logger.info("Finished streaming tokens.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
