import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain-style message classes & utilities
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages

# Your custom factories / libraries
from agent_resources.agent_factory import AgentFactory
from langgraph.checkpoint.memory import MemorySaver

# vLLM / OpenAI-compatible library
from openai import OpenAI

# Import constants from constants.py
from agent_service.app.constants import (
    API_BASE_URL,
    API_KEY,
    LLM_ID,
    MAX_NEW_TOKENS,
    TOP_P,
    TEMPERATURE,
    REPETITION_PENALTY,
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure the OpenAI client
client_params = {"api_key": API_KEY, "base_url": API_BASE_URL}
client = OpenAI(**client_params)

class ChatVLLMWrapper:
    """
    Wraps a chat completion endpoint using OpenAI/vLLM.
    Accepts a list of BaseMessage objects and returns an AIMessage.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ):
        self.client = client
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Accepts a list of messages. Calls chat completion endpoint.
        Returns an AIMessage containing the new assistant message.
        """
        try:
            openai_messages = convert_to_openai_messages(messages)
            params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            # Example logic to conditionally include repetition_penalty
            if not API_KEY:
                params["repetition_penalty"] = self.repetition_penalty

            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content.strip()

            return AIMessage(content=content)

        except Exception as e:
            logger.error(f"Error invoking vLLM client: {e}", exc_info=True)
            raise RuntimeError(f"vLLM invocation failed: {e}")


# Create the wrapper
llm = ChatVLLMWrapper(
    client=client,
    model=LLM_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
)

# Setup FastAPI
app = FastAPI(
    title="Conversational Agent API",
    description="An API for a conversation agent powered by vLLM and LangGraph.",
    version="1.0.0",
)

# Shared memory
shared_memory = MemorySaver()

# Construct an agent from your factory
agent_factory = AgentFactory(llm=llm, memory=shared_memory)
agent = agent_factory.factory("conversational_agent")


class QueryRequest(BaseModel):
    agent_type: str
    user_query: str


class QueryResponse(BaseModel):
    response: str


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    start_time = time.perf_counter()
    user_query = request.user_query

    # Convert user query to a HumanMessage
    human_message = HumanMessage(content=user_query)

    try:
        # agent.run(...) eventually calls llm.invoke(...)
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
