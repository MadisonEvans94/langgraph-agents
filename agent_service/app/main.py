import os
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain-style message classes
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages

# AgentFactory is presumably your own code that knows how to build/return the agent
from agent_resources.agent_factory import AgentFactory
from langgraph.checkpoint.memory import MemorySaver

# vLLM / OpenAI-compatible library
from openai import OpenAI  

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Check environment configs
downstream_server = os.getenv("VLLM_DOWNSTREAM_HOST")
if not downstream_server:
    raise ValueError("VLLM_DOWNSTREAM_HOST environment variable is not set.")

openai_api_base = f"{downstream_server}/v1"
api_key = os.getenv("OPENAI_API_KEY")

# Model & hyperparams
llm_id = os.getenv("LLM_ID", "meta-llama/Llama-3.1-8B-Instruct") 
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 512))
top_p = float(os.getenv("TOP_P", 1.0))
temperature = float(os.getenv("TEMPERATURE", 1.0))
repetition_penalty = float(os.getenv("REPETITION_PENALTY", 1.0))

client_params = {"api_key": api_key}
client_params["base_url"] = openai_api_base
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
            # Convert LangChain messages to OpenAI format using LangChain utility
            openai_messages = convert_to_openai_messages(messages)
            params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            if not api_key:
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
    model=llm_id,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
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
        # agent.run(...) eventually calls llm.invoke(...) with the entire conversation
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
