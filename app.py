import os
import logging
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from agent_service.app.utils import load_llm_configs
from agent_resources.agents.supervisor.supervisor import SupervisorAgent
from agent_resources.prompts import SUPERVISOR_AGENT_PROMPT
from agent_resources.state_types import SupervisorState

load_dotenv()


# === LLM Config Loading ===
USE_LLM_PROVIDER = os.getenv("USE_LLM_PROVIDER", "True").lower() in ("1", "true", "yes")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
all_configs = load_llm_configs()
llm_configs = all_configs.get(LLM_PROVIDER if USE_LLM_PROVIDER else "vllm")
if not llm_configs:
    raise ValueError(f"Could not load LLM configs for provider '{LLM_PROVIDER}'")

logger = logging.getLogger(__name__)
tavily_search_tool = TavilySearchResults()

def web_search(query: str):
    logger.info("Local web_search called with %r", query)
    results, _ = tavily_search_tool._run(query)
    return results

def add(a: int, b: int) -> int:
    logger.info("Local add called with a=%s, b=%s", a, b)
    return a + b

def multiply(a: int, b: int) -> int:
    logger.info("Local multiply called with a=%s, b=%s", a, b)
    return a * b

def fibonacci(n: int) -> list[int]:
    logger.info("Local fibonacci called with n=%s", n)
    if n <= 0:
        raise ValueError("n must be a positive integer")
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

TOOLS = [
    Tool(name="web_search", func=web_search, description="Perform a web search for up-to-date information."),
    Tool(name="add", func=add, description="Add two numbers."),
    Tool(name="multiply", func=multiply, description="Multiply two numbers."),
    Tool(name="fibonacci", func=fibonacci, description="Generate the Fibonacci sequence up to n terms."),
]


# Now build your SupervisorAgent with the real MCP tools
agent = SupervisorAgent(
    llm_configs=llm_configs,
    tools=TOOLS,
    use_llm_provider=USE_LLM_PROVIDER,
)

# Expose the compiled graph for LangGraph Studio
graph = agent.state_graph