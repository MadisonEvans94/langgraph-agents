# visualize_workflow.py
import logging
import os
import asyncio
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from agent_resources.agent_factory import AgentFactory
from utils import get_available_agents, prompt_user_for_agent
from agent_service.app.utils import load_llm_configs


# ────────────────────────────────────────────────────────────
# Dummy stand-in for the real `image_search` MCP tool
# ────────────────────────────────────────────────────────────
class DummyImageSearchTool(BaseTool):
    """
    Stub used only for visualising the MarketingAgent’s graph.
    Returns a placeholder URL instead of hitting a real backend.
    """
    name: str = "image_search"
    description: str = "Dummy image search (visualisation-only)."

    async def _arun(self, query: str) -> Dict[str, str]:
        return {"url": "https://dummy.local/image.jpg"}

    def _run(self, query: str) -> Dict[str, str]:
        return {"url": "https://dummy.local/image.jpg"}


# ────────────────────────────────────────────────────────────
# Setup + config
# ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
load_dotenv()

full_llm_configs = load_llm_configs(config_path="config.yaml")
llm_configs = full_llm_configs.get("openai")
if llm_configs is None:
    raise KeyError("Could not find 'openai' entry in full_llm_configs")


# ────────────────────────────────────────────────────────────
# Main visualisation routine
# ────────────────────────────────────────────────────────────
async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    shared_memory = MemorySaver()
    agent_factory = AgentFactory(memory=shared_memory)

    available_agents = get_available_agents(agent_factory)
    if not available_agents:
        logger.error("No agents available in the AgentFactory.")
        print("No agents available to visualise.")
        return

    selected_agent_type = prompt_user_for_agent(available_agents)

    # Inject stub tools ONLY when we need them for a graph render
    extra_tools: List[BaseTool] = []
    if selected_agent_type == "marketing_agent":
        extra_tools.append(DummyImageSearchTool())

    try:
        agent = agent_factory.factory(
            agent_type=selected_agent_type,
            llm_configs=llm_configs,
            use_llm_provider=True,
            use_mcp=False,          # <─ we don’t have a live MCP session here
            tools=extra_tools,      # <─ stub goes in here
        )
        logger.info(f"Instantiated agent: {selected_agent_type}")
    except Exception as e:
        logger.error(
            f"Failed to instantiate agent '{selected_agent_type}': {e}",
            exc_info=True,
        )
        print(
            f"Error: Could not instantiate agent '{selected_agent_type}'. "
            "Check logs for details."
        )
        return

    save_directory = os.path.dirname(__file__)
    save_path = os.path.join(
        save_directory,
        f"agent_resources/agents/{selected_agent_type}/"
        f"{selected_agent_type}_workflow.png",
    )

    agent.visualize_workflow(save_path=save_path)
    print(f"✅ Workflow visualisation saved to {save_path}")


if __name__ == "__main__":
    asyncio.run(main())
