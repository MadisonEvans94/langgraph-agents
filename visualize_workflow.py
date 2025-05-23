import logging
import os
import uuid
import asyncio
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from agent_resources.agent_factory import AgentFactory
from utils import get_available_agents, prompt_user_for_agent
from agent_service.app.utils import load_llm_configs

logger = logging.getLogger(__name__)
load_dotenv()
full_llm_configs = load_llm_configs(config_path="config.yaml")
# select the OpenAI configs for the visualization
llm_configs = full_llm_configs.get("openai")
if llm_configs is None:
    raise KeyError("Could not find 'openai' entry in full_llm_configs")

async def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    shared_memory = MemorySaver()
    agent_factory = AgentFactory(memory=shared_memory)

    available_agents = get_available_agents(agent_factory)
    if not available_agents:
        logger.error("No agents available in the AgentFactory.")
        print("No agents available to visualize.")
        return

    selected_agent_type = prompt_user_for_agent(available_agents)

    try:
        agent = agent_factory.factory(
            selected_agent_type,
            llm_configs=llm_configs,
            use_llm_provider=True,
            use_mcp=False
        )
        logger.info(f"Instantiated agent: {selected_agent_type}")
    except Exception as e:
        logger.error(f"Failed to instantiate agent '{selected_agent_type}': {e}", exc_info=True)
        print(f"Error: Could not instantiate agent '{selected_agent_type}'. Check logs for details.")
        return

    save_directory = os.path.dirname(__file__)
    # Derive agent folder from the agent's module path
    module_path = agent.__class__.__module__  # e.g. 'agent_resources.agents.marketing_agent.analysis_agent'
    parts = module_path.split('.')
    agent_folder = parts[-2]  # the directory under 'agents'
    agent_name = parts[-1]
    # Construct and ensure the output directory exists
    save_dir = os.path.join(save_directory, 'agent_resources', 'agents', agent_folder)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{agent_name}_workflow.png")

    agent.visualize_workflow(save_path=save_path)

if __name__ == "__main__":
    asyncio.run(main())
