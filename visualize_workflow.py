import logging
import os
import uuid
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.agent_factory import AgentFactory
from utils import get_available_agents, prompt_user_for_agent
from agent_service.app.utils import load_llm_configs
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv()
llm_configs = load_llm_configs()

def main():
    # If you still need the OpenAI key for something else, you can read it here
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Create your shared memory object
    shared_memory = MemorySaver()

    # Instantiate the AgentFactory without passing in `llm`
    agent_factory = AgentFactory(memory=shared_memory, thread_id=str(uuid.uuid4()))

    # Determine which agent types are available
    available_agents = get_available_agents(agent_factory)
    if not available_agents:
        logger.error("No agents available in the AgentFactory.")
        print("No agents available to visualize.")
        return

    # Let the user pick which agent type to visualize
    selected_agent_type = prompt_user_for_agent(available_agents)

    # Attempt to instantiate the chosen agent
    try:
        agent = agent_factory.factory(selected_agent_type, llm_configs=llm_configs)
        logger.info(f"Instantiated agent: {selected_agent_type}")
    except Exception as e:
        logger.error(
            f"Failed to instantiate agent '{selected_agent_type}': {e}", exc_info=True
        )
        print(
            f"Error: Could not instantiate agent '{selected_agent_type}'. Check logs for details."
        )
        return
    
    # Define the path where you want to save the visualization
    save_directory = os.path.dirname(__file__)
    save_path = os.path.join(
        save_directory,
        f"agent_resources/agents/{selected_agent_type}/{selected_agent_type}_workflow.png"
    )

    # Visualize and save the workflow (requires your agent to have visualize_workflow method)
    agent.visualize_workflow(save_path=save_path)

if __name__ == "__main__":
    main()
