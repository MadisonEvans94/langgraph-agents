from typing import Dict, Type, List
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from agent_resources.utils import ChatVLLMWrapper
from .agents.conversational_agent_with_routing.conversational_agent_with_routing import ConversationalAgentWithRouting
from .agents.react_agent.react_agent import ReactAgent
from .agents.conversational_agent.conversational_agent import ConversationalAgent
from .base_agent import Agent
import uuid
# vLLM / OpenAI-compatible library
from openai import OpenAI

class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    """

    def __init__(self, llm_configs: List[Dict], memory: MemorySaver, thread_id = None):
        """
        Initialize the factory with shared dependencies.

        :param llm: Language model instance.
        :param memory: Shared persistent memory.
        """
        self.memory = memory
        self.thread_id = thread_id if thread_id else str(uuid.uuid4())

        self.agent_registry: Dict[str, Type[Agent]] = {
            'react_agent': ReactAgent,
            'conversational_agent': ConversationalAgent,
            'conversational_agent_with_routing': ConversationalAgentWithRouting
        }
        self._build_llm_list(llm_configs)

    def _build_llm_list(self, llm_configs: List[Dict]) -> List[BaseChatModel]: 
        #TODO: implement function that takes list of config objects and returns list of llm runnables 
        self.llm_list = []
        for llm_config in llm_configs: 

            # Create the wrapper
            llm = ChatVLLMWrapper(
                client=OpenAI(api_key=llm_config['api_key'], base_url=llm_config['base_url']), 
                model=llm_config['params']['model_id'], 
                max_new_tokens=llm_config['params']['max_new_tokens'], 
                temperature=llm_config['params']['temperature'], 
                top_p=llm_config['params']['top_p'], 
                repetition_penalty=llm_config['params']['repetition_penalty']
            )
            self.llm_list.append(llm)

    

    def factory(self, agent_type: str, **kwargs) -> Agent:
        """
        Create an agent instance.

        :param agent_type: Type of agent to create.
        :param kwargs: Additional arguments for dynamic configuration (e.g., sub_agents).
        :return: Initialized agent instance.
        """
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_class(llm_list=self.llm_list, memory=self.memory, thread_id=self.thread_id)
