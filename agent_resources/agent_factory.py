from typing import Dict, Type
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from .agents.conversational_agent_with_routing.conversational_agent_with_routing import ConversationalAgentWithRouting
from .agents.react_agent.react_agent import ReactAgent
from .agents.conversational_agent.conversational_agent import ConversationalAgent
from .base_agent import Agent
import uuid

class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    """

    def __init__(self, llm: ChatOpenAI, memory: MemorySaver):
        """
        Initialize the factory with shared dependencies.

        :param llm: Language model instance.
        :param memory: Shared persistent memory.
        """
        self.llm = llm
        self.memory = memory
        self.thread_id = str(uuid.uuid4())

        self.agent_registry: Dict[str, Type[Agent]] = {
            'react_agent': ReactAgent,
            'conversational_agent': ConversationalAgent,
            'conversational_agent_with_routing': ConversationalAgentWithRouting
        }

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

        return agent_class(llm=self.llm, memory=self.memory, thread_id=self.thread_id)
