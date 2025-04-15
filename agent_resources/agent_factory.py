# agent_factory.py
from typing import Dict, Type
from langgraph.checkpoint.memory import MemorySaver

from .agents.mcp_agent.mcp_agent import MCPAgent
from .agents.conversational_agent_with_routing.conversational_agent_with_routing import (
    ConversationalAgentWithRouting,
)
from agent_resources.base_agent import Agent
from .agents.conversational_agent.conversational_agent import ConversationalAgent
import uuid


class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    """

    def __init__(self, memory: MemorySaver, thread_id: str = None):
        self.memory = memory
        self.thread_id = thread_id if thread_id else str(uuid.uuid4())

        self.agent_registry: Dict[str, Type[Agent]] = {
            "conversational_agent": ConversationalAgent,
            "conversational_agent_with_routing": ConversationalAgentWithRouting,
            "mcp_agent": MCPAgent
        }

    def factory(
        self,
        agent_type: str,
        use_openai: bool = False,
        **kwargs
    ) -> Agent:
        """
        Create an agent instance.
        Additional kwargs (such as llm_configs) will be passed to the agent's constructor.
        """
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Pass the shared memory, thread_id, use_openai flag, and any additional configuration to the agent.
        return agent_class(
            memory=self.memory, thread_id=self.thread_id, use_openai=use_openai, **kwargs
        )
