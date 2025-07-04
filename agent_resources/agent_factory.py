# agent_resources/agent_factory.py

from typing import Dict, Type, Optional, List
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool

# from .agents.content_agent.composite_agent import CompositeAgent
# from .agents.content_agent.orchestrator_agent import OrchestratorAgent
from .agents.supervisor.supervisor import SupervisorAgent      # ← newly added
# from .agents.content_agent.planning_agent import PlanningAgent
from .agents.react_agent.react_agent import ReactAgent
from .agents.conversational_agent_with_routing.conversational_agent_with_routing import ConversationalAgentWithRouting
from .agents.conversational_agent.conversational_agent import ConversationalAgent

from agent_resources.base_agent import Agent

class AgentFactory:
    """
    Factory for creating named agents with shared memory and tool lists.
    """
    def __init__(self, memory: MemorySaver):
        self.memory = memory
        self.agent_registry: Dict[str, Type[Agent]] = {
            "conversational_agent": ConversationalAgent,
            "react_agent": ReactAgent,
            # "orchestrator_agent": OrchestratorAgent,
            "supervisor_agent": SupervisorAgent,  # ← newly registered
            # "composite_agent": CompositeAgent,
            # "planning_agent": PlanningAgent,
        }

    def factory(
        self,
        agent_type: str,
        thread_id: Optional[str] = None,
        use_llm_provider: bool = False,
        tools: List[BaseTool] = [],
        **kwargs
    ) -> Agent:
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        thread_id = thread_id or str(uuid.uuid4())
        return agent_class(
            memory=self.memory,
            thread_id=thread_id,
            use_llm_provider=use_llm_provider,
            tools=tools,
            **kwargs,
        )