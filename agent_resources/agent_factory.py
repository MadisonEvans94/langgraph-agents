from typing import Dict, Type, Optional, List
from langgraph.checkpoint.memory import MemorySaver
from .agents.content_agent.orchestrator_agent import OrchestratorAgent
from .agents.react_agent.react_agent import ReactAgent
from .agents.conversational_agent_with_routing.conversational_agent_with_routing import ConversationalAgentWithRouting
from agent_resources.base_agent import Agent
from .agents.conversational_agent.conversational_agent import ConversationalAgent
import uuid
from langchain.tools import BaseTool


class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    Handles MCP and non-MCP agents.
    """
    def __init__(self, memory: MemorySaver):
        self.memory = memory
        self.agent_registry: Dict[str, Type[Agent]] = {
            "conversational_agent": ConversationalAgent,
            # "conversational_agent_with_routing": ConversationalAgentWithRouting,
            "react_agent": ReactAgent, 
            "orchestrator_agent": OrchestratorAgent
        }

    def factory(
        self,
        agent_type: str,
        thread_id: Optional[str] = None,
        use_llm_provider: bool = False,
        tools: List[BaseTool] = [],
        **kwargs
    ) -> Agent:
        """
        Create an agent instance.
        `use_llm_provider` selects whether to use the configured LLM backend.
        """
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Assign persistent memory and thread ID
        thread_id = thread_id or str(uuid.uuid4())
        return agent_class(
            memory=self.memory,
            thread_id=thread_id,
            use_llm_provider=use_llm_provider,
            tools=tools,
            **kwargs,
        )