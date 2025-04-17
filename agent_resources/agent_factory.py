from typing import Dict, Type, Optional
from langgraph.checkpoint.memory import MemorySaver
from .agents.mcp_agent.mcp_agent import MCPAgent
from .agents.conversational_agent_with_routing.conversational_agent_with_routing import ConversationalAgentWithRouting
from agent_resources.base_agent import Agent
from .agents.conversational_agent.conversational_agent import ConversationalAgent
import uuid
from agent_resources.tools.tool_registry import ToolRegistry

class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    Handles MCP and non-MCP agents asynchronously.
    """
    def __init__(self, memory: MemorySaver):
        self.memory = memory
        self.agent_registry: Dict[str, Type[Agent]] = {
            "conversational_agent": ConversationalAgent,
            "conversational_agent_with_routing": ConversationalAgentWithRouting,
            "mcp_agent": MCPAgent,
        }

    async def factory(
        self,
        agent_type: str,
        thread_id: Optional[str] = None,
        use_openai: bool = False,
        **kwargs
    ) -> Agent:
        """
        Asynchronously create an agent instance.
        Expects an explicit `tools` list in kwargs, or falls back to the registry.
        """
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Ensure tools list is provided or fallback to default registry
        if "tools" not in kwargs:
            kwargs["tools"] = ToolRegistry.get_tools(["tavily_search"])

        # Assign persistent memory and thread ID
        thread_id = thread_id or str(uuid.uuid4())
        return agent_class(
            memory=self.memory,
            thread_id=thread_id,
            use_openai=use_openai,
            **kwargs,
        )