from typing import Dict, Type, Optional
from langgraph.checkpoint.memory import MemorySaver
from .agents.mcp_agent.mcp_agent import MCPAgent
from .agents.conversational_agent_with_routing.conversational_agent_with_routing import ConversationalAgentWithRouting
from agent_resources.base_agent import Agent
from .agents.conversational_agent.conversational_agent import ConversationalAgent
import uuid
import asyncio

# Import the async MCP tool loader from langchain_mcp_adapters
from langchain_mcp_adapters.tools import load_mcp_tools

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

    async def async_factory(
        self,
        agent_type: str,
        use_openai: bool = False,
        use_mcp: bool = False,
        mcp_session: Optional[object] = None,
        **kwargs
    ) -> Agent:
        """
        Asynchronously create an agent instance.
        If use_mcp is True and an mcp_session is provided, tools are loaded from the MCP server.
        Additional kwargs (such as llm_configs) will be passed to the agent's constructor.
        """
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        if use_mcp:
            if mcp_session is None:
                raise ValueError("MCP session must be provided when use_mcp is True.")
            tools = await load_mcp_tools(mcp_session)
            kwargs["tools"] = tools

        if "tools" not in kwargs:
            from agent_resources.tools.tool_registry import ToolRegistry
            kwargs["tools"] = ToolRegistry.get_tools(['tavily_search'])

        return agent_class(
            memory=self.memory, thread_id=self.thread_id, use_openai=use_openai, **kwargs
        )

    def factory(
        self,
        agent_type: str,
        use_openai: bool = False,
        **kwargs
    ) -> Agent:
        """
        Synchronous wrapper for creating an agent instance.
        For MCP agents that need asynchronous tool loading, use async_factory.
        Additional kwargs (such as llm_configs) will be passed to the agent's constructor.
        """
        if kwargs.get("use_mcp", False):
            return asyncio.run(self.async_factory(agent_type, use_openai, **kwargs))
        else:
            from agent_resources.tools.tool_registry import ToolRegistry
            if "tools" not in kwargs:
                kwargs["tools"] = ToolRegistry.get_tools(['tavily_search'])
            agent_class = self.agent_registry.get(agent_type)
            if agent_class is None:
                raise ValueError(f"Unknown agent type: {agent_type}")
            return agent_class(
                memory=self.memory, thread_id=self.thread_id, use_openai=use_openai, **kwargs
            )
