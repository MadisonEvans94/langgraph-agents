# agent_resources/agents/content_agent/orchestrator_agent.py

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional

from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from agent_resources.agents.content_agent.math_agent import MathAgent
from agent_resources.agents.content_agent.web_search_agent import WebSearchAgent
from agent_resources.base_agent import Agent
from agent_resources.prompts import ORCHESTRATOR_AGENT_SYSTEM_PROMPT
from agent_resources.state_types import OrchestratorState, Task

logger = logging.getLogger(__name__)

class OrchestratorAgent(Agent):
    """
    Supervisor agent that routes user queries to sub-agents and can
    drain a pre-seeded `tasks` queue via `process_tasks`.
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[Tool] | None = None,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "orchestrator",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Ensure use_llm_provider is set before building the LLM dict
        self.use_llm_provider = use_llm_provider
        self._build_llm_dict(llm_configs)
        self.name = name
        self._llm_configs = llm_configs
        self.memory = memory
        self.thread_id = thread_id or "default"

        # Raw MCP tools; we'll wrap these next
        self._raw_tools = tools or []

        # Build the dynamic ReAct graph and wrap tools
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]

        # Wrap math sub-agent as a Tool
        math_agent = MathAgent(
            llm_configs=self._llm_configs,
            tools=self._raw_tools,
            use_llm_provider=self.use_llm_provider,
        )
        math_tool = Tool(
            name="math_agent",
            description="Handle arithmetic and numerical computations.",
            func=lambda q: asyncio.run(math_agent.ainvoke(HumanMessage(content=q))).content,
            coroutine=lambda q: math_agent.ainvoke(HumanMessage(content=q)),
        )

        # Wrap web-search sub-agent as a Tool
        web_agent = WebSearchAgent(
            llm_configs=self._llm_configs,
            tools=self._raw_tools,
            use_llm_provider=self.use_llm_provider,
        )
        web_tool = Tool(
            name="web_search_agent",
            description="Retrieve factual information via web search.",
            func=lambda q: asyncio.run(web_agent.ainvoke(HumanMessage(content=q))).content,
            coroutine=lambda q: web_agent.ainvoke(HumanMessage(content=q)),
        )

        # Replace self.tools with the wrapped tools
        self.tools = [math_tool, web_tool]

        # Build dynamic prompt sections
        tools_section = "\n".join(f"• {t.name}: {t.description}" for t in self.tools)
        tool_catalog = tools_section

        system = SystemMessage(
            content=ORCHESTRATOR_AGENT_SYSTEM_PROMPT.format(
                tools_section=tools_section,
                tool_catalog=tool_catalog,
            )
        )

        return create_react_agent(
            llm,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=system,
            name=self.name,
            state_schema=OrchestratorState,
        )

    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        """
        Default React-based invocation when no pre-defined tasks are provided.
        """
        out = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return out["messages"][-1]

    async def process_tasks(self, tasks: List[Task]) -> AIMessage:
        """
        Execute a list of Task dicts by directly invoking the matching tool.

        - If 'assigned_to' is present, use that tool.
        - Otherwise, attempt to match based on tool names.
        """
        results: List[str] = []

        for task in tasks:
            desc = task["description"]
            # Determine tool name:
            tool_name: Optional[str] = task.get("assigned_to")
            if not tool_name:
                # Fallback: pick first tool whose name appears in the description
                for t in self.tools:
                    if t.name in desc.lower():
                        tool_name = t.name
                        break

            # Default to the first tool if still unresolved
            if not tool_name and self.tools:
                tool_name = self.tools[0].name

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                result = f"[Error: no tool named {tool_name}]"
            else:
                try:
                    ai = await tool.coroutine(desc)
                    result = ai.content
                except Exception as e:
                    result = f"[Tool error: {e}]"

            results.append(f"{task['id']}: {result}")

        return AIMessage(content="\n".join(results))

    def run(self, message: HumanMessage) -> AIMessage:
        # synchronous convenience
        return asyncio.run(self.ainvoke(message))