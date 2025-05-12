# agent_resources/agents/content_agent/orchestrator_agent.py

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List

from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent_resources.agents.content_agent.math_agent import MathAgent
from agent_resources.agents.content_agent.web_search_agent import WebSearchAgent
from agent_resources.base_agent import Agent
from agent_resources.prompts import ORCHESTRATOR_AGENT_SYSTEM_PROMPT
from agent_resources.state_types import OrchestratorState

logger = logging.getLogger(__name__)


class OrchestratorAgent(Agent):
    """
    Supervisor agent that routes user queries to domain-specific sub-agents:
      - math_agent for arithmetic and sequences
      - web_search_agent for up-to-date information
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
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self._llm_configs = llm_configs
        self.memory = memory
        self.thread_id = thread_id or "default"

        # compile its own graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]

        # split out the incoming low-level tools into math vs web
        math_tools = [t for t in self.tools if t.name in {"add", "multiply", "fibonacci"}]
        web_tools  = [t for t in self.tools if t.name == "web_search"]

        # wrap each sub-agent as a Tool
        math_agent = MathAgent(
            llm_configs={"default_llm": self._llm_configs["default_llm"]},
            tools=math_tools,
            use_llm_provider=True,
        )
        math_tool = Tool(
            name="math_agent",
            description="Delegate arithmetic or sequence computations to a specialized math agent.",
            func=lambda q: asyncio.run(math_agent.ainvoke(HumanMessage(content=q))).content,
            coroutine=lambda q: math_agent.ainvoke(HumanMessage(content=q)),
        )

        web_agent = WebSearchAgent(
            llm_configs={"default_llm": self._llm_configs["default_llm"]},
            tools=web_tools,
            use_llm_provider=self.use_llm_provider,
        )
        web_tool = Tool(
            name="web_search_agent",
            description="Perform web searches to retrieve real-time or external information.",
            func=lambda q: asyncio.run(web_agent.ainvoke(HumanMessage(content=q))).content,
            coroutine=lambda q: web_agent.ainvoke(HumanMessage(content=q)),
        )

        # build the orchestrator system prompt from our new constant
        tools_section = "\n".join(
            f"{i}. {tool.name}: {tool.description}"
            for i, tool in enumerate([math_tool, web_tool], start=1)
        )
        orchestrator_prompt = SystemMessage(
            content=ORCHESTRATOR_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)
        )

        return create_react_agent(
            llm,
            tools=[math_tool, web_tool],
            checkpointer=self.memory,
            prompt=orchestrator_prompt,
            name=self.name,
            state_schema=OrchestratorState,       # ← use your custom state
        )

    async def ainvoke(self, message: HumanMessage):
        logger.info("Orchestrator (async) received → %s", message.content)
        resp = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return resp["messages"][-1]

    def run(self, message: HumanMessage):
        """Sync convenience wrapper."""
        return asyncio.run(self.ainvoke(message))