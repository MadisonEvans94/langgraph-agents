# agent_resources/agents/content_agent/planning_agent.py

from __future__ import annotations
import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent

from agent_resources.base_agent import Agent
from agent_resources.prompts import PLANNING_AGENT_SYSTEM_PROMPT
from agent_resources.state_types import OrchestratorState

logger = logging.getLogger(__name__)

class PlanningAgent(Agent):
    """
    Agent that either answers directly or emits a JSON task list for decomposition.
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[Tool] | None = None,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "planning_agent",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()

    def _build_system_prompt(self) -> str:
        # Build dynamic lists for both placeholders
        tools_section = "\n".join(
            f"• {t.name}: {t.description or 'no description'}"
            for t in self.tools
        ) or "• (no tools registered)"

        tool_catalog = "\n".join(
            f"• {t.name}"
            for t in self.tools
        ) or "• (no tools registered)"

        return PLANNING_AGENT_SYSTEM_PROMPT.format(
            tools_section=tools_section,
            tool_catalog=tool_catalog,
        )

    def build_graph(self):
        llm = self.llm_dict["alternate_llm"]
        prompt = SystemMessage(content=self._build_system_prompt())
        return create_react_agent(
            llm,
            tools=[],
            prompt=prompt,
            checkpointer=self.memory,
            name=self.name,
            state_schema=OrchestratorState,
        )

    async def ainvoke(self, message: HumanMessage):
        resp = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return resp["messages"][-1]

    def run(self, message: HumanMessage):
        out = self.state_graph.invoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return out["messages"][-1]