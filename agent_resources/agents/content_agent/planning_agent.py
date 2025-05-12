# agent_resources/agents/content_agent/planning_agent.py
from __future__ import annotations
import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent

from agent_resources.base_agent import Agent
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from agent_resources.state_types import OrchestratorState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #
PLANNER_PROMPT = REACT_AGENT_SYSTEM_PROMPT + """
You are a planning assistant.

• If the user's query can be answered with a direct answer, do so and finish.

• If the query requires multiple steps or external tool calls, OUTPUT ONLY a
  JSON array of task objects with these keys:

  - "id":        a string starting at "1"
  - "description": an imperative sentence describing the task

DO NOT include an "assigned_to" field.
"""

# --------------------------------------------------------------------------- #
# Agent class
# --------------------------------------------------------------------------- #
class PlanningAgent(Agent):
    """
    Either returns a direct answer, or emits a JSON task list that the
    Orchestrator can later route to sub‑agents.
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
        # NOTE: the base Agent.__init__ (via super()) does NOT expect extra kwargs.
        # We therefore call it with **kwargs (empty by default) like the other agents.
        super().__init__(**kwargs)

        # ---- custom init ----
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools or []          # planner uses no external tools, but keep field
        self._build_llm_dict(llm_configs) # sets self.llm_dict
        self.memory = memory
        self.thread_id = thread_id or "default"

        self.state_graph = self.build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_system_prompt(self) -> str:
        return PLANNER_PROMPT

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        prompt = SystemMessage(content=self._build_system_prompt())
        return create_react_agent(
            llm,
            tools=[],                 # planner itself calls no tools
            prompt=prompt,
            checkpointer=self.memory,
            name=self.name,
            state_schema=OrchestratorState,  # keeps messages if caller cares
        )

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
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