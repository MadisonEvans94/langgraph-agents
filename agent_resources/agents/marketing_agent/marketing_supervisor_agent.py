# agent_resources/agents/marketing_supervisor_agent.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage, HumanMessage
from agent_resources.base_agent import Agent
from agent_resources.state_types import MarketingSupervisorState
from agent_resources.agents.marketing_agent.marketing_agent import MarketingAgent

logger = logging.getLogger(__name__)

_SUPERVISOR_PROMPT = """
You are the Marketing Supervisor.

────────────────────────────────────────────────────────────────
Your job is to decide which specialised agent should handle
the user's request given the ingested **document_text**.

• If the goal is to produce any marketing collateral
  (HTML brochure, landing-page, key points, image search, …)
  → delegate via the tool **transfer_to_marketing_agent**.

• If you can answer directly without a sub-agent, just reply.
Never perform a sub-agent's task yourself.
Always use ONE hand-off tool at a time.
"""


class MarketingSupervisorAgent(Agent):
    """
    Top-level supervisor that routes work to a pool of marketing-focused agents.
    Add more agents (e.g. SocialPostAgent, VideoAgent) to `sub_agents` as needed.
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        memory: Optional[Any] = None,
        thread_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        name: str = "marketing_supervisor",
        sub_agents: Optional[List[Agent]] = None,
        **kwargs,
    ):
        # ── basic setup ─────────────────────────────────────────
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"

        # ── adopt or create sub-agents ─────────────────────────
        self.sub_agents: List[Agent] = sub_agents or []
        if not self.sub_agents:
            raise ValueError(
                "MarketingSupervisorAgent requires at least one sub-agent."
            )

        # compile graphs for downstream use
        self.state_graph = self.build_graph()

    # --------------------------------------------------------------------- #
    #  Abstract-method implementation
    # --------------------------------------------------------------------- #
    def build_graph(self) -> StateGraph:
        """
        Construct and compile the supervisor StateGraph using `create_supervisor`.
        """
        compiled_sub_graphs = [agent.state_graph for agent in self.sub_agents]

        sg: StateGraph = create_supervisor(
            agents=compiled_sub_graphs,
            model=self.llm_dict["default_llm"],
            prompt=_SUPERVISOR_PROMPT,
            state_schema=MarketingSupervisorState,
            supervisor_name=self.name,
            output_mode="last_message",
        )
        return sg.compile(name=self.name)

    # invoke / ainvoke / stream come from Agent.base
