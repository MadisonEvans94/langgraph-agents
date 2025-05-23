import logging
from typing import Dict, List, Optional

from agent_resources.base_agent import Agent
from agent_resources.state_types import MarketingAgentState
from langgraph.graph import StateGraph, START, END

from agent_resources.agents.marketing_agent.analysis_agent_nodes import (
    extract_pdf_node,
    summarise_node,
    extract_key_points_node,
    detect_domain_node,
)

logger = logging.getLogger(__name__)

class AnalysisAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id: Optional[str] = None,
        tools=None,
        name: str = "analysis_agent",
        **kwargs,
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()
        self.runner = self.state_graph

    def build_graph(self):

        llm = self.llm_dict["default_llm"]

        sg = StateGraph(MarketingAgentState)
        # Nodes
        sg.add_node("extract_pdf", extract_pdf_node)
        sg.add_node("summarise", lambda state: summarise_node(state, llm))
        sg.add_node("extract_key_points", lambda state: extract_key_points_node(state, llm))
        sg.add_node("detect_domain", lambda state: detect_domain_node(state, llm))

        # Edges
        sg.add_edge(START, "extract_pdf")
        sg.add_edge("extract_pdf", "summarise")

        # fan-out after summarise
        sg.add_edge("summarise", "extract_key_points")
        sg.add_edge("summarise", "detect_domain")

        # join to END
        sg.add_edge("extract_key_points", END)
        sg.add_edge("detect_domain", END)

        return sg.compile()

    async def ainvoke(self, path: str, messages=None):
        """
        Run the graph and return the final state (which includes 'summary',
        'key_points', and 'domain').
        """
        init_state = {"path": path, "messages": messages or []}
        return await self.runner.ainvoke(init_state)