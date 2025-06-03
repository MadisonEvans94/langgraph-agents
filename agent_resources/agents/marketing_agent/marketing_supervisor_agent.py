from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from langgraph_supervisor.handoff import (
    _normalize_agent_name,
    create_handoff_tool,
)
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage        
from agent_resources.base_agent import Agent
from agent_resources.prompts import MARKETING_SUPERVISOR_PROMPT
from agent_resources.state_types import MarketingSupervisorState  
from langgraph.graph import StateGraph, START

logger = logging.getLogger(__name__)


# helper that ALWAYS injects the PDF 
def _inject_doc(state: dict) -> dict:
    doc = state.get("document_text")
    if not doc:
        return {}                    

    # append text to the last user message (or create one)
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            m.content += f"\n\nDOCUMENT:\n{doc}"
            break
    else:
        state["messages"].append(HumanMessage(content=f"DOCUMENT:\n{doc}"))

    return {"messages": state["messages"]}   


class MarketingSupervisorAgent(Agent):
    """Routes work to whichever marketing sub-agent is appropriate."""

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
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"

        self.sub_agents = sub_agents or []
        if not self.sub_agents:
            raise ValueError("MarketingSupervisorAgent needs at least one sub-agent")

        self.state_graph = self.build_graph() 

    def _make_handoff_tools(self) -> list:
        """Create one custom hand-off tool per sub-agent, carrying its description."""
        handoff_tools = []
        for agent in self.sub_agents:
            handoff_tools.append(
                create_handoff_tool(
                    agent_name=agent.name,
                    description=agent.description or f"Handoff to {agent.name}",  
                    name=f"transfer_to_{_normalize_agent_name(agent.name)}",
                )
            )
        return handoff_tools

    def build_graph(self) -> StateGraph:
        compiled = [a.state_graph for a in self.sub_agents]

        sg = create_supervisor(
            agents=compiled,
            model=self.llm_dict["default_llm"],
            prompt=MARKETING_SUPERVISOR_PROMPT,
            tools=self._make_handoff_tools(), 
            state_schema=MarketingSupervisorState,      
            supervisor_name=self.name,
            output_mode="last_message",
        )
        # prepend the doc on every turn
        sg.add_node("_inject_doc", _inject_doc)         
        sg.add_edge(START, "_inject_doc")             
        sg.add_edge("_inject_doc", self.name)           

        return sg.compile(name=self.name)
