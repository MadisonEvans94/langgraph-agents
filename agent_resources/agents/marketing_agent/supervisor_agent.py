# agent_resources/agents/marketing_agent/supervisor_agent.py

import logging
from typing import Dict, Optional, Any, List

from agent_resources.base_agent import Agent
from agent_resources.state_types import SupervisorAgentState
from agent_resources.agents.marketing_agent.analysis_agent import AnalysisAgent
from agent_resources.agents.marketing_agent.image_agent import ImageAgent

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

class SupervisorAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        name: str = "supervisor_agent",
        **kwargs,
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"

        # find the single image_search MCP tool
        image_tools = [t for t in self.tools if getattr(t, "name", "") == "image_search"]
        if not image_tools:
            raise RuntimeError("Supervisor requires the 'image_search' MCP tool")

        # build our two workers
        self.analysis_agent = AnalysisAgent(
            llm_configs=llm_configs,
            memory=memory,
            thread_id=self.thread_id,
            tools=[],
            use_llm_provider=self.use_llm_provider,
        )
        self.image_agent = ImageAgent(
            llm_configs=llm_configs,
            memory=memory,
            thread_id=self.thread_id,
            tools=image_tools,
            use_llm_provider=self.use_llm_provider,
        )

        # now build & compile the graph
        self.state_graph = self.build_graph()
        self.runner = self.state_graph

    def build_graph(self):
        sg = StateGraph(SupervisorAgentState)

        # 1) run the analysis_agent on the PDF path → put its full state under "analysis"
        async def run_analysis_step(state):
            path = state.get("analysis", {}).get("path")
            if not path:
                raise ValueError("Missing 'path' in analysis state")
            messages = state.get("messages", [])
            analysis_out = await self.analysis_agent.ainvoke(path, messages)
            return {"analysis": analysis_out}

        sg.add_node("run_analysis", run_analysis_step)
        sg.set_entry_point("run_analysis")

        # 2) run the image_agent on the summary we just extracted
        async def run_image_step(state):
            summary = state["analysis"]["summary"]
            image_out = await self.image_agent.ainvoke(summary)
            return {
                "image_query": image_out["query"],
                "images": image_out["images"],
            }

        sg.add_node("run_image", run_image_step)
        sg.add_edge("run_analysis", "run_image")

        # 3) assemble a final payload
        def assemble(state):
            return {
                "analysis": state["analysis"],
                "image_query": state["image_query"],
                "images": state["images"],
                "messages": state.get("messages", []),
            }

        sg.add_node("assemble", assemble)
        sg.add_edge("run_image", "assemble")
        sg.add_edge("assemble", END)

        return sg.compile()

    async def ainvoke(self, path: str, messages=None):
        init_state = {
            "analysis": {"path": path},
            "messages": messages or [],
        }
        return await self.runner.ainvoke(init_state)