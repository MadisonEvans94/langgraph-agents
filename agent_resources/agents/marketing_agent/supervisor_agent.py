import logging
from typing import Dict, Optional, Any, List

from agent_resources.base_agent import Agent
from agent_resources.state_types import SupervisorAgentState
from agent_resources.agents.marketing_agent.analysis_agent import AnalysisAgent
from agent_resources.agents.marketing_agent.image_agent import ImageAgent
from agent_resources.agents.marketing_agent.html_agent import HTMLAgent

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

        # initialize subagents
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
        self.html_agent = HTMLAgent(
            llm_configs=llm_configs,
            memory=memory,
            thread_id=self.thread_id,
            tools=[],
            use_llm_provider=self.use_llm_provider,
        )

        # build and compile graph
        self.state_graph = self.build_graph()
        self.runner = self.state_graph

    def build_graph(self):
        sg = StateGraph(SupervisorAgentState)

        # Step 1: Analysis
        async def run_analysis_step(state):
            path = state.get("analysis", {}).get("path")
            if not path:
                raise ValueError("Missing 'path' in analysis state")
            messages = state.get("messages", [])
            analysis_out = await self.analysis_agent.ainvoke(path, messages)
            return {"analysis": analysis_out}

        sg.add_node("run_analysis", run_analysis_step)
        sg.set_entry_point("run_analysis")

        # Step 2: Image search
        async def run_image_step(state):
            summary = state["analysis"]["summary"]
            image_out = await self.image_agent.ainvoke(summary)
            return {
                "image_query": image_out["query"],
                "images": image_out["images"],
            }

        sg.add_node("run_image", run_image_step)
        sg.add_edge("run_analysis", "run_image")

        # Step 3: HTML generation
        async def run_html_step(state):
            summary = state["analysis"]["summary"]
            images = state.get("images", [])
            if not images:
                raise ValueError("No images available for HTML generation")
            html_out = await self.html_agent.ainvoke(summary, images[0])
            return {"html": html_out["html"]}

        sg.add_node("run_html", run_html_step)
        sg.add_edge("run_image", "run_html")

        # Step 4: Assemble final state
        def assemble(state):
            return {
                "messages": state.get("messages", []),
                "analysis": state["analysis"],
                "image_query": state["image_query"],
                "images": state["images"],
                "html": state["html"],
            }

        sg.add_node("assemble", assemble)
        sg.add_edge("run_html", "assemble")
        sg.add_edge("assemble", END)

        return sg.compile()

    async def ainvoke(self, path: str, messages=None):
        init_state = {
            "analysis": {"path": path},
            "messages": messages or [],
        }
        return await self.runner.ainvoke(init_state)