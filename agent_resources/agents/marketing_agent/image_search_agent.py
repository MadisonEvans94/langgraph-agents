import logging
from typing import Dict, Optional

from agent_resources.base_agent import Agent
from agent_resources.state_types import ImageSearchAgentState
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

class ImageSearchAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id: Optional[str] = None,
        tools=None,
        name: str = "image_search_agent",
        **kwargs,
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        # we expect exactly one tool here: the image_search tool
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()
        self.runner = self.state_graph

    def build_graph(self):
        sg = StateGraph(ImageSearchAgentState)

        async def image_search_node(state: ImageSearchAgentState) -> dict:
            # 'query' carries the raw query string
            query = state["query"]
            image_tool = self.tools[0]  # just the one image_search tool
            if hasattr(image_tool, "ainvoke"):
                images = await image_tool.ainvoke({"query": query})
            else:
                images = image_tool.invoke({"query": query})
            return {"images": images}

        sg.add_node("image_search", image_search_node)
        sg.set_entry_point("image_search")
        sg.add_edge("image_search", END)

        return sg.compile()

    async def ainvoke(self, query: str):
        init_state: ImageSearchAgentState = {"query": query, "images": [], "messages": []}
        return await self.runner.ainvoke(init_state)