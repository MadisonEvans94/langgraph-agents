import logging
from typing import Dict, Optional
from functools import partial

from agent_resources.base_agent import Agent
from agent_resources.state_types import ImageAgentState
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from agent_resources.prompts import QUERY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

async def generate_query_node(state: ImageAgentState, llm) -> dict:
    summary = state["summary"]
    prompt = QUERY_EXTRACTION_PROMPT.format(summary=summary)
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content="Please provide exactly the search query."),
    ]
    if hasattr(llm, "ainvoke"):
        resp = await llm.ainvoke(msgs)
    else:
        resp = llm.invoke(msgs)
    query = resp.content.strip()
    logger.info(f"Generated search query: {query}")
    return {"query": query}

async def image_search_node(state: ImageAgentState, image_tool) -> dict:
    query = state["query"]
    args = {"query": query}
    if hasattr(image_tool, "ainvoke"):
        images = await image_tool.ainvoke(args)
    else:
        images = image_tool.invoke(args)
    return {"images": images}

class ImageAgent(Agent):
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
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.image_tool = next(
            (t for t in self.tools if getattr(t, "name", "") == "image_search"),
            None
        )
        if self.image_tool is None:
            raise RuntimeError("image_search tool must be provided")
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        image_tool = self.image_tool

        sg = StateGraph(ImageAgentState)
        sg.add_node("generate_query", partial(generate_query_node, llm=llm))
        sg.add_node("image_search", partial(image_search_node, image_tool=image_tool))
        sg.set_entry_point("generate_query")
        sg.add_edge("generate_query", "image_search")
        sg.add_edge("image_search", END)

        return sg.compile()

    async def ainvoke(self, summary: str):
        init_state: ImageAgentState = {
            "summary": summary,
            "query": "",
            "images": [],
            "messages": [],
        }
        return await self.state_graph.ainvoke(init_state)