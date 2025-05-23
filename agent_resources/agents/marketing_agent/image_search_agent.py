# agent_resources/agents/marketing_agent/image_search_agent.py

import logging
from functools import partial
from typing import Dict, Optional

from agent_resources.base_agent import Agent
from agent_resources.state_types import ImageSearchAgentState
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

async def generate_query_node(state: ImageSearchAgentState, llm) -> dict:
    """
    Take the incoming summary paragraph and ask the LLM
    for a maximally-three-word, highly specific search query.
    """
    summary = state["summary"]
    prompt = (
        "You are a specialized query extraction assistant. "
        "Given the paragraph below, produce a concise, three-word or fewer search query "
        "that best captures the core product described. Respond with only the query:\n\n"
        f"{summary}"
    )
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content="Please provide exactly the search query."),
    ]
    # invoke appropriately
    if hasattr(llm, "ainvoke"):
        resp = await llm.ainvoke(msgs)
    else:
        resp = llm.invoke(msgs)
    query = resp.content.strip()
    return {"query": query}


async def image_search_node(state: ImageSearchAgentState, image_tool) -> dict:
    """
    Take the generated 'query' and run your Unsplash image_search tool.
    """
    query = state["query"]
    args = {"query": query}
    if hasattr(image_tool, "ainvoke"):
        images = await image_tool.ainvoke(args)
    else:
        images = image_tool.invoke(args)
    return {"images": images}


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
        self.tools = tools or []
        # build and store your LLMs
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()
        self.runner = self.state_graph

    def build_graph(self):
        # bind your default LLM and the single image_search tool
        llm = self.llm_dict["default_llm"]
        image_tool = self.tools[0]

        sg = StateGraph(ImageSearchAgentState)

        # register nodes as direct async callables returning dict
        sg.add_node(
            "generate_query",
            partial(generate_query_node, llm=llm),
        )
        sg.add_node(
            "image_search",
            partial(image_search_node, image_tool=image_tool),
        )

        sg.set_entry_point("generate_query")
        sg.add_edge("generate_query", "image_search")
        sg.add_edge("image_search", END)

        return sg.compile()

    async def ainvoke(self, summary: str):
        """
        Entry point: feed in your summary paragraph,
        then run query → search.
        """
        init_state: ImageSearchAgentState = {
            "summary": summary,
            "query": "",
            "images": [],
            "messages": [],
        }
        return await self.runner.ainvoke(init_state)