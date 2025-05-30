# agent_resources/agents/marketing_agent/social_post_agent.py
from __future__ import annotations
import logging, json
from typing import Dict, List, Optional, Any
from functools import partial

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END
from agent_resources.base_agent import Agent

logger = logging.getLogger(__name__)

SOCIAL_POST_PROMPT = """
You are a social-media copywriter.
Write ONE catchy post (≤280 characters) that teases the product below.
Include 2-3 relevant hashtags. Respond with only the post text.

PRODUCT DOC:
{doc}
"""

async def craft_post_node(state: Dict, *, llm) -> Dict:
    # the last user message already contains the document text
    doc_blob = state["messages"][-1].content
    prompt   = SOCIAL_POST_PROMPT.format(doc=doc_blob)
    resp     = await llm.ainvoke([SystemMessage(content=prompt)])
    # Ensure we always return an AIMessage
    if not isinstance(resp, AIMessage):
        resp = AIMessage(content=str(resp.content))
    return {"messages": [resp]}

class SocialMediaAgent(Agent):
    """
    One-step agent: craft_post → END
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: Optional[List[Any]] = None,
        description: str = "Write a tweet-length social post that teases the document.",
        name: str = "social_post_agent",
        **kwargs,
    ):
        self.description      = description
        self.name             = name
        self.tools            = tools or []
        self.use_llm_provider = kwargs.get("use_llm_provider", False)

        self._build_llm_dict(llm_configs)
        self.state_graph = self.build_graph()

    # build a *very* small graph
    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        sg  = StateGraph(dict)        
        sg.add_node("craft_post", partial(craft_post_node, llm=llm))
        sg.set_entry_point("craft_post")
        sg.add_edge("craft_post", END)
        return sg.compile(name=self.name)

    # (optional)
    async def ainvoke(self, messages: List[AnyMessage]):
        return await self.state_graph.ainvoke({"messages": messages})
