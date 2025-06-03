from __future__ import annotations

import logging
import json
from functools import partial
from typing import Any, Dict, List, Optional
import jinja2
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from .constants import HTML_TEMPLATE
from agent_resources.base_agent import Agent
from agent_resources.state_types import LandingPageAgentState
from agent_resources.prompts import COMBINED_ANALYSIS_PROMPT, QUERY_EXTRACTION_PROMPT, JSON_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


_TEMPLATE = jinja2.Template(HTML_TEMPLATE)


# CORE NODES

async def analysis_node(state: LandingPageAgentState, *, llm) -> Dict:
    prompt_stack = [
        SystemMessage(content=COMBINED_ANALYSIS_PROMPT),
        *state["messages"],
    ]
    resp = await llm.ainvoke(prompt_stack) if hasattr(llm, "ainvoke") else llm.invoke(prompt_stack)
    if not isinstance(resp, AIMessage):
        resp = AIMessage(content=str(resp.content))
    return {"messages": [resp], "analysis": resp.content}


async def generate_query_node(state: LandingPageAgentState, *, llm) -> Dict:
    prompt = QUERY_EXTRACTION_PROMPT.format(analysis=state["analysis"])
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content="Please provide exactly the search query."),
    ]
    resp = await llm.ainvoke(msgs) if hasattr(llm, "ainvoke") else llm.invoke(msgs)
    query = resp.content.strip()
    logger.info(f"[LandingPageAgent] generated image search query: {query}")
    return {"messages": [AIMessage(content=query)]}


async def image_search_node(state: LandingPageAgentState, *, image_tool) -> Dict:
    query_txt = state["messages"][-1].content
    result = await image_tool.ainvoke({"query": query_txt}) if hasattr(image_tool, "ainvoke") else image_tool.invoke({"query": query_txt})
    url = (
        result[0] if isinstance(result, list) and result
        else result.get("url", "") if isinstance(result, dict)
        else str(result)
    )
    logger.info(f"[LandingPageAgent] selected image URL: {url}")
    return {"messages": [AIMessage(content=f"[image_url] {url}")], "image_url": url}


async def inject_summary_node(state: LandingPageAgentState) -> Dict:
    return {"messages": [AIMessage(content=state["analysis"])]}


async def render_html_node(state: LandingPageAgentState, *, llm) -> Dict:
    analysis  = state["messages"][-1].content
    image_url = state["image_url"]

    # Ask the LLM for title, tagline, features, why—all at once
    prompt = JSON_EXTRACTION_PROMPT.format(analysis=analysis)
    resp   = await llm.ainvoke([SystemMessage(content=prompt)]) if hasattr(llm, "ainvoke") else llm.invoke([SystemMessage(content=prompt)])
    data   = json.loads(resp.content)

    html = _TEMPLATE.render(
        title    = data["title"],
        tagline  = data["tagline"],
        image_url= image_url,
        features = data["features"],
        why      = data["why"],
    )
    return {
        "messages": [AIMessage(content=html)], 
        "html": html, 
        }



# MAIN AGENT 

class LandingPageAgent(Agent):
    """
    analysis → generate_query → image_search → inject_summary
             → render_html → END
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        memory: Optional[Any] = None,
        thread_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        description: str = "Generate a responsive HTML landing page from the source document.", 
        name: str = "landing_page_agent",
        **kwargs,
    ):
        self.description = description 
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"

        self.image_tool = next((t for t in self.tools if getattr(t, "name", "") == "image_search"), None)
        if self.image_tool is None:
            raise RuntimeError("LandingPageAgent requires an 'image_search' MCP tool")

        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        sg  = StateGraph(LandingPageAgentState)

        sg.add_node("analyze_text", partial(analysis_node,    llm=llm))
        sg.add_node("generate_query", partial(generate_query_node, llm=llm))
        sg.add_node("image_search", partial(image_search_node,  image_tool=self.image_tool))
        sg.add_node("inject_summary", inject_summary_node)
        sg.add_node("render_html", partial(render_html_node,   llm=llm))

        sg.set_entry_point("analyze_text")
        sg.add_edge("analyze_text", "generate_query")
        sg.add_edge("generate_query", "image_search")
        sg.add_edge("image_search", "inject_summary")
        sg.add_edge("inject_summary","render_html")
        sg.add_edge("render_html", END)
        # TODO: add name=self.name to all 
        return sg.compile(name=self.name)

    async def ainvoke(self, messages: List[AnyMessage]):
        logger.debug(f"[LandingPageAgent] starting with {len(messages)} input message(s)")
        return await self.state_graph.ainvoke({"messages": messages})
