# agent_resources/agents/marketing_agent/supervisor_agent.py
from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, List, Optional

import jinja2
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent_resources.base_agent import Agent
from agent_resources.state_types import MarketingAgentState
from agent_resources.prompts import COMBINED_ANALYSIS_PROMPT, QUERY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# ── CORE NODES ────────────────────────────────────────

async def analysis_node(state: MarketingAgentState, *, llm) -> Dict:
    prompt_stack = [
        SystemMessage(content=COMBINED_ANALYSIS_PROMPT),
        *state["messages"],
    ]
    resp = await llm.ainvoke(prompt_stack) if hasattr(llm, "ainvoke") else llm.invoke(prompt_stack)
    if not isinstance(resp, AIMessage):
        resp = AIMessage(content=str(resp.content))
    return {
        "messages": [resp],
        "analysis": resp.content,
    }

async def generate_query_node(state: MarketingAgentState, *, llm) -> Dict:
    prompt = QUERY_EXTRACTION_PROMPT.format(analysis=state["analysis"])
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content="Please provide exactly the search query."),
    ]
    resp = await llm.ainvoke(msgs) if hasattr(llm, "ainvoke") else llm.invoke(msgs)
    query = resp.content.strip()
    logger.info(f"[MarketingAgent] generated image search query: {query}")
    return {"messages": [AIMessage(content=query)]}

async def image_search_node(state: MarketingAgentState, *, image_tool) -> Dict:
    query_txt = state["messages"][-1].content
    result = await image_tool.ainvoke({"query": query_txt}) if hasattr(image_tool, "ainvoke") else image_tool.invoke({"query": query_txt})
    url = (
        result[0] if isinstance(result, list) and result
        else result.get("url", "") if isinstance(result, dict)
        else str(result)
    )
    logger.info(f"[MarketingAgent] selected image URL: {url}")
    return {
        "messages": [AIMessage(content=f"[image_url] {url}")],
        "image_url": url,
    }

async def inject_summary_node(state: MarketingAgentState) -> Dict:
    # Push the analysis content back into messages so our render step can reuse it
    return {"messages": [AIMessage(content=state["analysis"])]}


# ── DETERMINISTIC HTML TEMPLATE ──────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f9f9f9; }
    .container { max-width: 800px; margin: 50px auto; padding: 20px; background-color: #fff; border-radius: 8px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .header { display: flex; flex-wrap: wrap; align-items: center; gap: 20px; margin-bottom: 30px; }
    .hero-text { flex: 1; }
    .hero-text .tagline { font-size: 1.25rem; color: #555; margin-top: 8px; }
    .hero-image { flex: 1; text-align: center; }
    .hero-image img { max-width: 100%; border-radius: 8px; }
    .features { display: flex; justify-content: space-between; margin: 30px 0; }
    .features ul { display: flex; flex: 1; gap: 20px; padding: 0; margin: 0; }
    .feature-item { list-style: none; flex: 1; text-align: center; }
    .feature-item::before { content: '•'; display: block; font-size: 2rem; color: #007BFF; margin-bottom: 8px; }
    .why-section { background-color: #f1f1f1; padding: 20px; border-radius: 8px; }
    .why-section h2 { margin-top: 0; font-size: 1.5rem; }
    .cta-button { display: inline-block; margin-top: 20px; padding: 10px 20px;
                  background-color: #007BFF; color: #fff; text-decoration: none; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <section class="header">
      <div class="hero-text">
        <h1 class="title">{{ title }}</h1>
        <p class="tagline">{{ tagline }}</p>
      </div>
      <div class="hero-image">
        <img src="{{ image_url }}" alt="{{ title }} Console">
      </div>
    </section>
    <section class="features">
      <ul>
      {% for feat in features %}
        <li class="feature-item">{{ feat }}</li>
      {% endfor %}
      </ul>
    </section>
    <section class="why-section">
      <h2>Why {{ title }}?</h2>
      <p>{{ why }}</p>
    </section>
  </div>
</body>
</html>"""

async def render_html_node(state: MarketingAgentState, *, llm) -> Dict:
    analysis = state["messages"][-1].content
    image_url = state["image_url"]

    # 1) Generate tagline
    tag_resp = await llm.ainvoke([
        SystemMessage(content=(
            "From this analysis, write a punchy marketing tagline (≤12 words). "
            f"\n\nANALYSIS:\n{analysis}"
        ))
    ])
    tagline = tag_resp.content.strip()

    # 2) Generate features
    feat_resp = await llm.ainvoke([
        SystemMessage(content=(
            "List exactly three feature phrases (≤5 words each), comma-separated. "
            f"\n\nANALYSIS:\n{analysis}"
        ))
    ])
    features = [f.strip() for f in feat_resp.content.split(",")]

    # 3) Generate “why” paragraph
    why_resp = await llm.ainvoke([
        SystemMessage(content=(
            "Write a 2–3 sentence paragraph starting “Why PlayStation 5?” "
            f"that sells this product based on the analysis.\n\nANALYSIS:\n{analysis}"
        ))
    ])
    why = why_resp.content.strip()

    # 4) Render final HTML
    html = jinja2.Template(TEMPLATE).render(
        title="PlayStation 5",
        tagline=tagline,
        image_url=image_url,
        features=features,
        why=why,
    )

    return {"messages": [AIMessage(content=html)]}


# ── MAIN AGENT ────────────────────────────────────────

class MarketingAgent(Agent):
    """
    analysis → generate_query → image_search → inject_summary → render_html → END
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        memory: Optional[Any] = None,
        thread_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        name: str = "marketing_supervisor_agent",
        **kwargs,
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"

        self.image_tool = next((t for t in self.tools if getattr(t, "name", "") == "image_search"), None)
        if not self.image_tool:
            raise RuntimeError("MarketingAgent requires an 'image_search' MCP tool")

        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        sg = StateGraph(MarketingAgentState)

        sg.add_node("analyze_text",   partial(analysis_node, llm=llm))
        sg.add_node("generate_query", partial(generate_query_node, llm=llm))
        sg.add_node("image_search",   partial(image_search_node, image_tool=self.image_tool))
        sg.add_node("inject_summary", inject_summary_node)
        sg.add_node("render_html",    partial(render_html_node, llm=llm))

        sg.set_entry_point("analyze_text")
        sg.add_edge("analyze_text",   "generate_query")
        sg.add_edge("generate_query", "image_search")
        sg.add_edge("image_search",   "inject_summary")
        sg.add_edge("inject_summary","render_html")
        sg.add_edge("render_html",     END)

        return sg.compile()

    async def ainvoke(self, messages: List[AnyMessage]):
        logger.debug(f"[MarketingAgent] starting with {len(messages)} input message(s)")
        return await self.state_graph.ainvoke({"messages": messages})
