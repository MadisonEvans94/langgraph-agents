import logging
from typing import Dict, Optional
from functools import partial

from agent_resources.base_agent import Agent
from agent_resources.state_types import HTMLAgentState
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

def generate_html_prompt(summary: str, image_url: str) -> str:
    return f"""You are a senior frontend UI specialist. Create a beautiful, professional product spotlight HTML page using modern design principles.

Guidelines:
- Use a centered card-style layout with a max-width.
- The title should say "Product Spotlight" and have visual hierarchy (e.g., larger, bold).
- Display the image from this URL: {image_url}, styled responsively and framed nicely.
- Include the following summary as the main description beneath the image, inside a styled section: {summary}
- Apply thoughtful spacing, typography (sans-serif), and light shadows or borders.
- Include a <style> block for all CSS. Use modern CSS with Flexbox or Grid if appropriate.
- Do NOT include any external libraries or JS (no Bootstrap, Google Fonts, etc).
- Output only the HTML, no markdown or explanation.
"""

def make_render_html_node(llm):
    async def render_html_node(state: HTMLAgentState) -> dict:
        prompt = generate_html_prompt(state["summary"], state["image_url"])
        logger.debug("Sending prompt to LLM for HTML rendering.")
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(prompt)
        else:
            response = llm.invoke(prompt)
        return {"html": response.content.strip()}
    return render_html_node

class HTMLAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id: Optional[str] = None,
        tools=None,
        name: str = "html_agent",
        **kwargs,
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        sg = StateGraph(HTMLAgentState)
        sg.add_node("render_html", make_render_html_node(llm))
        sg.set_entry_point("render_html")
        sg.add_edge("render_html", END)
        return sg.compile()

    async def ainvoke(self, summary: str, image_url: str, messages=None):
        init_state = {
            "summary": summary,
            "image_url": image_url,
            "messages": messages or [],
        }
        return await self.state_graph.ainvoke(init_state)