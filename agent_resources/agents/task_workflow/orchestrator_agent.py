# agent_resources/agents/task_workflow/orchestrator_agent.py
"""
OrchestratorAgent
─────────────────
• Wraps MathAgent and WebSearchAgent as LangChain Tools.
• Each sub‑agent only sees the MCP tools it needs.
• CompositeAgent (and PlanningAgent) rely on .tools, so we set it
  at construction time.
• build_graph() keeps the zero‑arg signature expected by the rest of
  the code‑base; a private _make_graph(checkpointer) does the real work.
• process_tasks() uses simple semantic rules to pick the correct tool
  and invokes it directly, avoiding “role='tool' must follow 'tool_calls'”
  API errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from agent_resources.agents.task_workflow.math_agent import MathAgent
from agent_resources.agents.task_workflow.web_search_agent import WebSearchAgent
from agent_resources.base_agent import Agent
from agent_resources.prompts import ORCHESTRATOR_AGENT_SYSTEM_PROMPT
from agent_resources.state_types import OrchestratorState, Task

logger = logging.getLogger(__name__)


class OrchestratorAgent(Agent):
    """
    Supervisor agent that routes tasks to sub‑agents.
    """

    # ───────────────────────── constructor ──────────────────────────
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[Tool] | None = None,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "orchestrator",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_llm_provider = use_llm_provider
        self._build_llm_dict(llm_configs)

        self.name = name
        self._llm_configs = llm_configs
        self.memory = memory
        self.thread_id = thread_id or "default"

        # Raw MCP tools (add, multiply, fibonacci, web_search, …)
        self._raw_tools = tools or []

        # Build the memory‑enabled graph for normal user queries
        self.state_graph = self.build_graph()

    # ───────────────────────── graph builders ───────────────────────
    def _make_graph(self, checkpointer):
        """
        Centralised graph constructor – pass checkpointer=None for
        a completely stateless version (used inside process_tasks).
        """
        llm = self.llm_dict["default_llm"]

        # Split MCP tools for each sub‑agent
        math_tools   = [t for t in self._raw_tools if t.name in {"add", "multiply", "fibonacci"}]
        search_tools = [t for t in self._raw_tools if t.name == "web_search"]

        # MathAgent tool‑wrapper
        math_agent = MathAgent(
            llm_configs=self._llm_configs,
            tools=math_tools,
            use_llm_provider=self.use_llm_provider,
            memory=None,          # stateless
        )
        math_tool = Tool(
            name="math_agent",
            description="Handle arithmetic and numerical computations.",
            func=lambda q: asyncio.run(
                math_agent.ainvoke(HumanMessage(content=q))
            ).content,
            coroutine=lambda q: math_agent.ainvoke(HumanMessage(content=q)),
        )

        # WebSearchAgent tool‑wrapper
        web_agent = WebSearchAgent(
            llm_configs=self._llm_configs,
            tools=search_tools,
            use_llm_provider=self.use_llm_provider,
            memory=None,          # stateless
        )
        web_tool = Tool(
            name="web_search_agent",
            description="Retrieve factual information and real‑time data via web search.",
            func=lambda q: asyncio.run(
                web_agent.ainvoke(HumanMessage(content=q))
            ).content,
            coroutine=lambda q: web_agent.ainvoke(HumanMessage(content=q)),
        )

        # Expose both wrapped tools
        self.tools = [math_tool, web_tool]

        system = SystemMessage(
            content=ORCHESTRATOR_AGENT_SYSTEM_PROMPT.format(
                tools_section="\n".join(f"• {t.name}: {t.description}" for t in self.tools),
                tool_catalog="\n".join(f"• {t.name}" for t in self.tools),
            )
        )

        return create_react_agent(
            llm,
            tools=self.tools,
            prompt=system,
            checkpointer=checkpointer,
            name=self.name,
            state_schema=OrchestratorState,
        )

    def build_graph(self):
        """Zero‑arg wrapper retained for external callers."""
        return self._make_graph(checkpointer=self.memory)

    # ───────────────────────── public entrypoints ───────────────────
    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        """Handle a free‑form user query (no explicit tasks)."""
        out = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return out["messages"][-1]

    async def process_tasks(self, tasks: List[Task]) -> AIMessage:
        """
        For each task, pick the correct sub‑agent tool with simple
        semantic rules and invoke it directly.
        """
        results: List[str] = []

        # Build a stateless graph once (ensures .tools is populated)
        self._make_graph(checkpointer=None)

        for task in tasks:
            desc_lc = task["description"].lower()

            # 1️⃣ explicit assignment wins
            tool_name: Optional[str] = task.get("assigned_to")

            # 2️⃣ semantic routing
            if not tool_name:
                if any(kw in desc_lc for kw in ["weather", "temperature", "forecast", "humidity", "wind"]):
                    tool_name = "web_search_agent"
                elif any(sym in desc_lc for sym in ["+", "-", "*", "×", "x", "divide", "multiply",
                                                    "add", "subtract", "fibonacci", "calculate"]):
                    tool_name = "math_agent"

            # 3️⃣ fallback to first registered tool
            if not tool_name:
                tool_name = self.tools[0].name

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                result = f"[Error: no tool named {tool_name}]"
            else:
                try:
                    ai = await tool.coroutine(task["description"])
                    result = ai.content
                except Exception as e:
                    logger.error("Tool %s error: %s", tool.name, e, exc_info=True)
                    result = f"[Tool error: {e}]"

            results.append(f"{task['id']}: {result}")

        return AIMessage(content="\n".join(results))

    # Sync helper
    def run(self, message: HumanMessage) -> AIMessage:
        return asyncio.run(self.ainvoke(message))
