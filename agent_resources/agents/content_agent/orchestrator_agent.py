# agent_resources/agents/content_agent/orchestrator_agent.py

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List

from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from agent_resources.agents.content_agent.math_agent import MathAgent
from agent_resources.agents.content_agent.web_search_agent import WebSearchAgent
from agent_resources.base_agent import Agent
from agent_resources.prompts import ORCHESTRATOR_AGENT_SYSTEM_PROMPT
from agent_resources.state_types import OrchestratorState, Task

logger = logging.getLogger(__name__)

class OrchestratorAgent(Agent):
    """
    Supervisor agent that routes user queries to sub-agents and can
    drain a pre-seeded `tasks` queue via `process_tasks`.
    """

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
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self._llm_configs = llm_configs
        self.memory = memory
        self.thread_id = thread_id or "default"
        # compile the dynamic ReAct graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]

        # 1️⃣ Wrap math sub-agent as a Tool
        math_tools = [t for t in self.tools if t.name in {"add", "multiply", "fibonacci"}]
        math_agent = MathAgent(
            llm_configs={"default_llm": self._llm_configs["default_llm"]},
            tools=math_tools,
            use_llm_provider=True,
        )
        math_tool = Tool(
            name="math_agent",
            description="Delegate arithmetic or sequences to the math agent.",
            func=lambda q: asyncio.run(math_agent.ainvoke(HumanMessage(content=q))).content,
            coroutine=lambda q: math_agent.ainvoke(HumanMessage(content=q)),
        )

        # 2️⃣ Wrap web search sub-agent as a Tool
        web_tools = [t for t in self.tools if t.name == "web_search"]
        web_agent = WebSearchAgent(
            llm_configs={"default_llm": self._llm_configs["default_llm"]},
            tools=web_tools,
            use_llm_provider=self.use_llm_provider,
        )
        web_tool = Tool(
            name="web_search_agent",
            description="Perform web searches for external information.",
            func=lambda q: asyncio.run(web_agent.ainvoke(HumanMessage(content=q))).content,
            coroutine=lambda q: web_agent.ainvoke(HumanMessage(content=q)),
        )

        # 3️⃣ Replace self.tools with the wrapped tools
        self.tools = [math_tool, web_tool]

        # 4️⃣ Build tools_section and tool_catalog for the system prompt
        tools_section = "\n".join(f"• {t.name}: {t.description}" for t in self.tools)
        tool_catalog = tools_section  # or build separately if desired

        system = SystemMessage(
            content=ORCHESTRATOR_AGENT_SYSTEM_PROMPT.format(
                tools_section=tools_section,
                tool_catalog=tool_catalog,
            )
        )

        # 5️⃣ Create the ReAct agent with dynamic prompt and wrapped tools
        return create_react_agent(
            llm,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=system,
            name=self.name,
            state_schema=OrchestratorState,
        )

    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        """
        Fallback into ReAct if no tasks are pre-seeded.
        """
        out = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return out["messages"][-1]

    async def process_tasks(self, tasks: List[dict]) -> AIMessage:
        """
        Execute a list of tasks concurrently.

        • If task has 'assigned_to', use that tool.
        • Otherwise, heuristically pick math_agent vs web_search_agent.
        """
        async def _run_task(task: dict) -> str:
            desc = task["description"]

            # determine which tool to call
            if "assigned_to" in task:
                tool_name = task["assigned_to"]
            elif any(tok in desc.lower() for tok in ["add", "sum", "multiply", "+", "-", "*", "fibonacci"]) \
                 or any(char.isdigit() for char in desc):
                tool_name = "math_agent"
            else:
                tool_name = "web_search_agent"

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                return f"{task['id']}: [Error: no tool {tool_name}]"

            try:
                ai = await tool.coroutine(desc)
                result = ai.content
            except Exception as e:
                result = f"[Tool error: {e}]"

            return f"{task['id']}: {result}"

        # run all tasks in parallel and aggregate
        results = await asyncio.gather(*[_run_task(t) for t in tasks])
        return AIMessage(content="\n".join(results))

    def run(self, message: HumanMessage) -> AIMessage:
        # synchronous convenience
        return asyncio.run(self.ainvoke(message))