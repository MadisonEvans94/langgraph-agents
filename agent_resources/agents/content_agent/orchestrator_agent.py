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
    Supervisor agent that routes user queries to sub-agents, and can
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
        # compile the fallback ReAct graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]

        # Wrap math_agent as a tool
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

        # Wrap web_search_agent as a tool
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

        # Set self.tools to the wrapped agent tools so process_tasks finds them
        self.tools = [math_tool, web_tool]

        # Build system prompt listing the two sub-tools
        tools_section = "\n".join(
            f"{i}. {t.name}: {t.description}"
            for i, t in enumerate([math_tool, web_tool], start=1)
        )
        orchestrator_prompt = SystemMessage(
            content=ORCHESTRATOR_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)
        )

        return create_react_agent(
            llm,
            tools=[math_tool, web_tool],
            checkpointer=self.memory,
            prompt=orchestrator_prompt,
            name=self.name,
            state_schema=OrchestratorState,  # declare custom state
        )

    async def ainvoke(self, message: HumanMessage):
        """
        Fallback to ReAct if no tasks are pre-seeded.
        """
        # If tasks exist in state, ReAct won't see them—use process_tasks instead.
        return (await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        ))["messages"][-1]

    async def process_tasks(self, tasks: List[dict]) -> AIMessage:
        """
        Execute tasks without using ReAct to dodge the new OpenAI tool‑call
        format.  Heuristic:
          • math keywords / digits -> math_agent
          • otherwise              -> web_search_agent
        """
        summary = []
        for task in tasks:
            desc = task["description"]

            # heuristic router
            if any(tok in desc.lower() for tok in ["add", "sum", "multiply", "+", "-", "*", "fibonacci"]) \
               or any(char.isdigit() for char in desc):
                tool_name = "math_agent"
            else:
                tool_name = "web_search_agent"

            # explicit routing still wins if provided
            if "assigned_to" in task:
                tool_name = task["assigned_to"]

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                result = f"[Error: no tool {tool_name}]"
            else:
                ai = await tool.coroutine(desc)
                result = ai.content

            summary.append(f"{task['id']}: {result}")

        return AIMessage(content="\n".join(summary))
    def run(self, message: HumanMessage):
        return asyncio.run(self.ainvoke(message))