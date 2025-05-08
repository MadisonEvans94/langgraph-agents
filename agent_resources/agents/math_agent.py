# agent_resources/agents/math_agent.py

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from langchain.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from agent_resources.base_agent import Agent

logger = logging.getLogger(__name__)


class MathAgent(Agent):
    """
    Math sub-agent that handles arithmetic and Fibonacci using
    the add/multiply/fibonacci tools passed in.
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[Tool],
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "math_agent",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools
        self._build_llm_dict(llm_configs)

        self.memory = memory
        self.thread_id = thread_id or "default"

        # compile its own graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        """Compile a ReAct agent with the add/multiply/fibonacci tools."""
        llm = self.llm_dict["default_llm"]

        # Use base class to build the standard REACT prompt + tools list
        prompt = SystemMessage(content=self._build_system_prompt())

        logger.debug("MathAgent.build_graph → tools = %s", [t.name for t in self.tools])

        return create_react_agent(
            llm, 
            tools=self.tools,
            checkpointer=self.memory,
            prompt=prompt,
            name=self.name,
        )

    async def ainvoke(self, message: HumanMessage):
        """Async entrypoint; returns the AIMessage produced by this agent."""
        logger.info("MathAgent (async) received → %s", message.content)
        resp = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return resp["messages"][-1]

    def run(self, message: HumanMessage):
        """Sync convenience method."""
        return asyncio.run(self.ainvoke(message))
