from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from agent_resources.base_agent import Agent

logger = logging.getLogger(__name__)


class WebSearchAgent(Agent):
    """
    Web-search sub-agent that wraps the `web_search` tool to fetch up-to-date info.
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[Tool],
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "web_search_agent",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools
        self._build_llm_dict(llm_configs)

        self.memory = memory
        self.thread_id = thread_id or "default"

        # compile its own LangGraph graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        """Compile a ReAct agent graph that uses the `web_search` tool."""
        llm = self.llm_dict["default_llm"]

        # system prompt using your base-class template
        prompt = SystemMessage(content=self._build_system_prompt())

        logger.debug("WebSearchAgent.build_graph → tools = %s", [t.name for t in self.tools])

        return create_react_agent(
            llm,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=prompt,
            name=self.name,
        )

    async def ainvoke(self, message: HumanMessage):
        """Async entrypoint."""
        logger.info("WebSearchAgent (async) received → %s", message.content)
        resp = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        return resp["messages"][-1]

    def run(self, message: HumanMessage):
        """Sync convenience wrapper."""
        return asyncio.run(self.ainvoke(message))