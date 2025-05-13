from __future__ import annotations
import logging
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio
from langgraph_supervisor import create_supervisor

from agent_resources.base_agent import Agent
from agent_resources.agents.supervisor.math_agent import MathAgent
from agent_resources.agents.supervisor.web_search_agent import WebSearchAgent

logger = logging.getLogger(__name__)

class SupervisorAgent(Agent):
    """
    Supervisor agent that orchestrates MathAgent and WebSearchAgent.
    """
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "supervisor_agent",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # keep original config dicts for sub-agents
        self.llm_configs = llm_configs

        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools or []
        self._build_llm_dict(llm_configs)

        self.memory = memory
        self.thread_id = thread_id or "default"

        # build and compile the supervisor's own LangGraph
        self.state_graph = self.build_graph()

    def build_graph(self):
        """
        Build a supervisor workflow over MathAgent and WebSearchAgent,
        then compile it so .ainvoke/.invoke are available.
        """
        # Split the MCP tools for each sub-agent
        math_tools = [t for t in self.tools if t.name in ("add", "multiply", "fibonacci")]
        web_tools  = [t for t in self.tools if t.name == "web_search"]

        # Instantiate sub-agents with their respective tools
        math_agent = MathAgent(
            llm_configs=self.llm_configs,
            tools=math_tools,
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="math_agent",
        )
        web_agent = WebSearchAgent(
            llm_configs=self.llm_configs,
            tools=web_tools,
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="web_search_agent",
        )

        # Supervisor LLM and prompt
        llm = self.llm_dict["default_llm"]
        prompt = SystemMessage(content=(
            "You are a supervisor agent managing two sub‐agents:\n"
            "- `math_agent`: for arithmetic and sequence calculations\n"
            "- `web_search_agent`: for up‐to‐date information and research\n"
            "Delegate each user request to the appropriate sub‐agent."
        ))

        logger.debug(
            "SupervisorAgent.build_graph → creating supervisor with subagents: %s, %s",
            math_agent.name,
            web_agent.name,
        )

        # Create the supervisor graph
        workflow = create_supervisor(
            agents=[math_agent.state_graph, web_agent.state_graph],
            model=llm,
            prompt=prompt,
        )
        # Compile it to get a graph that supports .ainvoke and .invoke
        return workflow.compile()

    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        """
        Async entrypoint: run the compiled supervisor graph and return the final AIMessage.
        """
        logger.info("SupervisorAgent (async) received → %s", message.content)
        resp = await self.state_graph.ainvoke(
            {"messages": [message]},
            config=self._default_config(),
        )
        last = resp["messages"][-1]
        if isinstance(last, AIMessage):
            return last
        # Wrap non-AIMessage in AIMessage
        return AIMessage(content=getattr(last, "content", str(last)))

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Sync convenience wrapper.
        """
        return asyncio.run(self.ainvoke(message))