# agent_resources/agents/orchestrator_agent.py

import logging
import asyncio
import concurrent.futures
import threading
from queue import Queue
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from agent_resources.base_agent import Agent

logger = logging.getLogger(__name__)

class OrchestratorAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id=None,
        tools=None,
        use_llm_provider = False, 
        name="orchestrator",
        **kwargs
    ):
        self.name = name
        self.use_llm_provider = use_llm_provider
        # load all MCP-wrapped tools
        self.tools = tools or []
        # build your LLM dict just like ReactAgent does
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        # compile the orchestrator graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]

        # split out your two tool-sets
        math_tools = [t for t in self.tools if t.name in {"add", "multiply", "fibonacci"}]
        web_tools  = [t for t in self.tools if t.name == "web_search"]

        logger.info(
            "OrchestratorAgent.build_graph: %d math tools (%s), %d web tools (%s)",
            len(math_tools), [t.name for t in math_tools],
            len(web_tools),  [t.name for t in web_tools],
        )

        # prompts for sub-agents
        math_prompt = SystemMessage(
            content="You are a math assistant. Use the add, multiply, and fibonacci tools to answer math queries."
        )
        web_prompt = SystemMessage(
            content="You are a web search assistant. Use the web_search tool to fetch up-to-date information."
        )

        # build sub-agents
        math_agent = create_react_agent(
            llm,
            tools=math_tools,
            checkpointer=None,
            prompt=math_prompt,
            name="math_agent",
        )
        web_agent = create_react_agent(
            llm,
            tools=web_tools,
            checkpointer=None,
            prompt=web_prompt,
            name="web_search_agent",
        )

        # wrap each sub-agent as a LangChain Tool
        def make_tool(name, agent, description):
            def tool_fn(query: str):
                logger.info("OrchestratorAgent: routing to %s with query → %s", name, query)
                result = self._call_subagent(agent, query)
                logger.info("OrchestratorAgent: %s returned → %s", name, result)
                return result
            return Tool(name=name, func=tool_fn, description=description)

        math_agent_tool = make_tool(
            "math_agent",
            math_agent,
            "Delegate to the math sub-agent for arithmetic or sequence computations."
        )
        web_agent_tool = make_tool(
            "web_search_agent",
            web_agent,
            "Delegate to the web search sub-agent for queries needing external data."
        )

        # orchestrator’s own system prompt
        orchestrator_prompt = SystemMessage(
            content=(
                "You are an orchestrator.  \n"
                "- If the user’s question is about arithmetic or sequences, call the `math_agent` tool.  \n"
                "- If they ask for up-to-date facts or web research, call the `web_search_agent` tool.  \n"
                "Return only the tool’s output."
            )
        )

        # now compile the orchestrator
        return create_react_agent(
            llm,
            tools=[math_agent_tool, web_agent_tool],
            checkpointer=self.memory,
            prompt=orchestrator_prompt,
            name=self.name,
        )

    def _call_subagent(self, agent, query: str) -> str:
        """Invoke a React-style agent via its async interface in a new thread."""
        logger.debug(
            "OrchestratorAgent._call_subagent: spawning thread for %s (async)",
            getattr(agent, "name", "<unknown>")
        )
        result_queue = Queue()
        
        def run_agent():
            # Create a fresh event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    agent.ainvoke({"messages": [HumanMessage(content=query)]})
                )
                result_queue.put(result)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        thread.join()
        
        result = result_queue.get()
        logger.debug("OrchestratorAgent._call_subagent: raw messages → %s", result["messages"])
        return result["messages"][-1].content

    def run(self, message: HumanMessage):
        """Base Agent run implementation."""
        logger.info("OrchestratorAgent: received user message → %s", message.content)
        resp = self.state_graph.invoke({"messages": [message]})
        return resp["messages"][-1]