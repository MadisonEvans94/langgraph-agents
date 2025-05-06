# agent_resources/agents/orchestrator_agent.py

import logging
import asyncio
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
        thread_id: str | None = None,
        tools=None,
        use_llm_provider: bool = False,
        name: str = "orchestrator",
        **kwargs,
    ):
        self.name = name
        self.use_llm_provider = use_llm_provider
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def build_graph(self):
        llm = self.llm_dict["default_llm"]

        # Partition MCP tools
        math_tools = [t for t in self.tools if t.name in {"add", "multiply", "fibonacci"}]
        web_tools  = [t for t in self.tools if t.name == "web_search"]

        logger.info(
            "OrchestratorAgent.build_graph: %d math tools (%s), %d web tools (%s)",
            len(math_tools), [t.name for t in math_tools],
            len(web_tools),  [t.name for t in web_tools],
        )

        # Sub‑agent prompts
        math_prompt = SystemMessage(
            content=(
                "You are a math assistant. Use the `add`, `multiply`, and `fibonacci` "
                "tools to answer math queries."
            )
        )
        web_prompt = SystemMessage(
            content="You are a web‑search assistant. Use the `web_search` tool to fetch up‑to‑date information."
        )

        # Build sub‑agents
        math_agent = create_react_agent(
            llm, tools=math_tools, checkpointer=None, prompt=math_prompt, name="math_agent"
        )
        web_agent = create_react_agent(
            llm, tools=web_tools, checkpointer=None, prompt=web_prompt, name="web_search_agent"
        )

        # Helper: wrap sub‑agent as a LangChain Tool
        def make_tool(name: str, agent, description: str) -> Tool:
            async def tool_fn(query: str) -> str:
                logger.info("Orchestrator → %s | query=%s", name, query)
                result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
                answer = result["messages"][-1].content
                logger.info("%s → Orchestrator | result=%s", name, answer)
                return answer

            def sync_stub(query: str) -> str:          # satisfies mandatory `func`
                return asyncio.run(tool_fn(query))

            return Tool(
                name=name,
                description=description,
                func=sync_stub,      # required positional argument
                coroutine=tool_fn,    # real async implementation
                # NOTE: return_direct omitted so the React loop wraps output in AIMessage
            )

        math_agent_tool = make_tool(
            "math_agent",
            math_agent,
            "Delegate to the math sub‑agent for arithmetic or sequence computations.",
        )
        web_agent_tool = make_tool(
            "web_search_agent",
            web_agent,
            "Delegate to the web‑search sub‑agent for queries that need external data.",
        )

        # Orchestrator prompt
        orchestrator_prompt = SystemMessage(
            content=(
                "You are an **orchestrator**.\n\n"
                "• For arithmetic or sequence questions, call `math_agent`.\n"
                "• For up‑to‑date facts or research, call `web_search_agent`.\n\n"
                "Return only the tool output."
            )
        )

        # Compile the orchestrator graph
        return create_react_agent(
            llm,
            tools=[math_agent_tool, web_agent_tool],
            checkpointer=self.memory,
            prompt=orchestrator_prompt,
            name=self.name,
        )

    # ------------------------------------------------------------------ #
    # Synchronous convenience entry‑point
    # ------------------------------------------------------------------ #
    def run(self, message: HumanMessage):
        logger.info("OrchestratorAgent: received user message → %s", message.content)
        resp = self.state_graph.invoke({"messages": [message]})
        return resp["messages"][-1]
