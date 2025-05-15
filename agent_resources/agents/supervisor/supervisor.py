from __future__ import annotations
import logging
import asyncio
import pprint
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_handoff_tool

from agent_resources.base_agent import Agent
from agent_resources.state_types import SupervisorState
from agent_resources.agents.supervisor.planning_agent import PlanningAgent
from agent_resources.agents.supervisor.math_agent import MathAgent
from agent_resources.agents.supervisor.web_search_agent import WebSearchAgent
from agent_resources.prompts import SUPERVISOR_AGENT_PROMPT

logger = logging.getLogger(__name__)

class SupervisorAgent(Agent):
    """
    SupervisorAgent that:
      1) Plans first (via PlanningAgent)
      2) Uses the built in create_supervisor loop
      3) Aggregates all task results and returns the final answer
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
        self.llm_configs      = llm_configs
        self.use_llm_provider = use_llm_provider
        self._build_llm_dict(llm_configs)
        self.tools            = tools or []
        self.memory           = memory
        self.thread_id        = thread_id or "default"
        self.name             = name

        # Build & compile the orchestrator graph once
        self.state_graph = self.build_graph().compile()

    def build_graph(self):
        # 1) Instantiate your sub-agents
        math_agent = MathAgent(
            llm_configs=self.llm_configs,
            tools=[t for t in self.tools if t.name in ("add", "multiply", "fibonacci")],
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="math_agent",
        )
        web_agent = WebSearchAgent(
            llm_configs=self.llm_configs,
            tools=[t for t in self.tools if t.name == "web_search"],
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="web_search_agent",
        )

        # 2) Create the hand-off tools for those agents
        handoff_tools = [
            create_handoff_tool(agent_name=math_agent.name),
            create_handoff_tool(agent_name=web_agent.name),
        ]

        # 3) (Optional) If your LLM requires explicit binding for function calling:
        llm = self.llm_dict["default_llm"]
        if hasattr(llm, "bind_tools"):
            llm = llm.bind_tools(handoff_tools)

        # 4) Prompt that drives the loop & dispatch
        supervisor_prompt = SystemMessage(content=SUPERVISOR_AGENT_PROMPT)

        # 5) Build & return the supervisor StateGraph
        return create_supervisor(
            agents         = [math_agent.state_graph, web_agent.state_graph],
            model          = llm,
            tools          = handoff_tools,          # <–– THIS is the fix
            prompt         = supervisor_prompt,
            state_schema   = SupervisorState,      # so it carries `tasks`
            supervisor_name= self.name,
        )

    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        logger.info("SupervisorAgent received user query:\n%s", message.content)

        # A) Plan first, outside the main graph
        planner = PlanningAgent(
            llm_configs     = self.llm_configs,
            memory          = self.memory,
            thread_id       = self.thread_id,
            use_llm_provider= self.use_llm_provider,
            name            = "planning_agent",
        )
        plan_result = await planner.ainvoke(message)
        tasks = plan_result.get("tasks", [])
        logger.info(
            "\n\nStructured tasks from PlanningAgent:\n\n%s\n\n",
            pprint.pformat(tasks, indent=2, underscore_numbers=True)
        )

        # B) Seed the supervisor graph with that task list
        initial_state = {"messages": [message], "tasks": tasks}
        resp = await self.state_graph.ainvoke(
            initial_state,
            config=self._default_config(),
        )

        # C) Return the final assistant message
        last = resp["messages"][-1]
        return last if isinstance(last, AIMessage) else AIMessage(content=str(last))

    def run(self, message: HumanMessage) -> AIMessage:
        return asyncio.run(self.ainvoke(message))