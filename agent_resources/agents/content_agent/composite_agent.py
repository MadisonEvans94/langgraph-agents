""" CompositeAgent = PlanningAgent + OrchestratorAgent in one wrapper. """

from __future__ import annotations
import json
from typing import List, Dict

from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool

from agent_resources.base_agent import Agent
from agent_resources.agents.content_agent.planning_agent import PlanningAgent
from agent_resources.agents.content_agent.orchestrator_agent import OrchestratorAgent


class CompositeAgent(Agent):
    """
    Composite agent that hides the plan‑then‑execute workflow
    behind a single Agent interface (ainvoke / run).
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[BaseTool] = None,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "composite_agent",
        **kwargs,
    ):
        # base Agent initializer (sets .memory, .thread_id, etc.)
        super().__init__(**kwargs)

        self.name = name
        self.tools = tools or []
        self.thread_id = thread_id or "default"
        self.llm_configs = llm_configs
        self.use_llm_provider = use_llm_provider
        self.memory = memory

        # prepare sub‑agents
        self.planner = PlanningAgent(
            llm_configs=llm_configs,
            tools=self.tools,
            memory=memory,
            thread_id=thread_id,
            use_llm_provider=use_llm_provider,
        )
        self.orchestrator = OrchestratorAgent(
            llm_configs=llm_configs,
            tools=self.tools,
            memory=memory,
            thread_id=thread_id,
            use_llm_provider=use_llm_provider,
        )

    # ------------------------------------------------------------------ #
    # REQUIRED by Agent ­– composite agent never uses its own graph,
    # so we return None to satisfy the abstract method contract.
    # ------------------------------------------------------------------ #
    def build_graph(self):
        return None

    # ------------------------------------------------------------------ #
    async def ainvoke(self, message: HumanMessage):
        # 1) planning step
        plan_msg = await self.planner.ainvoke(message)
        raw = plan_msg.content.strip()

        # 2) detect if JSON task list or direct answer
        tasks = None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "tasks" in obj:
                tasks = obj["tasks"]
            elif isinstance(obj, list):
                tasks = obj
        except json.JSONDecodeError:
            pass

        # 3) execute tasks if present
        if tasks:
            result_msg = await self.orchestrator.process_tasks(tasks)
            return result_msg

        # 4) return planner’s direct answer
        return AIMessage(content=raw)

    def run(self, message: HumanMessage):
        # Default Agent.run() relies on state_graph, so override to use ainvoke
        return self.loop.run_until_complete(self.ainvoke(message))