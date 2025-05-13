# agent_resources/agents/task_workflow/composite_agent.py

from __future__ import annotations
import json
import asyncio
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool

from agent_resources.base_agent import Agent
from agent_resources.agents.task_workflow.planning_agent import PlanningAgent
from agent_resources.agents.task_workflow.orchestrator_agent import OrchestratorAgent


class CompositeAgent(Agent):
    """
    Composite “query agent” that transparently does:
      1. Plan with PlanningAgent
      2. If tasks → execute with OrchestratorAgent
      3. Else → return direct answer
    """

    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        tools: List[BaseTool] | None = None,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "composite_agent",
        **kwargs,
    ):
        # initialize base Agent (sets .memory, .thread_id, etc.)
        super().__init__(**kwargs)

        self.name = name
        self.thread_id = thread_id or "default"
        self.llm_configs = llm_configs
        self.use_llm_provider = use_llm_provider
        self.memory = memory

        # 1️⃣  Instantiate the orchestrator with the raw MCP tools
        self.orchestrator = OrchestratorAgent(
            llm_configs=llm_configs,
            tools=tools or [],
            memory=memory,
            thread_id=thread_id,
            use_llm_provider=use_llm_provider,
        )

        # 2️⃣  Pass the *wrapped* orchestrator.tools into the planner
        self.planner = PlanningAgent(
            llm_configs=llm_configs,
            tools=self.orchestrator.tools,
            memory=memory,
            thread_id=thread_id,
            use_llm_provider=use_llm_provider,
        )

        # We don't need our own state_graph since we override ainvoke/run
        self.state_graph = None

    def build_graph(self):
        return None
    
    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        # --- Step 1: Planning ---
        plan_msg = await self.planner.ainvoke(message)
        raw = plan_msg.content.strip()

        # --- Step 2: Detect tasks vs direct answer ---
        tasks: Optional[List[dict]] = None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "tasks" in obj:
                tasks = obj["tasks"]
            elif isinstance(obj, list):
                tasks = obj
        except json.JSONDecodeError:
            pass

        # --- Step 3: If tasks, orchestrate them; otherwise return direct ---
        if tasks:
            return await self.orchestrator.process_tasks(tasks)

        return AIMessage(content=raw)

    def run(self, message: HumanMessage) -> AIMessage:
        # Synchronous helper for convenience
        return asyncio.run(self.ainvoke(message))