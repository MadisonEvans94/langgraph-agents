# agent_resources/agents/supervisor/supervisor.py

from __future__ import annotations
import logging
import asyncio
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph_supervisor.handoff import create_handoff_tool

from agent_resources.base_agent import Agent
from agent_resources.state_types import OrchestratorState
from agent_resources.agents.supervisor.planning_agent import PlanningAgent
from agent_resources.agents.supervisor.math_agent import MathAgent
from agent_resources.agents.supervisor.web_search_agent import WebSearchAgent

logger = logging.getLogger(__name__)

class SupervisorAgent(Agent):
    """
    Custom supervisor that:
      1) Runs PlanningAgent to break the user query into tasks
      2) Runs a ReAct‐style supervisor loop to dispatch each task
      3) Aggregates results and returns the final answer
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
        self.llm_configs = llm_configs
        self.use_llm_provider = use_llm_provider
        self._build_llm_dict(llm_configs)
        self.tools = tools or []
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.name = name

        # Build & compile the orchestrator graph once
        self.state_graph = self.build_graph().compile()

    def build_graph(self) -> StateGraph:
        # 1) Instantiate your three agents
        planning_agent = PlanningAgent(
            llm_configs=self.llm_configs,
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="planning_agent",
        )
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

        # 2) Create handoff tools for supervisor to call
        handoff_tools = [
            create_handoff_tool(agent_name=planning_agent.name),
            create_handoff_tool(agent_name=math_agent.name),
            create_handoff_tool(agent_name=web_agent.name),
        ]

        # 3) Bind the tools into your supervisor LLM
        llm = self.llm_dict["default_llm"]
        if hasattr(llm, "bind_tools"):
            llm = llm.bind_tools(handoff_tools)

        # 4) Build the supervisor’s ReAct node
        supervisor_prompt = SystemMessage(content=(
"""You are the supervisor. You have a list of tasks in state['tasks'], each with id, description, status, and result.

On each turn:
- If there is any task whose status is "pending", pick the next pending task and call the matching tool (`transfer_to_math_agent` or `transfer_to_web_search_agent`) with the task description.
- If all tasks have status "done", produce the final assistant answer **without** calling any tools.
"""
        ))
        supervisor_react = create_react_agent(
            name="supervisor",
            model=llm,
            tools=handoff_tools,
            prompt=supervisor_prompt,
            state_schema=OrchestratorState,
        )

        # 5) Wire up the StateGraph
        builder = StateGraph(state_schema=OrchestratorState)

        # Start → PlanningAgent
        builder.add_node(planning_agent.state_graph, destinations=(supervisor_react.name,))
        builder.add_edge(START, planning_agent.state_graph.name)

        # PlanningAgent → Supervisor
        builder.add_node(
            supervisor_react,
            destinations=(math_agent.name, web_agent.name, END),
        )
        builder.add_edge(planning_agent.state_graph.name, supervisor_react.name)

        # MathAgent → Supervisor (return only)
        builder.add_node(math_agent.state_graph, destinations=(supervisor_react.name,))
        builder.add_edge(math_agent.state_graph.name, supervisor_react.name)

        # WebSearchAgent → Supervisor (return only)
        builder.add_node(web_agent.state_graph, destinations=(supervisor_react.name,))
        builder.add_edge(web_agent.state_graph.name, supervisor_react.name)

        return builder

    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        logger.info("SupervisorAgent received → %s", message.content)
        initial_state = {"messages": [message], "tasks": []}
        resp = await self.state_graph.ainvoke(
            initial_state,
            config=self._default_config(),
        )

        last = resp["messages"][-1]
        if isinstance(last, AIMessage):
            return last
        return AIMessage(content=getattr(last, "content", str(last)))

    def run(self, message: HumanMessage) -> AIMessage:
        return asyncio.run(self.ainvoke(message))