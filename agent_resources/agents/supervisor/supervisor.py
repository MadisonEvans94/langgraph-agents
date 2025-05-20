# agent_resources/agents/supervisor/supervisor.py

from __future__ import annotations
import asyncio
import logging
import pprint
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from agent_resources.base_agent import Agent
from agent_resources.state_types import SupervisorState
from agent_resources.agents.supervisor.planning_agent import PlanningAgent
from agent_resources.agents.supervisor.math_agent import MathAgent
from agent_resources.agents.supervisor.web_search_agent import WebSearchAgent

logger = logging.getLogger(__name__)


class SupervisorAgent(Agent):
    """
    Orchestrates a queue-based, dependency-aware workflow between
    web-search and math sub-agents.
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
    ):
        super().__init__(**kwargs)
        self.llm_configs = llm_configs
        self.use_llm_provider = use_llm_provider
        self._build_llm_dict(llm_configs)

        self.tools = tools or []
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.name = name

        # Sub‐agents
        self.math_agent = MathAgent(
            llm_configs=self.llm_configs,
            tools=[t for t in self.tools if t.name in ("add", "multiply", "fibonacci")],
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="math_agent",
        )
        self.web_agent = WebSearchAgent(
            llm_configs=self.llm_configs,
            tools=[t for t in self.tools if t.name == "web_search"],
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="web_search_agent",
        )

        # Compile the orchestration graph
        self.state_graph = self.build_graph().compile()


    def build_graph(self) -> StateGraph:
        builder = StateGraph(SupervisorState)

        # ─────────────────────────────────────────── Scheduler Node
        def scheduler(state: dict):
            tasks = state["tasks"]
            ready = state["ready_queue"]

            # All done → summary
            if all(t["status"] == "done" for t in tasks):
                state["next"] = "summary"
                return state

            # Nothing ready → spin
            if not ready:
                state["next"] = "scheduler"
                return state

            # Dequeue and dispatch
            task_id = ready.pop(0)
            state["current_task_id"] = task_id
            assigned = next(t for t in tasks if t["id"] == task_id)["assigned_to"]
            state["next"] = assigned
            return state

        builder.add_node("scheduler", scheduler)
        builder.add_conditional_edges(
            "scheduler",
            lambda s: s["next"],
            {
                "web_search_agent": "web_search_agent",
                "math_agent": "math_agent",
                "summary": "summary",
                "scheduler": "scheduler",
            },
        )

        # ─────────────────────────────────────────── Runner Factory
        def make_runner(agent_graph, agent_name: str):
            async def _run(state: dict, config=None):
                # pull current task
                tid = state.pop("current_task_id", None)
                if tid is None:
                    raise RuntimeError(f"{agent_name} invoked with no current_task_id")

                # execute sub-agent
                task = next(t for t in state["tasks"] if t["id"] == tid)
                result_state = await agent_graph.ainvoke(
                    {"messages": [HumanMessage(content=task["description"])]},
                    config=config or {},
                )

                # record result
                task["result"] = result_state["messages"][-1].content
                task["status"] = "done"

                # unblock dependents
                str_tid = str(tid)
                for child in state["dependents"].get(str_tid, []):
                    state["in_degree"][child] -= 1
                    if state["in_degree"][child] == 0:
                        state["ready_queue"].append(int(child))

                # force mutation detection
                state["tasks"] = list(state["tasks"])
                state["ready_queue"] = list(state["ready_queue"])
                state["in_degree"] = dict(state["in_degree"])

                # hand back to scheduler
                state["next"] = "scheduler"
                return state

            _run.__name__ = agent_name
            return _run

        # add runner nodes (without destinations), then wire edges explicitly
        builder.add_node("web_search_agent", make_runner(self.web_agent.state_graph, "web_search_agent"))
        builder.add_edge("web_search_agent", "scheduler")

        builder.add_node("math_agent", make_runner(self.math_agent.state_graph, "math_agent"))
        builder.add_edge("math_agent", "scheduler")

        # ─────────────────────────────────────────── Summary Node
        async def summary_node(state: dict, config=None):
            summary = "\n".join(
                f"Task {t['id']}: {t['description']} → {t['result']}"
                for t in state["tasks"]
            )
            state["messages"].append(AIMessage(content=summary))
            return state

        builder.add_node("summary", summary_node)
        builder.add_edge("summary", END)

        # kick it all off
        builder.add_edge(START, "scheduler")
        return builder


    async def ainvoke(self, message: HumanMessage) -> AIMessage:
        logger.info("SupervisorAgent received query:\n%s", message.content)

        # 1) planning
        planner = PlanningAgent(
            llm_configs=self.llm_configs,
            memory=self.memory,
            thread_id=self.thread_id,
            use_llm_provider=self.use_llm_provider,
            name="planning_agent",
        )
        plan_state = await planner.ainvoke(message)
        tasks = plan_state["tasks"]
        logger.info("Structured tasks:\n%s", pprint.pformat(tasks, indent=2))

        # 2) build dependency maps (string keys!)
        in_degree = {str(t["id"]): len(t["depends_on"]) for t in tasks}
        dependents = {str(t["id"]): [] for t in tasks}
        for t in tasks:
            for dep in t["depends_on"]:
                dependents[str(dep)].append(str(t["id"]))
        ready_queue = [t["id"] for t in tasks if in_degree[str(t["id"])] == 0]

        # 3) initial graph state
        init_state = {
            "messages": [message],
            "tasks": tasks,
            "in_degree": in_degree,
            "dependents": dependents,
            "ready_queue": ready_queue,
            "current_task_id": None,
            "next": "scheduler",
        }

        # 4) run
        final_state = await self.state_graph.ainvoke(init_state, config=self._default_config())

        # 5) return the summary message
        return final_state["messages"][-1]


    def run(self, message: HumanMessage) -> AIMessage:
        return asyncio.run(self.ainvoke(message))