# agent_resources/agents/supervisor/planning_agent.py

from __future__ import annotations
import logging
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from agent_resources.base_agent import Agent

logger = logging.getLogger(__name__)

class PlanningAgent(Agent):
    """
    Planning agent that takes the user's original query and breaks it into discrete tasks.
    """
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        *,
        memory=None,
        thread_id: str | None = None,
        use_llm_provider: bool = False,
        name: str = "planning_agent",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_llm_provider = use_llm_provider
        self._build_llm_dict(llm_configs)
        self.name = name
        self.memory = memory
        self.thread_id = thread_id or "default"
        # create_react_agent returns a compiled graph
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        prompt = SystemMessage(content=(
            "You are a planning agent.  Given the user's query in state['messages'], "
            "break it down into a JSON array of tasks, each with 'id' (integer) and 'description' (string)."
        ))

        # JSON Schema for an array of {id:int,description:str}
        tasks_json_schema = {
            "title": "TaskList",
            "description": "An array of tasks, each with id and description",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["id", "description"],
            },
        }

        # NOTE: we no longer pass state_schema here,
        # so the agent uses its default which *does** include* structured_response
        return create_react_agent(
            name=self.name,
            model=llm,
            tools=[],
            prompt=prompt,
            response_format=tasks_json_schema,
        )

    async def ainvoke(self, message: HumanMessage) -> dict:
        logger.info("PlanningAgent received → %s", message.content)
        initial_state: dict[str, Any] = {"messages": [message], "tasks": []}
        result = await self.state_graph.ainvoke(
            initial_state,
            config=self._default_config(),
        )

        # The parsed array now lives in `structured_response`
        raw_tasks: List[dict] = result.get("structured_response", [])
        logger.info("PlanningAgent raw structured_response → %s", raw_tasks)

        # Convert into your orchestration format
        structured_tasks = []
        for t in raw_tasks:
            structured_tasks.append({
                "id":          t["id"],
                "description": t["description"],
                "status":      "pending",
                "result":      None,
            })

        logger.info("PlanningAgent generated tasks → %s", structured_tasks)
        result["tasks"] = structured_tasks
        return result