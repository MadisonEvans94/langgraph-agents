from __future__ import annotations
# import logging
import json
import re
from typing import Dict, Any, List
from pprint import pformat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from agent_resources.base_agent import Agent
from agent_resources.state_types import SupervisorState
from agent_resources.prompts import PLANNING_AGENT_RAW_SYSTEM_PROMPT
from loguru import logger


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
        self.state_graph = self.build_graph()

    def build_graph(self):
        llm = self.llm_dict["default_llm"]
        prompt = SystemMessage(content=PLANNING_AGENT_RAW_SYSTEM_PROMPT)
        return create_react_agent(
            name=self.name,
            model=llm,
            prompt=prompt,
            tools=[],
            state_schema=None
        )

    async def ainvoke(self, message: HumanMessage) -> dict:
        logger.info("\nPlanningAgent received user query:\n%s\n", message.content)

        initial_state: dict[str, Any] = {"messages": [message], "tasks": []}
        result = await self.state_graph.ainvoke(
            initial_state,
            config=self._default_config(),
        )

        last = result["messages"][-1]
        content = last.content if isinstance(last, AIMessage) else str(last)
        logger.debug("PlanningAgent raw content → %s", content)

        m = re.search(r"```json\s*([\s\S]*?)```", content, re.IGNORECASE)
        json_text = m.group(1).strip() if m else content.strip()

        try:
            raw_tasks: List[dict] = json.loads(json_text)
        except Exception as e:
            logger.error("PlanningAgent failed to parse tasks JSON: %s", e)
            raw_tasks = []

        structured: List[dict] = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            structured.append({
                "id":          t.get("id"),
                "description": t.get("description"),
                "assigned_to": t.get("assigned_to"),
                "status":      "pending",
                "result":      None,
                "depends_on":  t.get("depends_on", []),
            })

        logger.success(f"\nPlanningAgent generated structured tasks:\n\n{pformat(structured, indent=2)}\n")
        result["tasks"] = structured
        return result