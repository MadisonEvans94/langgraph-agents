# agent_resources/agents/conversational_agent/conversational_agent.py

import logging
from functools import partial
from typing import Dict
from langgraph.graph import StateGraph, MessagesState, START, END
from agent_resources.base_agent import Agent
from .nodes import llm_node

logger = logging.getLogger(__name__)

class ConversationalAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id=None,
        tools=None,
        **kwargs
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.tools = tools or []
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()
        self._build_llm_dict(llm_configs)

    def build_graph(self):
        graph = StateGraph(MessagesState)

        for name, llm in self.llm_dict.items():
            graph.add_node(name, partial(llm_node, llm=llm))

        graph.add_edge(START, "default_llm")
        graph.add_edge("default_llm", END)

        return graph.compile(checkpointer=self.memory)
