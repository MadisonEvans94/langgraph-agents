import logging
from functools import partial
from typing import Dict
from langgraph.graph import StateGraph, MessagesState, END
from agent_resources.base_agent import Agent
from .nodes import routing_node, default_llm_node, alternate_llm_node

logger = logging.getLogger(__name__)

class ConversationalAgentWithRouting(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id: str | None = None,
        **kwargs
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.llm_dict = self._build_llm_dict(llm_configs)
        self.state_graph = self.build_graph()

    def build_graph(self):
        graph = StateGraph(MessagesState)

        graph.add_node("routing_pass", lambda state: state)
        graph.set_entry_point("routing_pass")

        graph.add_conditional_edges(
            "routing_pass",
            routing_node,
            path_map={
                "default_llm_node": "default_llm_node",
                "alternate_llm_node": "alternate_llm_node",
            },
        )

        graph.add_node(
            "default_llm_node",
            partial(default_llm_node, default_llm=self.llm_dict["default_llm"]),
        )
        graph.add_node(
            "alternate_llm_node",
            partial(alternate_llm_node, alternate_llm=self.llm_dict["alternate_llm"]),
        )

        graph.add_edge("default_llm_node", END)
        graph.add_edge("alternate_llm_node", END)

        return graph.compile(checkpointer=self.memory)
