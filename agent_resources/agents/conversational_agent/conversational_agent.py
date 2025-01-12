import logging
from functools import partial

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_resources.base_agent import Agent
from .nodes import State, llm_node

logger = logging.getLogger(__name__)

class ConversationalAgent(Agent):
    def __init__(self, llm=None, memory=None):
        self.llm = llm
        self.memory = memory if memory else MemorySaver()
        self.build_graph()

    def build_graph(self):
        state_graph = StateGraph(State)

        # add nodes 
        state_graph.add_node("start_node", lambda state: state)
        state_graph.add_node("llm_node", partial(llm_node, llm=self.llm))
        state_graph.add_node("end_node", lambda state: state)
        state_graph.set_entry_point("start_node")

        # add edges
        state_graph.add_edge("start_node", "llm_node")
        state_graph.add_edge("llm_node", "end_node")
        state_graph.add_edge("end_node", END)

        # compile graph
        self.state_graph = state_graph.compile(checkpointer=self.memory)

    def run(self, message: BaseMessage):
        try:
            config = {
                "configurable": {
                    "thread_id": "default",   # or any unique string
                }
            }

            response = self.state_graph.invoke({"messages": [message]}, config=config)
            ai_message = response["messages"][-1]

            if isinstance(ai_message, AIMessage):
                return ai_message
            else:
                logger.error("Unexpected message type in response.")
                raise ValueError("Expected AIMessage in the response.")

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content=f"Sorry, I encountered an error: {e}")