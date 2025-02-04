import logging
from functools import partial
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.base_agent import Agent
from .nodes import route_query, llm_node
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

load_dotenv(override=True)

logger = logging.getLogger(__name__)

class ConversationalAgentWithRouting(Agent):
    def __init__(self, llm_list = [], memory=None, thread_id=None):
        # TODO: Hardcoding LLM instances for now. Will refactor and make more dynamic in future 
        self.llm_list= llm_list
        self.alternate_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )
        self.memory = memory if memory else MemorySaver()
        self.thread_id = thread_id if thread_id else 'default'
        self.build_graph()

    def build_graph(self):
        state_graph = StateGraph(MessagesState)

        # Add nodes:
        state_graph.add_node("strong_llm_node", partial(llm_node, llm=self.llm_list[0])) #TODO: FIX THIS
        state_graph.add_node("regular_llm_node", partial(llm_node, llm=self.alternate_llm))

        # Define edges:
        state_graph.add_conditional_edges(START, route_query)
        state_graph.add_edge("strong_llm_node", END)
        state_graph.add_edge("regular_llm_node", END)

        self.state_graph = state_graph.compile(checkpointer=self.memory)

    def run(self, message: BaseMessage):
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
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
