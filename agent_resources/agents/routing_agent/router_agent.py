import logging
from functools import partial
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict
from agent_resources.utils import ChatVLLMWrapper
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.base_agent import Agent
from .nodes import route_query, llm_node, decision, router_node
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

load_dotenv(override=True)

logger = logging.getLogger(__name__)

class RouterAgent(Agent):

    def __init__(
        self, 
        llm_configs: Dict = None,
        memory=None,
        thread_id=None):
        
        if llm_configs is None:
            llm_configs = {}

        self.memory = memory if memory else MemorySaver()
        self.thread_id = thread_id if thread_id else "default"

        self.build_llm_dict(llm_configs)

        self.alternate_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )
        self.build_graph()

    def build_llm_dict(self, llm_configs: Dict) -> Dict[str, object]:
        """
        Build and store a dictionary mapping names to LLM objects.
        (The returned type is object or a more specific BaseChatModel if available.)
        """
        self.llm_dict = {
            name: ChatVLLMWrapper(
                client=OpenAI(
                    api_key=config.get("api_key", ""),
                    base_url=config.get("base_url", "")
                ),
                model=config.get("model_id", "gpt-3.5-turbo"),
                max_new_tokens=config.get("max_new_tokens", 512),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 1.0),
                repetition_penalty=config.get("repetition_penalty", 1.0)
            )
            for name, config in llm_configs.items()
        }
        return self.llm_dict

    def build_graph(self):
        state_graph = StateGraph(MessagesState)

        # Add nodes to the graph
        state_graph.add_node("router_node", partial(router_node, weak_llm=self.llm_dict["weak_llm"]))
        state_graph.add_edge(START, "router_node")
        #Add a node for each LLM in the dictionary
        for name, llm in self.llm_dict.items():
            state_graph.add_node(name, partial(llm_node, llm=llm))

        
        state_graph.add_conditional_edges(
            "router_node",
            lambda state: state["route"],
        ) 
        
        # Add each LLM node and connect it to the END.
        for name, llm in self.llm_dict.keys():
            state_graph.add_edge(name, END)
        
        self.state_graph = state_graph.compile(checkpointer=self.memory)    

    def run(self, message: BaseMessage):
        """
        Pass the incoming message through the state graph.
        """
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






    