import logging
from functools import partial
from typing import Dict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
from agent_resources.utils import ChatVLLMWrapper
from agent_resources.base_agent import Agent
from .nodes import llm_node
from langgraph.graph import MessagesState

logger = logging.getLogger(__name__)

class ConversationalAgent(Agent):
    def __init__(self, llm_configs: Dict = {}, memory=None, thread_id=None):
        self.memory = memory if memory else MemorySaver()
        self.thread_id = thread_id if thread_id else 'default'
        self.build_llm_dict(llm_configs)
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
                model=config["model_id"],
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                repetition_penalty=config["repetition_penalty"]
            )
            for name, config in llm_configs.items()
        }
        logger.info(f"self.llm_dict: {self.llm_dict}")
        return self.llm_dict


    def build_graph(self):
        state_graph = StateGraph(MessagesState)

        # Add a node for each llm in the dictionary
        for name, llm in self.llm_dict.items():
            state_graph.add_node(name, partial(llm_node, llm=llm))

        # add edges
        state_graph.add_edge(START, "default_llm")
        state_graph.add_edge("default_llm", END)

        # compile graph
        self.state_graph = state_graph.compile(checkpointer=self.memory)

    def run(self, message: BaseMessage):
     
        try:
            config = {
                "configurable": {
                    "thread_id": self.thread_id,
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