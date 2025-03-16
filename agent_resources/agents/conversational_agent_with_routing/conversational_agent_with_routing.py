import logging
from functools import partial
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from agent_resources.utils import ChatVLLMWrapper 
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.base_agent import Agent

from .nodes import (
    default_llm_node,
    routing_node,  
    alternate_llm_node
)
from langgraph.graph import MessagesState

load_dotenv(override=True)

logger = logging.getLogger(__name__)

class ConversationalAgentWithRouting(Agent):
    def __init__(
        self, 
        llm_configs: Dict = None, 
        memory=None, 
        thread_id=None
    ):
        """
        :param llm_configs: Dictionary describing how to build each LLM.
                            Example:
                            {
                              "default_llm": {
                                "api_key": "...",
                                "base_url": "...",
                                "model_id": "...",
                                "max_new_tokens": 512,
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "repetition_penalty": 1.0
                              },
                              "alternate_llm": {
                                "api_key": "...",
                                "base_url": "...",
                                "model_id": "...",
                                "max_new_tokens": 512,
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "repetition_penalty": 1.0
                              }
                            }
        :param memory: Checkpoint memory (optional).
        :param thread_id: Unique identifier for sessions (optional).
        """
        self.memory = memory if memory else MemorySaver()
        self.thread_id = thread_id if thread_id else "default"

        # Build the dictionary of LLMs (including default_llm and alternate_llm)
        self.build_llm_dict(llm_configs)

        # Build the state graph
        self.build_graph()

    def build_llm_dict(self, llm_configs: Dict) -> Dict[str, object]:
        """
        Uses ChatVLLMWrapper to call your custom vLLM endpoint.
        The llm_configs should include keys like "model_id", "base_url", etc.
        Expected configuration keys (e.g., in your config.yaml):
        - default_llm
        - alternate_llm
        """
        if llm_configs is None:
            raise ValueError("llm_configs cannot be None. Please provide a valid configuration dictionary.")

        # Ensure required llm keys are present.
        required_keys = ["default_llm", "alternate_llm"]
        for key in required_keys:
            if key not in llm_configs:
                raise ValueError(f"Missing required LLM configuration: '{key}'. Please include it in your config.yaml.")

        self.llm_dict = {}

        # Build an LLM for each config key provided.
        for name, config in llm_configs.items():
            if not config:
                raise ValueError(f"Configuration for '{name}' is empty. Please provide a valid configuration.")
            
            model_id = config.get("model_id")
            if model_id is None:
                raise ValueError(f"Configuration for '{name}' must include 'model_id'.")

            base_url = config.get("base_url")
            if base_url is None:
                raise ValueError(f"Configuration for '{name}' must include 'base_url'.")

            temperature = config.get("temperature", 0.7)
            openai_api_key = config.get("api_key", "")

            # Create a client that points to your vLLM endpoint.
            client = OpenAI(api_key=openai_api_key, base_url=base_url)
            
            llm = ChatVLLMWrapper(
                client=client,
                model=model_id,
                max_new_tokens=config.get("max_new_tokens", 512),
                temperature=temperature,
                top_p=config.get("top_p", 1.0),
                repetition_penalty=config.get("repetition_penalty", 1.0),
            )
            # llm = ChatOpenAI(
            #     model="gpt-3.5-turbo",
            #     temperature=0,
            #     max_tokens=None,
            #     timeout=None,
            #     max_retries=2,
            #     # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            #     # base_url="...",
            #     # organization="...",
            #     # other params...
            # )
            self.llm_dict[name] = llm

        return self.llm_dict

    def _build_system_prompt(self) -> str:
        """
        Build a ReAct system prompt that lists available tools.
        """
        return REACT_AGENT_SYSTEM_PROMPT.format(tools_section="")

    def build_graph(self):
        """
        Creates the LangGraph state graph with routing logic.
        Instead of using routing_node as the entry node, we use a simple pass-through node
        and then add a conditional edge that calls routing_node.
        """
        state_graph = StateGraph(MessagesState)

        # Create a pass-through node that just returns the state unchanged.
        state_graph.add_node("routing_pass", lambda state: state)
        state_graph.set_entry_point("routing_pass")

        # Add conditional edges using routing_node to determine the next node.
        state_graph.add_conditional_edges(
            "routing_pass",
            routing_node,  
            path_map={
                "alternate_llm_node": "alternate_llm_node",
                "default_llm_node": "default_llm_node",
            }
        )

        # Add processing nodes.
        state_graph.add_node("default_llm_node", partial(default_llm_node, default_llm=self.llm_dict["default_llm"]))
        state_graph.add_node("alternate_llm_node", partial(alternate_llm_node, alternate_llm=self.llm_dict["alternate_llm"]))

        # Set exit edges.
        state_graph.add_edge("default_llm_node", END)
        state_graph.add_edge("alternate_llm_node", END)

        # Compile the graph with memory checkpointing.
        self.state_graph = state_graph.compile(checkpointer=self.memory)

    def run(self, message: HumanMessage):
        """
        Processes a single user query (HumanMessage) by invoking the state graph,
        then extracts metadata (tools used, model used) for the frontend.
        """
        try:
            config = {
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_ns": "default",
                    "checkpoint_id": "default"
                }
            }

            # Invoke the state graph.
            response = self.state_graph.invoke({"messages": [message]}, config=config)
            return self._extract_response_metadata(response, message)

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content=f"Sorry, I encountered an error: {e}")


    def _extract_response_metadata(self, response: dict, input_message: HumanMessage) -> AIMessage:
        """
        Extracts the final AI message and relevant metadata (e.g., tools used, model used)
        from the state graph response.

        :param response: The dictionary returned from state_graph.invoke.
        :param input_message: The original HumanMessage input.
        :return: An AIMessage with the metadata attached.
        """
        all_msgs = response.get("messages", [])

        # Locate the index of the incoming HumanMessage.
        idx = next((i for i, m in enumerate(all_msgs) if m == input_message), None)
        if idx is None:
            return AIMessage(content="No matching user message found.")

        # Get all newly generated messages after the user's input.
        new_msgs = all_msgs[idx + 1:]
        used_tools = {m.name for m in new_msgs if getattr(m, "role", "") == "tool"}

        # The final AI message is typically the last one.
        final_ai_msg = new_msgs[-1] if new_msgs else AIMessage(content="No AI response.")

        # Extract the model used from the final AI message (set in nodes.py).
        node_model_used = getattr(final_ai_msg, "additional_kwargs", {}).get("model_used", "unknown")

        # Attach the metadata for frontend display.
        final_ai_msg.additional_kwargs["tools_used"] = list(used_tools)
        final_ai_msg.additional_kwargs["model_used"] = node_model_used

        return final_ai_msg
