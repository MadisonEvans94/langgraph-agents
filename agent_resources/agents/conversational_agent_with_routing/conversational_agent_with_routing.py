import logging
from functools import partial
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict
from langchain_openai import ChatOpenAI
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from agent_resources.utils import ChatVLLMWrapper  # <-- your custom wrapper
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.base_agent import Agent
from agent_resources.tools.tool_registry import ToolRegistry

from .nodes import (
    default_llm_node,
    routing_node,  # used only as conditional function
    check_tool_calls,
    react_logic_node,
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
                                "strong_llm": { ... },
                                "weak_llm": { ... }
                            }
        :param memory: Checkpoint memory (optional).
        :param thread_id: Unique identifier for sessions (optional).
        """
        self.memory = memory if memory else MemorySaver()
        self.thread_id = thread_id if thread_id else "default"

        # Retrieve the tools you want to use (e.g. 'tavily_search')
        self.tools = ToolRegistry.get_tools(["tavily_search"])

        # Build the dictionary of LLMs (including default_llm and alternate_llm)
        self.build_llm_dict(llm_configs)

        # Build the state graph
        self.build_graph()

    def build_llm_dict(self, llm_configs: Dict) -> Dict[str, object]:
        """
        Optionally use ChatVLLMWrapper to call your custom vLLM endpoint.
        """
        if llm_configs is None:
            llm_configs = {}

        self.llm_dict = {}

        # Build an LLM for each config key
        for name, config in llm_configs.items():
            config = config or {}
            model_id = config.get("model_id", "gpt-3.5-turbo")
            temperature = config.get("temperature", 0.7)
            openai_api_key = config.get("api_key", "")
            llm = ChatOpenAI(
                model=model_id,
                temperature=temperature,
                openai_api_key=openai_api_key,
            )
            self.llm_dict[name] = llm

        # Ensure default_llm
        if "default_llm" not in self.llm_dict:
            default_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=""
            )
            self.llm_dict["default_llm"] = default_llm

        # Ensure alternate_llm
        if "alternate_llm" not in self.llm_dict:
            alt_llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                openai_api_key=""
            )
            self.llm_dict["alternate_llm"] = alt_llm

        # Bind tools to default_llm if available.
        if self.tools:
            self.llm_dict["default_llm"] = self.llm_dict["default_llm"].bind_tools(self.tools)

        return self.llm_dict

    def _build_system_prompt(self) -> str:
        """
        Build a ReAct system prompt that lists available tools.
        """
        lines = []
        for i, tool in enumerate(self.tools, start=1):
            lines.append(f"{i}. {tool.name}: {tool.description}")
        tools_section = "\n".join(lines)
        return REACT_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)

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
            routing_node,  # this function returns a string mapping to the next node
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
        Processes a single user query (HumanMessage), returning the final AIMessage.
        """
        try:
            config = {
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_ns": "default",
                    "checkpoint_id": "default"
                }
            }

            # Invoke the state graph with the new user message.
            response = self.state_graph.invoke({"messages": [message]}, config=config)
            all_msgs = response["messages"]

            # Find the index of the incoming HumanMessage.
            idx = None
            for i, m in enumerate(all_msgs):
                if m == message:
                    idx = i
                    break

            if idx is None:
                return AIMessage(content="No matching user message found.")

            # Everything after idx is generated for this turn.
            new_msgs = all_msgs[idx+1:]
            used_tools = {m.name for m in new_msgs if getattr(m, "role", "") == "tool"}
            model_used = "gpt-4" if used_tools else "gpt-3.5-turbo"
            final_ai_msg = new_msgs[-1] if new_msgs else AIMessage(content="No AI response.")

            # Attach metadata.
            final_ai_msg.additional_kwargs["tools_used"] = list(used_tools)
            final_ai_msg.additional_kwargs["model_used"] = model_used

            return final_ai_msg

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content=f"Sorry, I encountered an error: {e}")
