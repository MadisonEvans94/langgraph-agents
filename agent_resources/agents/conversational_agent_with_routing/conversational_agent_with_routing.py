# agent_resources/conversational_agent.py

import logging
from functools import partial
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Iterator, Tuple, Any
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
        thread_id=None,
        use_openai: bool = False,
    ):
        self.memory = memory if memory else MemorySaver()
        self.thread_id = thread_id if thread_id else "default"
        self.use_openai = use_openai

        logger.info(f"Initializing ConversationalAgentWithRouting with use_openai={self.use_openai!r}")
        self.build_llm_dict(llm_configs)
        self.build_graph()

    def build_llm_dict(self, llm_configs: Dict) -> Dict[str, object]:
        if llm_configs is None:
            raise ValueError("llm_configs cannot be None. Provide a valid configuration dictionary.")

        required_keys = ["default_llm", "alternate_llm"]
        for key in required_keys:
            if key not in llm_configs:
                raise ValueError(f"Missing required LLM configuration: '{key}'.")

        self.llm_dict = {}
        logger.info("🛠️ Building LLM dictionary...")

        for name, config in llm_configs.items():
            if not config:
                raise ValueError(f"Configuration for '{name}' is empty.")

            model_id = config.get("model_id") or config.get("model")
            if model_id is None:
                raise ValueError(f"Configuration for '{name}' must include 'model_id' or 'model'.")

            temperature = config.get("temperature", 0.7)
            openai_api_key = config.get("api_key", "")
            base_url = config.get("base_url")
            max_new_tokens = config.get("max_new_tokens", 512)

            logger.info(f"🔹 Config for {name}: model={model_id}, temp={temperature}, streaming=True")

            if self.use_openai:
                # Standard OpenAI usage
                llm = ChatOpenAI(
                    model=model_id,
                    temperature=temperature,
                    # Use None for max_tokens if you prefer controlling it by the request
                    max_tokens=max_new_tokens,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=openai_api_key,
                    streaming=True,  # streaming from actual OpenAI
                )
            else:
                # For vLLM: just treat it like an OpenAI drop-in by specifying base_url
                if base_url is None:
                    raise ValueError(f"When using vLLM, 'base_url' is required for '{name}'.")

                # We can pass a dummy key if your vLLM doesn’t require authentication
                # but the library still expects an openai_api_key param.
                llm = ChatOpenAI(
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=openai_api_key or "EMPTY",
                    openai_api_base=base_url,  # <-- point to vLLM endpoint
                    streaming=True,            # streaming from vLLM
                )

            self.llm_dict[name] = llm
            logger.info(f"✅ Successfully created LLM instance for {name}")

        return self.llm_dict


    def _build_system_prompt(self) -> str:
        return REACT_AGENT_SYSTEM_PROMPT.format(tools_section="")

    def build_graph(self):
        logger.info("Building StateGraph with routing nodes.")
        state_graph = StateGraph(MessagesState)

        state_graph.add_node("routing_pass", lambda state: state)
        state_graph.set_entry_point("routing_pass")

        state_graph.add_conditional_edges(
            "routing_pass",
            routing_node,
            path_map={
                "alternate_llm_node": "alternate_llm_node",
                "default_llm_node": "default_llm_node",
            }
        )

        state_graph.add_node("default_llm_node", partial(default_llm_node, default_llm=self.llm_dict["default_llm"]))
        state_graph.add_node("alternate_llm_node", partial(alternate_llm_node, alternate_llm=self.llm_dict["alternate_llm"]))

        state_graph.add_edge("default_llm_node", END)
        state_graph.add_edge("alternate_llm_node", END)

        self.state_graph = state_graph.compile(checkpointer=self.memory)

    def run(self, message: HumanMessage):
        logger.info(f"Running synchronous run(...) with message={message.content!r}")
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            response = self.state_graph.invoke({"messages": [message]}, config=config)
            ai_message = response["messages"][-1]

            if isinstance(ai_message, AIMessage):
                ai_message.additional_kwargs["model_used"] = ai_message.additional_kwargs.get("model_used", "unknown")
                logger.info(f"Synchronous response: {ai_message.content!r}")
                return ai_message
            else:
                logger.error("Unexpected message type (not AIMessage).")
                raise ValueError("Expected AIMessage in response.")
        except Exception as e:
            logger.exception("Error generating synchronous response:")
            return AIMessage(content=f"Sorry, I encountered an error: {e}")

    def run_stream(self, message: HumanMessage) -> Iterator[Tuple[str, Any]]:
        """
        Streaming run method. Instead of returning final output directly,
        yields events from the graph execution. 
        Each item is (stream_mode, data).
        """
        logger.info(f"Running streaming run(...) with message={message.content!r}")
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            # We request "messages" and "updates"
            stream_iterator = self.state_graph.stream(
                {"messages": [message]},
                config=config,
                stream_mode=["messages", "updates"]
            )
            logger.info("Got stream_iterator from state_graph.stream(...)")

            for stream_mode, data in stream_iterator:
                logger.debug(f"run_stream => yield: ({stream_mode!r}, {data!r})")
                yield (stream_mode, data)

        except Exception as e:
            logger.exception("Error generating streaming response:")
            yield ("error", f"Error: {e}")

    def _extract_response_metadata(self, response: dict, input_message: HumanMessage) -> AIMessage:
        ...
        # (left unchanged)


    def _extract_response_metadata(self, response: dict, input_message: HumanMessage) -> AIMessage:
        """
        Extracts the final AI message and relevant metadata (e.g., tools used, model used)
        from the state graph response.
        """
        all_msgs = response.get("messages", [])
        idx = next((i for i, m in enumerate(all_msgs) if m == input_message), None)
        if idx is None:
            return AIMessage(content="No matching user message found.")

        new_msgs = all_msgs[idx + 1:]
        used_tools = {m.name for m in new_msgs if getattr(m, "role", "") == "tool"}
        final_ai_msg = new_msgs[-1] if new_msgs else AIMessage(content="No AI response.")
        node_model_used = getattr(final_ai_msg, "additional_kwargs", {}).get("model_used", "unknown")

        final_ai_msg.additional_kwargs["tools_used"] = list(used_tools)
        final_ai_msg.additional_kwargs["model_used"] = node_model_used

        return final_ai_msg