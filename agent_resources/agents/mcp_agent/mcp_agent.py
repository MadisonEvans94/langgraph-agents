import logging
from typing import Dict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from agent_resources.base_agent import Agent
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

logger = logging.getLogger(__name__)

@tool(description="add two numbers together")
def add(a: int, b: int) -> int:
    return a + b

class MCPAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict = None,
        memory=None,
        thread_id=None,
        tools=None,
        **kwargs
    ):
        """
        Initializes an MCPAgent by loading LLM configurations, accepting an externally provided
        list of tools, and building a reactive graph.
        """
        self.use_openai = kwargs.get("use_openai", False)
        self.tools = tools if tools is not None else []
        # self.tools=[add]
        self.llm_dict = self.build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id if thread_id else 'default'
        self.state_graph = self.build_graph()

    def build_graph(self):
        """
        Construct the agent's state graph using a dynamic system prompt.
        This utilizes langgraph.prebuilt.create_react_agent.
        """

        try:
            system_prompt = self._build_system_prompt()
            state_graph = create_react_agent(
                self.llm_dict['default_llm'],
                tools=self.tools,
                checkpointer=self.memory,
                prompt=SystemMessage(content=system_prompt),
            )
            return state_graph
        except ValueError as ve:
            logger.error(f"Validation error during agent compilation: {ve}")
            raise
        except Exception as e:
            logger.error("Unexpected error during agent compilation", exc_info=True)
            raise

    def build_llm_dict(self, llm_configs: Dict) -> Dict[str, object]:
        if llm_configs is None:
            raise ValueError("llm_configs cannot be None. Provide a valid configuration dictionary.")
        required_keys = ["default_llm", "alternate_llm"]
        for key in required_keys:
            if key not in llm_configs:
                raise ValueError(f"Missing required LLM configuration: '{key}'.")
        llm_dict = {}
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

            if self.use_openai:
                llm = ChatOpenAI(
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=openai_api_key,
                    streaming=True,
                )
            else:
                if base_url is None:
                    raise ValueError(f"When using vLLM, 'base_url' is required for '{name}'.")
                llm = ChatOpenAI(
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=openai_api_key or "EMPTY",
                    openai_api_base=base_url,
                    streaming=True,
                )

            llm_dict[name] = llm
        return llm_dict

    def _build_system_prompt(self) -> str:
        """
        Build the dynamic system prompt by injecting information about the available tools.
        """
        tools_section = self._build_tools_section()
        system_prompt = REACT_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)
        logger.info(f"\n--------------\n\nsystem prompt: \n\n\n{system_prompt}\n\n\n")
        return system_prompt

    def _build_tools_section(self) -> str:
        """
        Returns a string that enumerates the agent's available tools.
        """
        lines = []
        for i, tool in enumerate(self.tools, start=1):
            line = f"{i}. {tool.name}: {tool.description}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, message: BaseMessage):
        """
        Process a message synchronously and return the final AI response.
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
            return AIMessage(content="Sorry, I encountered an error while processing your request.")

    def run_stream(self, message: BaseMessage):
        """
        Process a message in streaming mode and return an iterator over the stream.
        """
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            return self.state_graph.stream({"messages": [message]}, config=config, stream_mode=["messages", "updates"])
        except Exception as e:
            logger.error("Error during streaming invocation", exc_info=True)
            raise e
        
    async def run_async(self, message: BaseMessage):
        """
        Process a message asynchronously and return the final AI response.
        """
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            response = await self.state_graph.ainvoke({"messages": [message]}, config=config)
            ai_message = response["messages"][-1]
            if isinstance(ai_message, AIMessage):
                return ai_message
            else:
                logger.error("Unexpected message type in async response.")
                raise ValueError("Expected AIMessage in the response.")
        except Exception as e:
            logger.error("Error generating async response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")

