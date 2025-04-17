import logging
from typing import Dict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from agent_resources.base_agent import Agent
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def make_llm(config: dict, use_llm_provider: bool) -> ChatOpenAI:
    model_id = config.get("model_id") or config.get("model")
    if model_id is None:
        raise ValueError("LLM config must include 'model_id' or 'model'.")
    temperature = config.get("temperature", 0.7)
    max_new_tokens = config.get("max_new_tokens") or config.get("max_tokens", 512)
    api_key = config.get("api_key", "")
    base_url = config.get("base_url")
    params = dict(
        model=model_id,
        temperature=temperature,
        max_tokens=max_new_tokens,
        timeout=None,
        max_retries=2,
        streaming=True,
        api_key=api_key or "EMPTY",
    )
    if not use_llm_provider:
        if not base_url:
            raise ValueError(f"When using vLLM, 'base_url' is required for model {model_id}.")
        params["api_base"] = base_url
    return ChatOpenAI(**params)

class MCPAgent(Agent):
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
        self.llm_dict = self.build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()

    def build_graph(self):
        system_prompt = self._build_system_prompt()
        return create_react_agent(
            self.llm_dict["default_llm"],
            tools=self.tools,
            checkpointer=self.memory,
            prompt=SystemMessage(content=system_prompt),
        )

    def build_llm_dict(self, llm_configs: Dict[str, dict]) -> Dict[str, ChatOpenAI]:
        if llm_configs is None:
            raise ValueError("llm_configs cannot be None.")
        required = ["default_llm", "alternate_llm"]
        for key in required:
            if key not in llm_configs:
                raise ValueError(f"Missing required LLM config: '{key}'.")
        logger.info("🛠️ Building LLM dictionary...")
        llm_dict: Dict[str, ChatOpenAI] = {
            name: make_llm(cfg, self.use_llm_provider)
            for name, cfg in llm_configs.items()
        }
        return llm_dict

    def _build_system_prompt(self) -> str:
        tools_section = self._build_tools_section()
        return REACT_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)

    def _build_tools_section(self) -> str:
        return "\n".join(
            f"{i}. {t.name}: {t.description}"
            for i, t in enumerate(self.tools, start=1)
        )

    def invoke(self, message: BaseMessage) -> AIMessage:
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            resp = self.state_graph.invoke({"messages": [message]}, config=config)
            last = resp["messages"][-1]
            if isinstance(last, AIMessage):
                return last
            raise ValueError("Expected AIMessage in response.")
        except Exception:
            logger.error("Error in invoke()", exc_info=True)
            return AIMessage(content="Sorry, I hit an error.")

    def stream(self, message: BaseMessage):
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            return self.state_graph.stream(
                {"messages": [message]},
                config=config,
                stream_mode=["messages", "updates"],
            )
        except Exception:
            logger.error("Error in stream()", exc_info=True)
            raise

    async def ainvoke(self, message: BaseMessage) -> AIMessage:
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            resp = await self.state_graph.ainvoke({"messages": [message]}, config=config)
            last = resp["messages"][-1]
            if isinstance(last, AIMessage):
                return last
            raise ValueError("Expected AIMessage in async response.")
        except Exception:
            logger.error("Error in ainvoke()", exc_info=True)
            return AIMessage(content="Sorry, I hit an async error.")