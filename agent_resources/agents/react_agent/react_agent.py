# agent_resources/agents/react_agent/react_agent.py
import logging
from typing import Dict
from langchain_core.messages import SystemMessage
from agent_resources.base_agent import Agent
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agent_resources.utils import make_llm

logger = logging.getLogger(__name__)

class ReactAgent(Agent):
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
        # build_llm_dict comes from BaseAgent
        self.llm_dict = self.build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        # build_graph is implemented below
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
        return {
            name: make_llm(cfg, self.use_llm_provider)
            for name, cfg in llm_configs.items()
        }

    def _build_system_prompt(self) -> str:
        tools_section = self._build_tools_section()
        return REACT_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)

    def _build_tools_section(self) -> str:
        return "\n".join(
            f"{i}. {t.name}: {t.description}"
            for i, t in enumerate(self.tools, start=1)
        )
