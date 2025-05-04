import logging
from typing import Dict
from langchain_core.messages import SystemMessage
from agent_resources.base_agent import Agent
from langgraph.prebuilt import create_react_agent


logger = logging.getLogger(__name__)

class ReactAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id=None,
        tools=None,
        name="react_agent", 
        **kwargs
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
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
            name=self.name
        )

    
