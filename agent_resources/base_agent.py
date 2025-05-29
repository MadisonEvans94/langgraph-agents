# agent_resources/base_agent.py
from abc import ABC, abstractmethod
from langchain.schema import AIMessage
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Dict
from agent_resources.utils import make_llm
import logging
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    # def __init__(self, *, description: str | None = None) -> None:
    #     self.description: str = description or ""

    def visualize_workflow(self, save_path: str = None):
        graph_image = self.state_graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        image = Image(graph_image)
        display(image)
        if save_path:
            with open(save_path, "wb") as f:
                f.write(graph_image)
            print(f"Workflow visualization saved at: {save_path}")

    def _build_llm_dict(self, llm_configs: Dict[str, dict]):
        if llm_configs is None:
            raise ValueError("llm_configs cannot be None.")
        if "default_llm" not in llm_configs:
            raise ValueError("Missing required LLM config: 'default_llm'.")
        llm_dict = {
            name: make_llm(cfg, self.use_llm_provider)
            for name, cfg in llm_configs.items()
        }
        self.llm_dict = llm_dict

    @abstractmethod
    def build_graph(self):
        pass

    def _default_config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def invoke(self, message, **kwargs) -> AIMessage:
        try:
            resp = self.state_graph.invoke(
                {"messages": [message]},
                config=self._default_config(),
            )
            last = resp["messages"][-1]
            if isinstance(last, AIMessage):
                return last
            raise ValueError("Expected AIMessage in response.")
        except Exception:
            logger.error("Error in invoke()", exc_info=True)
            return AIMessage(content="Sorry, I hit an error.")

    def stream(self, message, modes=None):
        try:
            return self.state_graph.stream(
                {"messages": [message]},
                config=self._default_config(),
                stream_mode=modes or ["messages", "updates"],
            )
        except Exception:
            logger.error("Error in stream()", exc_info=True)
            raise

    def _build_system_prompt(self) -> str:
        tools_section = self._build_tools_section()
        return REACT_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)

    def _build_tools_section(self) -> str:
        return "\n".join(
            f"{i}. {t.name}: {t.description}"
            for i, t in enumerate(self.tools, start=1)
        )

    async def ainvoke(self, message) -> AIMessage:
        try:
            resp = await self.state_graph.ainvoke(
                {"messages": [message]},
                config=self._default_config(),
            )
            last = resp["messages"][-1]
            if isinstance(last, AIMessage):
                return last
            raise ValueError("Expected AIMessage in async response.")
        except Exception:
            logger.error("Error in ainvoke()", exc_info=True)
            return AIMessage(content="Sorry, I hit an async error.")