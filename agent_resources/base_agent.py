from abc import ABC, abstractmethod
from langchain.schema import AIMessage
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Dict
from langchain_core.language_models.chat_models import BaseChatModel
from openai import OpenAI
from agent_resources.utils import ChatVLLMWrapper

class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    def visualize_workflow(self, save_path: str = None):
        """
        Visualize the agent's workflow. Optionally save the visualization as an image.

        :param save_path: Optional path to save the image.
        """
        graph_image = self.state_graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API)
        image = Image(graph_image)

        # Display in the notebook (if in a Jupyter environment)
        display(image)

        # Save the image if a save_path is provided
        if save_path:
            with open(save_path, "wb") as f:
                f.write(graph_image)
            print(f"Workflow visualization saved at: {save_path}")

    def _build_llm_dict(self, llm_configs: Dict) -> Dict[str, BaseChatModel]:
        """
        Build a mapping from config names to your wrapped chat-model instances.
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
        return self.llm_dict

    @abstractmethod
    def build_graph(self):
        """
        Compile the LangGraph state‐graph and return an executable graph object.
        """
        pass

    @abstractmethod
    def invoke(self, message, **kwargs) -> AIMessage:
        """
        Abstract sync entrypoint: take a BaseMessage (e.g. HumanMessage) and
        return an AIMessage. Subclasses should call .state_graph.invoke(...) under the hood.
        """
        pass