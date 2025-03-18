import logging
from langchain_core.messages import BaseMessage, AIMessage
from agent_resources.base_agent import Agent
from agent_resources.prompts import REACT_AGENT_SYSTEM_PROMPT
from agent_resources.tools.tool_registry import ToolRegistry
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

tools = ToolRegistry.get_tools(['tavily_search'])

class ReactAgent(Agent):
    def __init__(self, llm, memory=None, thread_id=None, tools=tools):
        self.tools = tools if tools else []
        self.llm = llm
        self.memory = memory
        self.thread_id = thread_id if thread_id else 'default'
        self.state_graph = self.build_graph()

    def build_graph(self):
        """
        Construct the agent's state graph using a dynamic system prompt.
        """
        try:
            system_prompt = self._build_system_prompt()

            state_graph = create_react_agent(
                self.llm,
                tools=self.tools,
                checkpointer=self.memory,
                state_modifier=system_prompt,  
            )
            return state_graph

        except ValueError as ve:
            logger.error(f"Validation error during agent compilation: {ve}")
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during agent compilation", exc_info=True)
            raise

    def _build_system_prompt(self) -> str:
        """
        Dynamically build the system prompt, injecting the name/description 
        of each tool into the template.
        """
        tools_section = self._build_tools_section()
        system_prompt = REACT_AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)
        return system_prompt

    def _build_tools_section(self) -> str:
        """
        Returns a string enumerating the agent's available tools 
        """
        lines = []
        for i, tool in enumerate(self.tools, start=1):
            line = f"{i}. {tool.name}: {tool.description}"
            lines.append(line)

        return "\n".join(lines)

    def run(self, message: BaseMessage):
        """
        Process a message and return the AI's final response.
        """
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
            return AIMessage(content="Sorry, I encountered an error while processing your request.")