import logging
from langchain_core.messages import BaseMessage, AIMessage
from agent_resources.base_agent import Agent
from agent_resources.tools.tool_registry import ToolRegistry
from langgraph.prebuilt import create_react_agent
from agent_resources.prompts import RAG_AGENT_PROMPT
logger = logging.getLogger(__name__)


class RAGAgent(Agent):

    def __init__(self, llm, memory=None, use_memory=True):
        retrieve_documents_tool = ToolRegistry.get_tool('retrieve_documents')
        self.tools = [retrieve_documents_tool]
        self.llm = llm
        self.memory = memory
        self.use_memory = use_memory
        self.agent = self.compile_graph()

    def compile_graph(self):
        try:
            system_prompt = RAG_AGENT_PROMPT

            # Conditionally pass checkpointer based on use_memory flag
            agent = create_react_agent(
                self.llm,
                tools=self.tools,
                checkpointer=self.memory if self.use_memory else None,
                state_modifier=system_prompt,
            )
            return agent

        except ValueError as ve:
            logger.error(f"Validation error during agent compilation: {ve}")
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during agent compilation", exc_info=True)
            raise

    def run(self, message: BaseMessage):
        """
        Process a message and return the AI's final response.
        """
        try:
            thread_id = "default"
            # Pass llm and retriever_tool via config so nodes can access them
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            response = self.agent.invoke(
                {"messages": [message]}, config=config)

            ai_message = response["messages"][-1]
            if isinstance(ai_message, AIMessage):
                return ai_message
            else:
                logger.error("Unexpected message type in response.")
                raise ValueError("Expected AIMessage in the response.")

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")
