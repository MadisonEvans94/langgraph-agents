import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState

def llm_node(state: MessagesState, llm: BaseChatModel) -> dict:
    """
    Node function to generate a response using the LLM directly.
    """
    if llm is None:
        logging.info("LLM is None in llm_node, returning state as-is.")
        return {}

    # Build the conversation prompt from existing messages
    full_context = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            full_context += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            full_context += f"AI: {msg.content}\n"

    # Invoke LLM with the full conversation context
    answer_msg = llm.invoke([HumanMessage(content=full_context)])

    # Return the new AI message
    return {"messages": [AIMessage(content=answer_msg.content)]}
