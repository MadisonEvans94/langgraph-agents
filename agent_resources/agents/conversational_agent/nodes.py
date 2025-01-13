import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState

def llm_node(state: MessagesState, llm: BaseChatModel) -> MessagesState:
    """
    Node function to generate a response using the LLM directly.
    """
    if llm is None:
        logging.info("LLM is None in llm_node, returning state as-is.")
        return state

    # Build the conversation prompt from existing messages
    full_context = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            full_context += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            full_context += f"AI: {msg.content}\n"

    answer_msg = llm.invoke([HumanMessage(content=full_context)])

    # Append the new AIMessage
    new_messages = state["messages"] + [
        AIMessage(content=answer_msg.content)
    ]
    return {"messages": new_messages}