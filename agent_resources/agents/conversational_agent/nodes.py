import logging
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.language_models.chat_models import BaseChatModel

class State(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state: State, llm: BaseChatModel) -> State:
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

    # Actually call the LLM using its 'invoke' method (typical for langchain_core)
    answer_msg = llm.invoke([HumanMessage(content=full_context)])
    logging.info(f"LLM answered with: {answer_msg.content}")

    # Append the new AIMessage
    new_messages = state["messages"] + [
        AIMessage(content=answer_msg.content)
    ]
    return {"messages": new_messages}