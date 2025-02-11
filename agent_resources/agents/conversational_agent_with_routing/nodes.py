from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

def route_query(state: MessagesState, llm_dict: dict) -> str:

    messages = state.get('messages', [])
    query = messages[-1].content.lower() if messages else ""

    # TODO: Decision logic should go here. Currently just dummy code that routes to 'llm_name' if 'llm_name' exists in query 
    for llm_name in llm_dict.keys():
        if llm_name in query:
            return llm_name

    # Fallback if no match
    return "default_llm"

def llm_node(state: MessagesState, llm) -> dict:
    """
    Calls the provided LLM with the current messages and returns the AIMessage response.
    """
    messages = state.get('messages', [])
    answer = llm.invoke(messages)
    return {"messages": [AIMessage(content=answer.content)]}
