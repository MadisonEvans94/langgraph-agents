from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

def route_query(state: MessagesState) -> str:
    """
    Inspects the last message in the state and returns the name of the next node.
    """
    #TODO: This will be more dynamic and less hard-coded 
    messages = state.get('messages', [])
    query = messages[-1].content.lower() if messages else ""
    return "strong_llm_node" if "strong" in query else "regular_llm_node"

def llm_node(state: MessagesState, llm) -> dict:
    """
    Calls the provided LLM with the current messages and returns the AIMessage response.
    """
    messages = state.get('messages', [])
    answer = llm.invoke(messages)
    return {"messages": [AIMessage(content=answer.content)]}
