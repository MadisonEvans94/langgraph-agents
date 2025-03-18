import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState


def router(state: MessagesState, llm_dict: dict):
    """
    Decides which LLM to use based on the current state and the LLM dictionary.
    """    
    latest_message = state.messages[-1]
    classifier_prompt = f"""
    You are an AI classifier. Determine if message history is needed to answer the following query.
    Respond with only "use_history" or "no_history".
    Query: {latest_message.content}
    """
    
    state.messages[-1] = SystemMessage(content=classifier_prompt)
    llm_choice_response = llm_node(state, "weak_llm")
    
    #Ensure valid response
    llm_decision = llm_choice_response.content.strip().lower()
    """
    if llm_decision not in ["use_history", "no_history"]:
        llm_name = "weak_llm"
    elif llm_decision == "use_history":
        llm_name = "strong_llm"
    elif llm_decision == "no_history":
        llm_name = "weak_llm"
    """
    return {"route": "use_history" if "use_history" in llm_decision else "no_history"}

def llm_node(state: MessagesState, llm):
    """
    Calls the generalized LLM that dynamically selects model and messages and returns the AIMessage response.
    """
    messages_conditional = state.messages if state["route"] == "use_history" else [state.messages[-1]]
    answer = llm.invoke(messages_conditional)
    state.messages.append(AIMessage(content=answer.content))
    return {"messages": [AIMessage(content=answer.content)]}