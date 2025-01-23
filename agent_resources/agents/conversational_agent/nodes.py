import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState

def llm_node(state: MessagesState, llm: BaseChatModel) -> dict:
    messages = state['messages']
    answer = llm.invoke(messages)
    return {"messages": [AIMessage(content=answer.content)]}
