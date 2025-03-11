import json
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langgraph.graph import MessagesState
import requests

def default_llm_node(state: MessagesState, default_llm):
    messages = state['messages']
    response = default_llm.invoke(messages)

    model_name = getattr(default_llm, "model_name", getattr(default_llm, "model", "unknown"))

    return {
        "messages": [
            AIMessage(
                content=response.content,
                # preserve any tool calls
                tool_calls=getattr(response, "tool_calls", []),
                # store the model name used
                additional_kwargs={"model_used": model_name}
            )
        ]
    }




def routing_node(state: MessagesState) -> str:
    """
    Determines the next node by sending the user query to the classifier container.
    The classifier should return a JSON response with an "intent" key.
    Returns a string that maps to the next node.
    In case of errors, an error message is appended to the state and the default node is returned.
    """
    messages = state.get("messages", [])
    query = messages[-1].content if messages else "default query"

    # Adjust the URL based on your container configuration.
    classifier_url = "http://classifier:8000/classify"

    try:
        response = requests.post(classifier_url, json={"query": query}, timeout=5)
        response.raise_for_status()
        data = response.json()
        intent = data.get("intent", "")
        
        # Route based on the classifier's intent.
        if intent.lower() == "research":
            return "alternate_llm_node"
        else:
            return "default_llm_node"
    except requests.exceptions.RequestException as e:
        error_msg = f"Error routing query via classifier: {str(e)}"
        messages.append(AIMessage(content=error_msg))
        return "default_llm_node"

        
def check_tool_calls(state: MessagesState) -> str:
    """
    Checks the last AIMessage in state.
    If it includes any tool_calls, returns 'react_logic_node';
    otherwise returns '__end__' (indicating no tool usage).
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", []):
        return "react_logic_node"
    return "__end__"

def react_logic_node(state: MessagesState, llm, tools, system_prompt, max_iterations: int = 3) -> dict:
    """
    Implements a ReAct-like iterative loop:
      1. Insert a SystemMessage with the system_prompt if none is present.
      2. Process any tool_calls found in the last AIMessage.
      3. Re-invoke the LLM with updated messages.
    """
    messages = state.setdefault("messages", [])
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=system_prompt)
        messages.insert(0, system_msg)

    iterations = 0
    while iterations < max_iterations:
        if not messages:
            break

        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        if not tool_calls:
            break

        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("args", "")
            tool = next((t for t in tools if getattr(t, "name", None) == tool_name), None)
            if not tool:
                continue

            tool_result = tool.invoke(args)
            if not isinstance(tool_result, str):
                try:
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                except TypeError:
                    tool_result_str = str(tool_result)
            else:
                tool_result_str = tool_result

            tool_msg = ToolMessage(
                name=tool_name,
                content=tool_result_str,
                tool_call_id=call["id"]
            )
            messages.append(tool_msg)

        new_response = llm.invoke(messages)
        new_ai_msg = AIMessage(
            content=new_response.content,
            tool_calls=getattr(new_response, "tool_calls", [])
        )
        messages.append(new_ai_msg)
        iterations += 1

    return state

def alternate_llm_node(state: MessagesState, alternate_llm):
    messages = state['messages']
    response = alternate_llm.invoke(messages)

    model_name = getattr(alternate_llm, "model_name", getattr(alternate_llm, "model", "unknown"))

    return {
        "messages": [
            AIMessage(
                content=response.content,
                tool_calls=getattr(response, "tool_calls", []),
                additional_kwargs={"model_used": model_name}
            )
        ]
    }