# agent_resources/agents/mcp_agent/nodes.py
import json
import logging
import asyncio
import uuid
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

def mcp_invoke_llm_node(state: dict, llm) -> dict:
    """
    A simple node that calls llm.invoke(...) once (synchronously).
    This returns an AIMessage (which might include a function_call) and appends it to state["messages"].
    """
    messages = state.get("messages", [])
    if not messages:
        return state  # no messages to respond to

    response = llm.invoke(messages)
    messages.append(response)
    state["messages"] = messages
    return state

def mcp_react_node(state: dict, llm, mcp_tools) -> dict:
    """
    Synchronous wrapper that calls our asynchronous ReAct node logic.
    This function wraps the async function using asyncio.run() so that the graph sees a synchronous
    function that returns a dict.
    """
    return asyncio.run(mcp_react_node_impl(state, llm, mcp_tools))

async def mcp_react_node_impl(state: dict, llm, mcp_tools) -> dict:
    """
    The asynchronous "ReAct with function calling" node.
    
    This function checks if the last AIMessage includes a function_call. If so, it:
      - Parses the function_call (extracting the tool name and arguments),
      - Finds the corresponding MCP tool,
      - Awaits the tool call (using tool.arun(...)),
      - Appends a ToolMessage (making sure to include a tool_call_id), and then
      - Re-invokes the LLM (using ainvoke) with the updated conversation.
    
    It repeats this loop up to a maximum number of iterations until the final AI message contains no function_call.
    """
    messages = state.get("messages", [])
    if not messages:
        return state

    # Insert a system prompt at the beginning if no prior SystemMessage
    if not isinstance(messages[0], SystemMessage):
        system_prompt = build_system_prompt_for_tools(mcp_tools)
        messages.insert(0, SystemMessage(content=system_prompt))
    
    max_loops = 3
    loops = 0
    while loops < max_loops:
        last_msg = messages[-1]
        # If the last message is not from the AI, re-invoke the LLM (asynchronously)
        if not isinstance(last_msg, AIMessage):
            response = await llm.ainvoke(messages)  # <-- pass messages directly!
            messages.append(response)
            loops += 1
            continue

        # Check if the last AI message contains a function_call
        function_call = last_msg.additional_kwargs.get("function_call")
        if not function_call:
            logger.debug("mcp_react_node_impl: no function_call found; finishing loop.")
            break  # No function call; we're done

        # Extract tool call details
        tool_name = function_call.get("name")
        arg_str = function_call.get("arguments", "{}")
        try:
            tool_args = json.loads(arg_str)
        except Exception as e:
            logger.error("Error parsing tool arguments: %s", e, exc_info=True)
            tool_args = {}

        # Lookup the MCP tool by name
        tool = next((t for t in mcp_tools if t.name == tool_name), None)
        if not tool:
            error_msg = f"No such tool: {tool_name}"
            messages.append(AIMessage(content=error_msg))
            break

        # Ensure a tool_call_id is present
        tool_call_id = function_call.get("id", str(uuid.uuid4()))

        # Call the tool asynchronously using its arun() method
        try:
            result = await tool.arun(tool_args)
        except Exception as e:
            result = f"Tool call error: {str(e)}"
        if not isinstance(result, str):
            try:
                result = json.dumps(result)
            except Exception:
                result = str(result)

        # Append a ToolMessage with the result, including the tool_call_id
        messages.append(ToolMessage(name=tool_name, content=result, tool_call_id=tool_call_id))
        
        # Re-invoke the LLM with the updated message list (pass messages directly)
        new_response = await llm.ainvoke(messages)
        messages.append(new_response)
        loops += 1

    state["messages"] = messages
    return state

def build_system_prompt_for_tools(mcp_tools) -> str:
    """
    Creates a system prompt that describes the available MCP tools.
    """
    lines = [
        "You have the following MCP tools available. "
        "When you want to use a tool, output a function_call with the tool's name and JSON arguments."
    ]
    for tool in mcp_tools:
        description = tool.description if tool.description else "No description provided."
        lines.append(f"- {tool.name}: {description}")
    return "\n".join(lines)
