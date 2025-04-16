import json
import logging
import asyncio
import uuid
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def mcp_react_node_impl(state: dict, llm, mcp_tools) -> dict:
    messages = state.get("messages", [])
    if not messages:
        logger.info("No messages found in state; exiting.")
        return state

    # Log the complete initial state (using repr of message contents)
    logger.info("Initial messages: %s", [repr(getattr(m, "content", "")) for m in messages])

    # Insert system prompt if missing.
    if not isinstance(messages[0], SystemMessage):
        system_prompt = build_system_prompt_for_tools(mcp_tools)
        messages.insert(0, SystemMessage(content=system_prompt))
        logger.info("Inserted system prompt: %s", system_prompt)
    
    # Log the entire messages list after system prompt insertion.
    logger.info("Messages after inserting system prompt: %s", [(type(m).__name__, m.content) for m in messages])
    
    max_loops = 3
    loops = 0
    previous_function_call = None

    while loops < max_loops:
        logger.info("ReAct loop iteration: %d", loops)
        logger.info("Messages (pre-LLM): %s", [(type(m).__name__, m.content) for m in messages])
        last_msg = messages[-1]
        logger.info("Last message type: %s; content: %s", type(last_msg).__name__, getattr(last_msg, "content", ""))

        # Log the entire raw state before invoking the LLM.
        logger.info("Invoking LLM with messages: %s", [repr(m) for m in messages])
        
        if not isinstance(last_msg, AIMessage):
            logger.info("Last message not AIMessage; invoking LLM asynchronously.")
            response = await llm.ainvoke(messages)
            logger.info("Raw LLM response: %s", repr(response))
            logger.info("LLM response content: %s", response.content)
            messages.append(response)
            loops += 1
            continue

        # Extract function_call from the AI message.
        function_call = last_msg.additional_kwargs.get("function_call")
        if not function_call:
            logger.info("No function_call found in last AIMessage; ending loop.")
            break

        logger.info("Function call received: %s", function_call)
        if previous_function_call == function_call:
            logger.info("Repeated function_call detected; breaking loop to avoid infinite recursion.")
            break
        previous_function_call = function_call

        tool_name = function_call.get("name")
        arg_str = function_call.get("arguments", "{}")
        try:
            tool_args = json.loads(arg_str)
            logger.info("Raw parsed arguments for tool '%s': %s", tool_name, tool_args)
            # Explicit conversion for numeric arguments.
            if "a" in tool_args and isinstance(tool_args["a"], str):
                tool_args["a"] = int(tool_args["a"])
            if "b" in tool_args and isinstance(tool_args["b"], str):
                tool_args["b"] = int(tool_args["b"])
            logger.info("Converted arguments for tool '%s': %s", tool_name, tool_args)
        except Exception as e:
            logger.error("Error parsing tool arguments: %s", e, exc_info=True)
            tool_args = {}

        # Log the entire function_call details (including any IDs)
        logger.info("Function call details for tool '%s': %s", tool_name, function_call)

        tool = next((t for t in mcp_tools if t.name == tool_name), None)
        if not tool:
            logger.error("No tool found with name: %s", tool_name)
            messages.append(AIMessage(content=f"No such tool: {tool_name}"))
            break

        tool_call_id = function_call.get("id", str(uuid.uuid4()))
        logger.info("Invoking tool '%s' with call ID: %s, arguments: %s", tool_name, tool_call_id, tool_args)
        try:
            result = await tool.arun(tool_args)
            logger.info("Tool '%s' returned: %s", tool_name, result)
        except Exception as e:
            result = f"Tool call error: {str(e)}"
            logger.error("Error during tool '%s' invocation: %s", tool_name, e, exc_info=True)
        
        if not isinstance(result, str):
            try:
                result = json.dumps(result)
            except Exception:
                result = str(result)

        messages.append(ToolMessage(name=tool_name, content=result, tool_call_id=tool_call_id))
        logger.info("Appended ToolMessage for tool '%s'", tool_name)
        
        new_response = await llm.ainvoke(messages)
        logger.info("LLM re-invoked; new raw response: %s", repr(new_response))
        logger.info("LLM re-invoked; new response content: %s", new_response.content)
        messages.append(new_response)
        loops += 1

    state["messages"] = messages
    logger.info("Final state messages: %s", [(type(m).__name__, m.content) for m in messages])
    return state

def build_system_prompt_for_tools(mcp_tools) -> str:
    lines = [
        "Available MCP tools (provide a function_call with the tool's name and JSON arguments):"
    ]
    for tool in mcp_tools:
        description = tool.description if tool.description else "No description provided."
        lines.append(f"{tool.name}: {description}")
    prompt = "\n".join(lines)
    logger.info("Built system prompt: %s", prompt)
    return prompt
