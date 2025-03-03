import streamlit as st
import requests
import os
from dotenv import load_dotenv
import logging 
load_dotenv()
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "")
logging.info(f'AGENT_SERVICE_URL: {AGENT_SERVICE_URL}')
st.title("LLM Agent Playground")

agent_types = [
    "conversational_agent_with_routing",
    "react_agent",
]
selected_agent_type = st.selectbox("Choose Agent Type", agent_types)

# Keep conversation in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages in the conversation
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Show Model Used
            if "metadata" in msg and "model_used" in msg["metadata"]:
                model_used = msg["metadata"]["model_used"]
                if model_used:
                    st.markdown(f"**🤖 Model used:** {model_used}")

            # Show Tools Used (directly below model used)
            if "metadata" in msg and "tools_used" in msg["metadata"]:
                tools_used = msg["metadata"]["tools_used"]
                if tools_used:
                    st.markdown(f"**🛠 Tools called for this question:** {', '.join(tools_used)}")

# Chat input
user_input = st.chat_input("Type your question here...")
if user_input:
    # First, show the user message container
    user_container = st.chat_message("user")
    with user_container:
        st.markdown(user_input)
    
    payload = {
        "agent_type": selected_agent_type,
        "user_query": user_input
    }

    try:
        response = requests.post(AGENT_SERVICE_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        agent_reply = data.get("response", "")
        model_used = data.get("model_used", "")
        tools_used = data.get("tools_used", [])
    except Exception as e:
        agent_reply = f"Error calling agent service: {e}"
        model_used = ""
        tools_used = []

    # Now display the assistant’s response, showing which model was used
    assistant_container = st.chat_message("assistant")
    with assistant_container:
        st.markdown(agent_reply)
        if model_used:
            st.markdown(f"**🤖 Model used:** {model_used}")
        if tools_used:
            st.markdown(f"**🛠 Tools called for this question:** {', '.join(tools_used)}")

    # Finally, store these two messages in session_state
    # (the user message, plus metadata about tools)
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "metadata": {}
    })
    # (the assistant message, plus metadata about model and tools)
    st.session_state["messages"].append({
        "role": "assistant",
        "content": agent_reply,
        "metadata": {
            "model_used": model_used,
            "tools_used": tools_used
        }
    })