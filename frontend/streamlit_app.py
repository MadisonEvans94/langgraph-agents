import streamlit as st
import requests
import os
from dotenv import load_dotenv
import logging
import st_tailwind as tw

# 1) Load environment variables
load_dotenv()
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "")
logging.info(f'AGENT_SERVICE_URL: {AGENT_SERVICE_URL}')

# 2) Configure Streamlit's page (title, layout, etc.)
#    NOTE: The actual dark theme comes from config.toml.
st.set_page_config(page_title="Agentic Demo", layout="wide")

# 3) Initialize Tailwind CSS (so we can use tw.write() with Tailwind classes)
tw.initialize_tailwind()

# 4) Optional: Inject some extra CSS to fine-tune backgrounds.
#    For example, to darken the chat input field further:
st.markdown("""
<style>
/* Make the overall app background a bit more uniform */
.stApp {
    background-color: #1F2937 !important; 
    color: #f9fafb !important;
}

/* Make the sidebar a slightly different darker color */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    color: #f9fafb !important;
}

/* Style the chat input box (role="textbox") */
div[role="textbox"] {
    background-color: #2B2B2B !important;
    color: #ffffff !important;
}

/* If you want to tweak the color of the chat bubbles themselves, 
   you can target them with something like:
   .stMarkdown { background-color: #2a2a2a !important; } 
   but be sure to refine selectors carefully to avoid collisions.
*/
</style>
""", unsafe_allow_html=True)

# 5) Sidebar with application information
with st.sidebar:
    tw.write("ℹ️ About This App", classes="text-xl font-semibold text-white mb-4")
    tw.write(
        """
        This is an **LLM-powered chatbot demo** built with Streamlit and Tailwind CSS.

        💡 **Features**:
        - Conversational AI with routing
        - Dark-themed modern UI
        - Displays model and tools used

        🚀 **How to use**:
        1. Type a question in the input box.
        2. The chatbot will respond using an LLM backend.
        3. See details like model and tools used in the response.
        """,
        classes="text-sm text-gray-300"
    )

# 6) Title using Tailwind-wrapped component
tw.write("🧠 Agentic Demo", classes="text-4xl font-bold text-center text-teal-400 my-6")

# Default agent type
selected_agent_type = "conversational_agent_with_routing"

# Keep conversation in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 7) Display existing messages
for msg in st.session_state["messages"]:
    avatar_icon = ":material/person:" if msg["role"] == "user" else ":material/robot_2:"
    
    with st.chat_message(msg["role"], avatar=avatar_icon):  # Use custom icons
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "metadata" in msg:
                if model_used := msg["metadata"].get("model_used"):
                    tw.write(f"✨ Model: {model_used}", classes="text-sm text-gray-300")
                if tools_used := msg["metadata"].get("tools_used"):
                    tw.write(f"🔧 Tools: {', '.join(tools_used)}", classes="text-sm text-gray-300")

# 8) Chat input
user_input = st.chat_input("Ask your question...")
if user_input:
    # Display user message with custom avatar
    with st.chat_message("user", avatar=":material/person:"):
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

    # Display assistant response with custom avatar
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        st.markdown(agent_reply)
        if model_used:
            tw.write(f"✨ Model: {model_used}", classes="text-sm text-gray-300")
        if tools_used:
            tw.write(f"🔧 Tools: {', '.join(tools_used)}", classes="text-sm text-gray-300")

    # Store messages in session state
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "metadata": {}
    })
    st.session_state["messages"].append({
        "role": "assistant",
        "content": agent_reply,
        "metadata": {
            "model_used": model_used,
            "tools_used": tools_used
        }
    })
