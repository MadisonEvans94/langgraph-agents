import json
import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv
import st_tailwind as tw

# Configure logging for Streamlit
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "")
logger.info(f"AGENT_SERVICE_URL: {AGENT_SERVICE_URL}")

# Configure Streamlit page
st.set_page_config(page_title="Agentic Demo", layout="wide")
tw.initialize_tailwind()

# Sidebar Info
with st.sidebar:
    tw.write("""🚀 **Context-based AI Chatbot**
    - Streams responses in real time.
    - Routes queries to different LLMs.
    - Displays model and tools used.
    """, classes="text-sm text-gray-300")

tw.write("🧠 Agentic Demo", classes="text-4xl font-bold text-center text-teal-400 my-6")

# Keep conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    avatar_icon = ":material/person:" if msg["role"] == "user" else ":material/robot_2:"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask your question...")
if user_input:
    logger.info(f"User input received: {user_input!r}")

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_input)

    payload = {"agent_type": "conversational_agent_with_routing", "user_query": user_input}
    logger.info(f"Sending API request to {AGENT_SERVICE_URL}/ask_stream with payload: {payload}")

    try:
        response = requests.post(f"{AGENT_SERVICE_URL}/ask_stream", json=payload, timeout=30, stream=True)
        response.raise_for_status()
        logger.info("Received successful response from /ask_stream")

        # Streaming response processing
        agent_reply = ""
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            response_placeholder = st.empty()  # Placeholder for updating text

            def stream_generator():
                """Yields streaming tokens as they arrive."""
                for line in response.iter_lines():
                    if line:
                        logger.debug(f"Raw streamed line: {line}")
                        try:
                            json_data = json.loads(line)
                            if "data" in json_data and isinstance(json_data["data"], str):
                                logger.info(f"Received streamed data: {json_data['data']!r}")
                                yield json_data["data"]  # Stream only the content text
                            else:
                                logger.warning(f"Unexpected JSON structure: {json_data}")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {line}")

            # Display streamed content in real time
            agent_reply = st.write_stream(stream_generator())

    except requests.exceptions.RequestException as e:
        agent_reply = f"Error calling agent service: {e}"
        logger.error(f"Error occurred: {e}", exc_info=True)
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            st.markdown(agent_reply)

    # Store messages in session
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": agent_reply})
