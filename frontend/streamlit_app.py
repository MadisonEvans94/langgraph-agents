import json
import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv
import st_tailwind as tw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "")

# 1. Set page config
st.set_page_config(page_title="Agentic Demo", layout="wide")

# 2. Initialize Tailwind
tw.initialize_tailwind()

# 3. Inject custom styles (light main area, dark sidebar, custom bubble colors)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #111827 !important;
        background-color: #f3f4f6 !important;
    }

    /* Sidebar: keep dark background, make text pop */
    section[data-testid="stSidebar"] {
        background-color: #202123 !important;
        color: #f3f4f6 !important;
    }

    /* Center content */
    .main-block {
        max-width: 750px;
        margin: 0 auto;
        padding-top: 2rem;
    }

    .main-block h1 {
        text-align: center;
        color: #111827;
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles: use soft gray background */
    .st-chat-message {
        border: 1px solid #e5e7eb !important;
        margin-bottom: 1rem !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        background-color: #f1f5f9 !important; /* <--- updated */
    }

    .st-chat-message:nth-of-type(even) {
        background-color: #e2e8f0 !important; /* assistant bubble slightly different */
    }

    .st-chat-message p {
        color: #111827 !important;
        margin: 0;
    }

    .st-chat-message em {
        color: #facc15 !important;
        font-style: italic !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
    }

    /* Chat input: keep it clean and matching */
    input[data-testid="stChatInput"] {
        background-color: #f9fafb !important;
        border: 1px solid #ccc !important;
        color: #111827 !important;
        border-radius: 6px !important;
    }

    input[data-testid="stChatInput"]:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px #60a5fa !important; /* Tailwind blue-400 */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# 4. Build sidebar content
with st.sidebar:
    # Main agent info
    tw.write(
        """
        **Routing Agent Demo**
        - Running on Intel® Xeon® 
        - Cores per Socket: 64
        - Sockets: 2
        """,
        classes="text-sm text-gray-300 mb-6"
    )

# 5. Create a main container to mimic ChatGPT's centered layout
with tw.container(classes="main-block"):
    # Title
    st.markdown("<h1>Agentic Demo</h1>", unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display conversation so far
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"]=="user" else ":material/robot_2:"):
            st.markdown(msg["content"])

    def stream_response(payload):
        """
        Generator that calls the /ask_stream endpoint with streaming
        and yields partial text and final model name events.
        """
        try:
            response = requests.post(
                f"{AGENT_SERVICE_URL}/ask_stream",
                json=payload,
                timeout=30,
                stream=True
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            yield "error", f"Error calling agent: {exc}"
            return

        text_so_far = ""
        for line in response.iter_lines():
            if not line:
                continue

            try:
                data_json = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip invalid lines

            mode = data_json.get("mode")
            data = data_json.get("data", "")

            if mode == "messages":
                # partial token
                text_so_far += data
                yield "partial", text_so_far

            elif mode == "model_used":
                # Final model name => append it, then we are done
                text_so_far += f"\n\n*(Model used: {data})*"
                yield "final", text_so_far
                break

            elif mode == "error":
                yield "error", data
                break
            else:
                pass

    # Main logic for new user input
    def main():
        user_input = st.chat_input("Ask your question...")
        if user_input:
            # Show user message
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(user_input)

            # Prepare streaming request
            payload = {
                "agent_type": "conversational_agent_with_routing",
                "user_query": user_input
            }
            logger.info(f"Sending user input to {AGENT_SERVICE_URL}/ask_stream => {payload}")

            # Show partial tokens as they arrive
            agent_reply_text = ""
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                placeholder = st.empty()
                for msg_type, content in stream_response(payload):
                    if msg_type in ("partial", "final"):
                        agent_reply_text = content
                        placeholder.markdown(agent_reply_text)
                    elif msg_type == "error":
                        agent_reply_text = content
                        placeholder.markdown(f"**Error**: {content}")
                        break

            # Store assistant message
            st.session_state["messages"].append({"role": "assistant", "content": agent_reply_text})


if __name__ == "__main__":
    main()
