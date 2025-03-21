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

st.set_page_config(page_title="Agentic Demo", layout="wide")
tw.initialize_tailwind()

with st.sidebar:
    tw.write(
        """
        🚀 **Context-based AI Chatbot**
        - Streams responses in real time
        - Routes queries to different LLMs
        - Displays model used
        """,
        classes="text-sm text-gray-300"
    )

tw.write("🧠 Agentic Demo", classes="text-4xl font-bold text-center text-teal-400 my-6")

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

    # Process each line
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
            # Unknown or updates
            pass


def main():
    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(user_input)

        payload = {
            "agent_type": "conversational_agent_with_routing",
            "user_query": user_input
        }
        logger.info(f"Sending user input to {AGENT_SERVICE_URL}/ask_stream => {payload}")

        # Prepare placeholder for partial streaming
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

        # Store assistant message after streaming completes
        st.session_state["messages"].append({"role": "assistant", "content": agent_reply_text})

if __name__ == "__main__":
    main()
