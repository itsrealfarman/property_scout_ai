"""
app.py — PropertyScout AI Streamlit front-end.

Run locally:
    export GROQ_API_KEY="your_key_here"
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this folder to a GitHub repo, create a new app pointing at app.py,
    then add GROQ_API_KEY under App settings -> Secrets as:
        GROQ_API_KEY = "your_key_here"
"""

import streamlit as st
from groq import Groq

from tools import load_profile
from agent import run_agent_turn

st.set_page_config(page_title="PropertyScout AI", page_icon="🏠", layout="centered")

# ---------------------------------------------------------------------------
# API key handling — works both locally (env var) and on Streamlit Cloud (secrets)
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
if not api_key:
    import os
    api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.title("🏠 PropertyScout AI")
    st.error(
        "No GROQ_API_KEY found.\n\n"
        "- Locally: set it as an environment variable before running `streamlit run app.py`.\n"
        "- On Streamlit Cloud: add it under **App settings → Secrets** as "
        '`GROQ_API_KEY = "your_key_here"`.\n\n'
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

client = Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # OpenAI-format messages, no system prompt (rebuilt each turn)
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # what's actually rendered in the chat UI

# ---------------------------------------------------------------------------
# Sidebar — profile + tips
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏠 PropertyScout AI")
    st.caption("A memory-aware, tool-using real estate advisor agent.")

    st.subheader("Saved profile")
    profile = load_profile()
    if any([profile.get("preferred_city"), profile.get("budget_pkr"), profile.get("property_type"), profile.get("notes")]):
        if profile.get("preferred_city"):
            st.write(f"**City:** {profile['preferred_city']}")
        if profile.get("budget_pkr"):
            st.write(f"**Budget:** PKR {profile['budget_pkr']:,}")
        if profile.get("property_type"):
            st.write(f"**Property type:** {profile['property_type']}")
        if profile.get("notes"):
            st.write("**Notes:**")
            for n in profile["notes"]:
                st.write(f"- {n}")
    else:
        st.caption("No saved preferences yet — tell the agent what you're looking for.")

    st.divider()
    st.caption(
        "Try asking:\n"
        "- \"I have a 5 marla plot in Faisalabad, what's it roughly worth?\"\n"
        "- \"What's the current property market like in Islamabad?\"\n"
        "- \"I'm looking for a house in Multan under 7,000,000 PKR.\""
    )
    st.divider()
    if st.button("Reset conversation"):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
st.title("🏠 PropertyScout AI")
st.caption("Ask about real estate prices, listings, or market conditions in any city.")

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🔧 Tools used: {', '.join(msg['tools_used'])}")

user_input = st.chat_input("e.g. What's a 10 marla house worth in Lahore?")

if user_input:
    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            current_profile = load_profile()
            try:
                reply_text, tools_used, updated_history = run_agent_turn(
                    client, st.session_state.chat_history, current_profile
                )
            except Exception as e:
                reply_text = f"Something went wrong talking to the model: {e}"
                tools_used = []
                updated_history = st.session_state.chat_history

        st.markdown(reply_text)
        if tools_used:
            st.caption(f"🔧 Tools used: {', '.join(tools_used)}")

    st.session_state.chat_history = updated_history
    st.session_state.display_history.append({
        "role": "assistant",
        "content": reply_text,
        "tools_used": tools_used,
    })
    st.rerun()
