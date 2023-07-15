import streamlit as st
import pickle
from know_net.graph_building import LLMGraphBuilder
import os

# Path to the .env file
env_file = ".env"

# Read the .env file and set environment variables
with open(env_file, "r") as file:
    for line in file:
        # Remove leading/trailing whitespace and newlines
        line = line.strip()
        if line and not line.startswith("#"):
            # Split the line into key-value pairs
            key, value = line.split("=", 1)

            # Set the environment variable
            os.environ[key] = value


st.title("KnowNet")

if st.button("Clear Chat"):
    st.session_state["messages"] = []


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with open("builder.pkl", "rb") as f:
    client = pickle.load(f)
client: LLMGraphBuilder

if prompt := st.chat_input("Start chat"):
    st.session_state.messages.append(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            last_message = st.session_state.messages[-1]
        except IndexError:
            last_message = None

        response = str(client.search(last_message))
        full_response += response
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
