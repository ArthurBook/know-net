import streamlit as st
import pickle
from text_generation import Client

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

if prompt := st.chat_input("Start chat"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            last_message = st.session_state.messages[-1]
        except IndexError:
            last_message = None

        response = client.ask_question(last_message)
        full_response += response
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
