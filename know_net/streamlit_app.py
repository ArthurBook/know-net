import pickle
import streamlit as st
from langchain.chat_models import ChatOpenAI
from know_net.graph_building import LLMGraphBuilder
from know_net.graphqa import VecGraphQAChain

# Path to the .env file
env_file = ".env"
PICKLED_BUILDER_PATH = ".pickled_builders/builder.pkl"


st.title("KnowNet")

if st.button("Clear Chat"):
    st.session_state["messages"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with open(PICKLED_BUILDER_PATH, "rb") as f:
    client = pickle.load(f)
client: LLMGraphBuilder
llm = ChatOpenAI(temperature=0)  # type: ignore
qa = VecGraphQAChain.from_llm(llm, graph=client, verbose=True)


if prompt := st.chat_input("Start chat"):
    st.session_state.messages.append(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        source_header = st.empty()
        source_placeholder = st.empty()
        full_response = ""
        try:
            last_message = st.session_state.messages[-1]
        except IndexError:
            last_message = None

        response = qa._call({"query": last_message})
        full_response += response["result"]
        message_placeholder.markdown(full_response)

        references = response["references"]
        if references:
            source_header.markdown("Sources:")
            source_placeholder.markdown(references)

    # st.session_state.messages.append({"role": "assistant", "content": full_response})
