import json
import tempfile
import uuid

try:
    from llama_index import (
        VectorStoreIndex,
        ServiceContext,
        Document,
        SimpleDirectoryReader,
    )
except ImportError:
    from llama_index.core import (
        VectorStoreIndex,
        ServiceContext,
        Document,
        SimpleDirectoryReader,
    )
from llama_index.core import Settings

from llama_index.core.memory import ChatMemoryBuffer
import streamlit as st
from .common import (
    check_apptoken_from_apikey,
    get_global_datadir,
    get_azure_llm,
    get_azure_embedding,
    write_stream_text,
)
import os
import time
from dotenv import load_dotenv
from .session import PageSessionState

load_dotenv()

llm = get_azure_llm()
embed_model = get_azure_embedding()
Settings.llm = llm
Settings.embed_model = embed_model


@st.cache_resource(ttl="1h", show_spinner=False)
def configure_index(uploaded_files):
    docs = []
    ragchat_dir = get_global_datadir(subpath="ragchat_dir")
    for file in uploaded_files:
        temp_filepath = os.path.join(ragchat_dir, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

    with st.spinner(text="Loading and indexing, please wait a moment..."):
        reader = SimpleDirectoryReader(input_dir=ragchat_dir, recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


def main():
    st.set_page_config(page_title="RAG Chat", page_icon="🗃", layout="wide")
    page_state = PageSessionState("ragchat")
    page_state.initn_attr("messages", [])
    page_state.initn_attr("index", None)
    page_state.initn_attr("chat_mode", "openai")
    page_state.initn_attr("chat_engine", None)

    with st.sidebar:
        st.title("🗃 RAG Chat")
        tab1, tab2 = st.tabs(["参数设置", "关于"])
        with tab1:
            apikey_box = st.empty()
            if not page_state.app_uid:
                apikey = st.query_params.get("apikey")
                if not apikey:
                    apikey = apikey_box.text_input("请输入 API Key", type="password")

                if apikey:
                    appuid = check_apptoken_from_apikey(apikey)
                    if appuid:
                        page_state.app_uid = appuid
                        page_state.apikey = apikey
                        # apikey_box.empty()

            if not page_state.app_uid:
                st.error("Auth is invalid")
                st.stop()
            param_box = st.container()

        with tab2:
            st.image(
                os.path.join(os.path.dirname(__file__), "ragchat.png"),
                use_column_width=True,
            )

    uploaded_files = param_box.file_uploader(
        "上传文件",
        accept_multiple_files=True,
        key="ragchat_file_uploader",
    )
    modes = ["openai", "context", "condense_plus_context", "condense_question", "best", "react"]
    chat_mode = param_box.selectbox(
        "选择聊天模式",
        modes,
        index=modes.index(page_state.chat_mode),
    )
    if chat_mode and param_box.button("改变模式"):
        page_state.chat_mode = chat_mode
        memory = ChatMemoryBuffer.from_defaults(token_limit=32000)
        if chat_mode == "condense_plus_context":
            page_state.chat_engine = page_state.index.as_chat_engine(
                chat_mode=chat_mode,
                memory=memory,
                llm=llm,
                context_prompt=(
                    "You are a chatbot, able to have normal interactions, "
                    "Here are the relevant documents for the context:\n"
                    "{context_str}"
                    "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                ),
                verbose=True,
            )
        elif chat_mode == "context":
            page_state.chat_engine = page_state.index.as_chat_engine(
                chat_mode=chat_mode,
                memory=memory,
                llm=llm,
                verbose=True,
            )
        else:
            page_state.chat_engine = page_state.index.as_chat_engine(
                chat_mode=chat_mode, verbose=True
            )
        
    if page_state.chat_engine and param_box.button("重置"):
        page_state.messages = []
        page_state.chat_engine.reset()

    if uploaded_files:
        index = configure_index(uploaded_files)
        page_state.index = index

    if not page_state.index:
        st.warning("Index is not configured, please upload files")
        st.stop()

    if not page_state.chat_engine:
        page_state.chat_engine = page_state.index.as_chat_engine(
            chat_mode=chat_mode, verbose=True
        )

    if not page_state.messages:
        page_state.add_chat_msg(
            "messages",
            {
                "role": "assistant",
                "content": "Welcome to RAG Chat! ",
            },
        )

    if prompt := st.chat_input(
        "Your question"
    ):  
        page_state.add_chat_msg(
            "messages",
            {
                "role": "user",
                "content": prompt,
            },
        )

    for message in page_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if page_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = page_state.chat_engine.stream_chat(prompt)
                full_response = ""
                for token in response.response_gen:
                    text = token
                    if text is not None:
                        full_response += text
                        placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                page_state.add_chat_msg(
                    "messages", {"role": "assistant", "content": full_response}
                )
