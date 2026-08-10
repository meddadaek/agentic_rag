"""Streamlit portfolio interface for the existing Agentic RAG workflow."""
from __future__ import annotations

import uuid
from typing import Callable

import streamlit as st
from langchain_core.documents import Document

from src.config import api_keys_status, validate_groq_api_key
from src.data_loaders import chunk_documents, load_from_csv, load_from_jsonl, load_from_text_folder, load_from_uploaded_files, load_from_wikipedia_api, load_synthetic_data
from src.vector_store import build_vector_store, get_retriever

st.set_page_config(page_title="Nexus RAG", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
    .stApp { background: #f7f8fc; color: #182033; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stChatMessage"],
    .stApp [data-testid="stChatMessage"] *, .stApp [data-testid="stExpander"] * {
        color: #182033;
    }
    .stApp input, .stApp textarea, .stApp [data-baseweb="input"] input,
    .stApp [data-baseweb="textarea"] textarea {
        color: #182033 !important;
        -webkit-text-fill-color: #182033;
        background: #ffffff;
    }
    .stApp [data-testid="stChatInput"] { background: #ffffff; border-radius: 12px; }
    .stApp [data-testid="stCaptionContainer"] { color: #5b6477; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #eef2ff !important; }
    .hero { padding: 1.8rem 0 .4rem; } .hero h1 { font-size: 3rem; margin: 0; letter-spacing: -.08rem; }
    .eyebrow { color: #5b5ce2; font-size: .78rem; font-weight: 800; text-transform: uppercase; letter-spacing: .12rem; }
    .source-card { background: white; border: 1px solid #e4e7ef; border-radius: 18px; padding: 1.1rem 1.25rem; margin: .6rem 0; }
    .stButton > button { border-radius: 10px; border: 0; background: #5b5ce2; color: white; font-weight: 700; padding: .55rem 1rem; }
    [data-testid="stChatMessage"] { border-radius: 14px; border: 1px solid #e6e8f0; }
</style>""", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {"documents": [], "retriever": None, "source_label": None, "messages": [], "index_id": uuid.uuid4().hex, "index_ready": False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_documents(documents: list[Document], source_label: str) -> None:
    if not documents:
        st.warning("No usable text was found. Try another file or source.")
        return
    st.session_state.documents = documents
    st.session_state.source_label = source_label
    st.session_state.index_ready = False
    st.success(f"Loaded {len(documents)} document(s). Build the knowledge base when ready.")


def build_index() -> None:
    if not st.session_state.documents:
        st.error("Load a knowledge source before building the index.")
        return
    try:
        with st.spinner("Creating a private session knowledge base…"):
            chunks = chunk_documents(st.session_state.documents)
            vector_db = build_vector_store(chunks, collection_name=f"agentic_rag_{st.session_state.index_id}", persist_directory=None)
            st.session_state.retriever = get_retriever(vector_db)
            st.session_state.index_ready = True
    except Exception as exc:
        st.session_state.index_ready = False
        st.error(
            "The knowledge base could not be created. The first run downloads "
            f"the embedding model; confirm your internet access and try again. Details: {exc}"
        )
        return
    st.success(f"Knowledge base ready: {len(chunks)} searchable chunks.")


def reset_workspace() -> None:
    for key in ("documents", "retriever", "source_label", "messages", "index_ready"):
        st.session_state[key] = [] if key in ("documents", "messages") else None
    st.session_state.index_ready = False
    st.session_state.index_id = uuid.uuid4().hex
    st.rerun()


def source_controls(source: str) -> None:
    if source == "Upload documents":
        files = st.file_uploader("Files", type=["txt", "pdf", "csv", "json", "jsonl"], accept_multiple_files=True, help="PDF, text, table, and JSON formats are supported.")
        if st.button("Load uploads", use_container_width=True) and files:
            set_documents(load_from_uploaded_files(files), "Uploaded documents")
    elif source == "Built-in demo":
        if st.button("Load demo knowledge", use_container_width=True):
            set_documents(load_synthetic_data(), "Built-in demo")
    elif source == "Wikipedia research":
        if st.button("Load Wikipedia articles", use_container_width=True):
            try: set_documents(load_from_wikipedia_api(), "Wikipedia research")
            except Exception as exc: st.error(f"Wikipedia could not be loaded: {exc}")
    else:
        path = st.text_input("Local path", placeholder="data/documents.csv")
        if st.button("Load local source", use_container_width=True) and path:
            try:
                loader: Callable = {"Text folder": load_from_text_folder, "CSV file": load_from_csv, "JSON / JSONL": load_from_jsonl}[source]
                set_documents(loader(path), source)
            except Exception as exc: st.error(f"Could not load this source: {exc}")


def render_chat(mode: str) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("details"):
                with st.expander("Evidence & agent activity"):
                    for activity in message["details"].get("activities", []): st.write(f"• {activity}")
                    st.caption(f"Retrieved context: {message['details'].get('document_count', 0)} chunk(s)")
                    for doc in message["details"].get("documents", [])[:3]:
                        st.markdown(f"**{doc.metadata.get('title', 'Source')}**")
                        st.caption(doc.page_content[:320] + ("…" if len(doc.page_content) > 320 else ""))
                    for result in message["details"].get("web_sources", [])[:3]:
                        if result.get("url"): st.markdown(f"Web source: [{result.get('title', result['url'])}]({result['url']})")

    placeholder = "Message Nexus directly" if mode == "Direct LLM" else "Ask your knowledge base anything"
    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                validate_groq_api_key()
                with st.status("Agent is working…", expanded=True) as status:
                    if mode == "Direct LLM":
                        status.write("Thinking with Groq.")
                        from src.chains import direct_chat_chain
                        response = {"generation": direct_chat_chain.invoke({"question": prompt}), "activities": ["Answered directly with the language model."], "documents": [], "web_sources": []}
                    else:
                        status.write("Searching the selected knowledge source.")
                        from src.graph import build_agentic_rag_graph
                        response = build_agentic_rag_graph(st.session_state.retriever).invoke({"question": prompt, "original_question": prompt, "activities": [], "web_sources": [], "answer_attempts": 0})
                    status.update(label="Answer ready", state="complete", expanded=False)
                answer = response.get("generation", "I couldn't generate an answer from the available evidence.")
            except Exception as exc:
                message = str(exc)
                if mode == "Agentic RAG" and ("HuggingFace" in message or "sentence-transformers" in message):
                    message = "The embedding model could not start. Check your internet connection once so the local model can download, then rebuild the knowledge base."
                answer, response = f"I couldn't complete that request: {message}", {}
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "details": {"activities": response.get("activities", []), "documents": response.get("documents", []), "web_sources": response.get("web_sources", []), "document_count": len(response.get("documents", []))}})


def main() -> None:
    init_state()
    with st.sidebar:
        st.markdown("## ✦ NEXUS RAG")
        st.caption("Document intelligence with web research")
        mode = st.radio("Chat mode", ["Agentic RAG", "Direct LLM"], help="Agentic RAG uses a selected knowledge source and web-research fallback. Direct LLM is a normal conversation with Groq.")
        if mode == "Agentic RAG":
            source = st.radio("Knowledge source", ["Upload documents", "Built-in demo", "Wikipedia research", "Text folder", "CSV file", "JSON / JSONL"])
            source_controls(source)
        st.divider()
        if mode == "Agentic RAG":
            st.metric("Documents loaded", len(st.session_state.documents))
            if st.button("Build knowledge base", use_container_width=True): build_index()
        if st.button("Reset workspace", use_container_width=True): reset_workspace()
        keys = api_keys_status()
        st.caption(f"Groq: {'ready' if keys['GROQ_API_KEY'] else 'missing'} · Tavily: {'ready' if keys['TAVILY_API_KEY'] else 'missing'}")

    title = "Chat freely, or research with evidence."
    description = "Talk directly with the model, or choose a source and let the agent retrieve, verify, and research when needed."
    st.markdown(f"<div class='hero'><div class='eyebrow'>{mode}</div><h1>{title}</h1><p>{description}</p></div>", unsafe_allow_html=True)
    if mode == "Agentic RAG" and not st.session_state.index_ready:
        st.markdown("<div class='source-card'><b>Start by choosing a source.</b><br/>Upload your own files or use one of the built-in research sources. Your documents are not mixed with other sessions.</div>", unsafe_allow_html=True)
    else:
        if mode == "Agentic RAG":
            st.caption(f"Active source: {st.session_state.source_label}")
        render_chat(mode)


if __name__ == "__main__":
    main()
