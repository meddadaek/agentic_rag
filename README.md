# Nexus RAG

An agentic RAG portfolio app built with Streamlit, LangGraph, Groq, Tavily, Chroma, and local HuggingFace embeddings.

## Run locally

1. Create a virtual environment and install `pip install -r requirements.txt`.
2. Add `GROQ_API_KEY` and `TAVILY_API_KEY` to `.env`.
3. Start the interface with `streamlit run streamlit_app.py`.

Groq powers generation and Tavily powers web-research fallback. The app can answer from selected documents without Tavily; Tavily is required only when local evidence is insufficient.

## Deployment

Deploy `streamlit_app.py` on Streamlit Community Cloud and add `GROQ_API_KEY` and `TAVILY_API_KEY` in the app's Secrets settings. Uploads are indexed in the current browser session only and are not persisted as a shared knowledge base.

## Supported uploads

TXT, text-based PDF, CSV, JSON, and JSONL files.
