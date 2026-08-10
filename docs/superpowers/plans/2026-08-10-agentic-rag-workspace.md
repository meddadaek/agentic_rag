# Agentic RAG Workspace Implementation Plan

**Goal:** Repair and extend the existing Agentic RAG app into a deployable document-upload and research workspace.

**Architecture:** Preserve the current LangGraph corrective-RAG flow, make its state and routing resilient, and expose it through a session-safe Streamlit chat interface. Use ephemeral session-scoped Chroma indexes for the deployed app so one user's uploads never mix with another's.

**Tech Stack:** Python, Streamlit, LangChain, LangGraph, Chroma, HuggingFace embeddings, Groq, Tavily.

## Tasks

- [ ] Repair ingestion and vector-store lifecycle; add upload parsing coverage.
- [ ] Extend graph state with user-facing activity, web sources, and a bounded answer-validation retry.
- [ ] Replace the rerun-prone Streamlit form with a session-backed source workspace and chat UI.
- [ ] Add deployment guidance and run syntax, unit, and startup checks.
