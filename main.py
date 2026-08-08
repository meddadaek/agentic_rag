#!/usr/bin/env python3
"""Agentic RAG System - Main Entry Point

This script builds the vector database, constructs the LangGraph agent,
and runs interactive queries against the system.

NO WIKIPEDIA FILE? NO PROBLEM! Choose any data source below.

Usage:
    python main.py

Prerequisites:
    - Groq API key (set as GROQ_API_KEY env var or enter when prompted)
    - Tavily API key (set as TAVILY_API_KEY env var or enter when prompted)
"""
import os
import sys

# Fix Windows console encoding for emoji output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Step 0: Setup & Configuration
# ---------------------------------------------------------------------------
from src.config import load_api_keys

print("=" * 65)
print("   🤖 AGENTIC RAG SYSTEM - Corrective RAG with LangGraph")
print("=" * 65)
print()

# Load API keys (prompts if not in environment)
groq_key, tavily_key = load_api_keys()
print("✅ API keys configured successfully\n")

# ---------------------------------------------------------------------------
# Step 1: Choose Your Data Source
# ---------------------------------------------------------------------------
print("📚 Step 1: Choose Your Data Source")
print("-" * 40)
print("""
The original project used a Wikipedia JSONL.GZ file from Google Drive.
If you don't have it, pick ANY of these options:

  [1] SYNTHETIC DATA  ← FASTEST (no downloads, works instantly)
      Uses built-in realistic Wikipedia-style articles about India.
      Perfect for testing and understanding the system.

  [2] WIKIPEDIA API   ← RECOMMENDED (fetches live articles)
      Downloads real Wikipedia articles via API.
      Install: pip install wikipedia
      Run first: python download_wikipedia.py

  [3] ORIGINAL FILE   (if you have it)
      Place 'simplewiki-2020-11-01.jsonl.gz' in the data/ folder.

  [4] TEXT FILES      (use your own documents)
      Put .txt files in data/my_docs/ folder.

  [5] CSV FILE        (custom dataset)
      Place your .csv in data/ with 'text' and 'title' columns.
""")

choice = input("Enter choice [1-5] (default: 1): ").strip() or "1"

from src.data_loaders import (
    load_synthetic_data,
    load_from_wikipedia_api,
    load_from_text_folder,
    load_from_csv,
    load_from_jsonl,
    chunk_documents
)
from src.vector_store import build_vector_store, get_retriever

if choice == "1":
    print("\n🚀 Using SYNTHETIC data (instant, no downloads)")
    docs = load_synthetic_data()

elif choice == "2":
    print("\n🌐 Using WIKIPEDIA API")
    print("   Fetching live articles...")
    try:
        docs = load_from_wikipedia_api(topics=None)
    except ImportError:
        print("\n❌ Wikipedia package not installed.")
        print("   Run: pip install wikipedia")
        print("   Or use option 1 (synthetic) instead.")
        sys.exit(1)

elif choice == "3":
    WIKIPEDIA_FILEPATH = "data/simplewiki-2020-11-01.jsonl.gz"
    if not os.path.exists(WIKIPEDIA_FILEPATH):
        print(f"\n❌ File not found: {WIKIPEDIA_FILEPATH}")
        print("   Download from: https://drive.google.com/file/d/1oWBnoxBZ1Mpeond8XDUSO6J9oAjcRDyW")
        print("   Or use option 1 or 2 instead.")
        sys.exit(1)
    from src.vector_store import load_wikipedia_data
    docs = load_wikipedia_data(WIKIPEDIA_FILEPATH, filter_keyword="india")

elif choice == "4":
    folder = input("Enter folder path [default: data/my_docs/]: ").strip() or "data/my_docs/"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"\n⚠️  Folder created: {folder}")
        print("   Add .txt files there and re-run.")
        sys.exit(1)
    docs = load_from_text_folder(folder)

elif choice == "5":
    csv_path = input("Enter CSV path [default: data/documents.csv]: ").strip() or "data/documents.csv"
    if not os.path.exists(csv_path):
        print(f"\n❌ File not found: {csv_path}")
        sys.exit(1)
    text_col = input("Text column name [default: text]: ").strip() or "text"
    title_col = input("Title column name [default: title]: ").strip() or "title"
    docs = load_from_csv(csv_path, text_column=text_col, title_column=title_col)

else:
    print("\n❌ Invalid choice. Using synthetic data (option 1).")
    docs = load_synthetic_data()

print(f"\n   Loaded {len(docs)} documents")

# Chunk documents
chunked_docs = chunk_documents(docs)
print(f"   Chunked into {len(chunked_docs)} segments")

# Build Chroma vector store
chroma_db = build_vector_store(chunked_docs)
print(f"   ✅ Vector database built and persisted to ./wikipedia_db")

# Create retriever
retriever = get_retriever(chroma_db)
print(f"   Retriever configured (top-3, threshold=0.3)")
print()

# ---------------------------------------------------------------------------
# Step 2: Build Agent Graph
# ---------------------------------------------------------------------------
from src.graph import build_agentic_rag_graph

print("🤖 Step 2: Building Agent Graph")
print("-" * 40)
agentic_rag = build_agentic_rag_graph(retriever)
print("   ✅ LangGraph compiled successfully")
print("   Nodes: retrieve → grade_documents → [rewrite_query | generate_answer]")
print("                              ↓ (if needed)")
print("                        web_search → generate_answer → END")
print()

# ---------------------------------------------------------------------------
# Step 3: Interactive Query Loop
# ---------------------------------------------------------------------------
print("💬 Step 3: Interactive Query Mode")
print("-" * 40)
print("Type your question below. Type 'exit' or 'quit' to stop.\n")

while True:
    query = input("📝 Your question: ").strip()

    if query.lower() in ("exit", "quit", "q"):
        print("\n👋 Goodbye!")
        break

    if not query:
        continue

    print()
    print("━" * 50)

    # Invoke the agent graph
    response = agentic_rag.invoke({"question": query})

    print("━" * 50)
    print()
    print("🎯 ANSWER:")
    print(response["generation"])
    print()
    print(f"   Web search used: {response.get('web_search_needed', 'N/A')}")
    print(f"   Context docs: {len(response['documents'])}")
    print()