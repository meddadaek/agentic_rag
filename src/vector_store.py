"""Vector database setup: embedding, indexing, and retrieval."""
import gzip
import json
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
    RETRIEVER_K,
    SCORE_THRESHOLD,
)


def build_vector_store(
    chunked_docs: list,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str | None = PERSIST_DIRECTORY,
) -> Chroma:
    """
    Build a Chroma vector database from chunked documents.
    Uses cosine similarity as the distance metric.

    Args:
        chunked_docs: List of Document objects (already chunked).

    Returns:
        Chroma vector database instance.
    """
    if not chunked_docs:
        raise ValueError("Cannot build a vector store without document chunks.")
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunked_docs,
        collection_name=collection_name,
        embedding=embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persist_directory,
    )


def get_retriever(chroma_db: Chroma):
    """
    Create a similarity-threshold retriever.
    Returns top-k documents only if their similarity score >= threshold.

    Args:
        chroma_db: Initialized Chroma database.

    Returns:
        Configured retriever instance.
    """
    retriever = chroma_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVER_K,
            "score_threshold": SCORE_THRESHOLD,
        },
    )
    return retriever


def load_wikipedia_data(
    filepath: str,
    filter_keyword: str = "india",
    max_docs: int = 50,
) -> List[Document]:
    """
    Load and filter Wikipedia articles from a JSONL.GZ file.

    Each line is expected to be a JSON object with at least a 'text' field
    and optionally 'title' and 'id' fields.

    Args:
        filepath: Path to the .jsonl or .jsonl.gz Wikipedia dump file.
        filter_keyword: Case-insensitive keyword to filter articles by.
        max_docs: Maximum number of matching documents to return.

    Returns:
        List of LangChain Document objects.
    """
    docs: List[Document] = []
    keyword = filter_keyword.lower()

    opener = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"
    encoding = "utf8" if filepath.endswith(".gz") else "utf-8"

    with opener(filepath, mode, encoding=encoding) as f:
        for line in f:
            if len(docs) >= max_docs:
                break

            data = json.loads(line.strip())
            text = data.get("text", "")
            title = data.get("title", "Unknown")

            if keyword in text.lower() or keyword in title.lower():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "title": title,
                            "article_id": data.get("id", len(docs)),
                        },
                    )
                )

    print(f"   Loaded {len(docs)} documents matching '{filter_keyword}'")
    return docs
