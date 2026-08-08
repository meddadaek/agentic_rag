"""Graph state definition for the Agentic RAG system.

The state is a TypedDict that gets passed between all nodes in the LangGraph.
Each node reads from and writes to this shared state.
"""
from typing import List, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our agent graph.

    This state object is passed between nodes in the LangGraph workflow.
    Each node can read from and write to specific keys in this state.

    Attributes:
        question: The original user question.
        generation: The LLM-generated final answer.
        web_search_needed: Flag ('yes' or 'no') indicating if web search is required.
        documents: List of retrieved/relevant context documents.
    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]