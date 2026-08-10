"""LangGraph builder for the Agentic RAG system.

This module constructs the directed graph that defines the agent workflow.
Nodes are connected via edges, with conditional routing based on the
web_search_needed flag.
"""
from langgraph.graph import END, StateGraph

from src.state import GraphState
from src.nodes import decide_after_validation, generate_answer, grade_documents, retrieve, rewrite_query, validate_answer, web_search, decide_to_generate


def build_agentic_rag_graph(retriever):
    """
    Build and compile the Agentic RAG LangGraph.

    Graph Flow:
    -----------
    START → retrieve → grade_documents → [decide_to_generate]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
              (web_search='No')                            (web_search='Yes')
                    │                                           │
                    ↓                                           ↓
            generate_answer ──→ END                    rewrite_query → web_search
                                                                      │
                                                                      ↓
                                                              generate_answer ──→ END

    Args:
        retriever: Configured Chroma retriever instance.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(GraphState)

    # -----------------------------------------------------------------------
    # Step 1: Define all nodes
    # -----------------------------------------------------------------------
    # Each node is a function that receives the current state and returns updates
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("validate_answer", validate_answer)

    # -----------------------------------------------------------------------
    # Step 2: Define edges (transitions between nodes)
    # -----------------------------------------------------------------------
    # Entry point: always start with retrieval
    workflow.set_entry_point("retrieve")

    # After retrieval, always grade the documents
    workflow.add_edge("retrieve", "grade_documents")

    # After grading, use conditional routing based on relevance
    workflow.add_conditional_edges(
        "grade_documents",           # Source node
        decide_to_generate,          # Routing function
        {                            # Mapping: return value → target node
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer"
        }
    )

    # If web search is needed: rewrite → search → generate
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate_answer")

    workflow.add_edge("generate_answer", "validate_answer")
    workflow.add_conditional_edges("validate_answer", decide_after_validation, {"generate_answer": "generate_answer", "end": END})

    # -----------------------------------------------------------------------
    # Step 3: Compile the graph
    # -----------------------------------------------------------------------
    # Compilation validates the graph structure and prepares it for execution
    app = workflow.compile()

    return app
