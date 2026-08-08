"""LangGraph node functions for the Agentic RAG workflow.

Each function represents a node in the agent graph. They receive the current
GraphState, perform their operation, and return updates to the state.
"""

from typing import List 
from langchain_core.documents import Document

from src.state import GraphState
from src.chains import question_rewriter ,qa_rag_chain, doc_grader
from src.tools import get_web_search_tool


def retrieve(state: GraphState, retriever) -> GraphState:
    """
    Retrieve relevant documents based on the question in the state.
    this node take the user question from the graph state and uses
    the similarity-threshold retriever to find the top-k relevant documents. The retrieved documents are then added to the state.
    

    Args:
        state: Current graph state containing the question.
        retriever: Configured retriever instance.

    Returns:
        Updated graph state with retrieved documents.
    """
    print("\n🔍 --- RETRIEVAL FROM VECTOR DB ---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"   Retrieved {len(documents)} documents")
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState) -> dict:
    """
    Grade retrieved documents for relevance using an LLM grader.

    For each retrieved document, we ask GPT-4o whether it is relevant
    to the user's question. If ANY document is irrelevant OR no documents
    were retrieved, we flag web_search_needed as 'Yes'.

    Only relevant documents are kept in the filtered_docs list.

    Args:
        state: Current graph state with documents and question.

    Returns:
        Updated state with filtered documents and web_search_needed flag.
    """
    print("\n📊 --- CHECK DOCUMENT RELEVANCE TO QUESTION ---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs: List[Document] = []
    web_search_needed = "No"

    if documents:
        for i, doc in enumerate(documents, 1):
            score = doc_grader.invoke({
                "question": question,
                "document": doc.page_content
            })
            grade = score.binary_score

            if grade == "yes":
                print(f"   ✅ Doc {i}: RELEVANT")
                filtered_docs.append(doc)
            else:
                print(f"   ❌ Doc {i}: NOT RELEVANT")
                web_search_needed = "Yes"
    else:
        print("   ⚠️  NO DOCUMENTS RETRIEVED")
        web_search_needed = "Yes"

    print(f"\n   Filtered: {len(filtered_docs)}/{len(documents)} relevant")
    print(f"   Web search needed: {web_search_needed}")

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_needed": web_search_needed
    }


def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrites the user question to an optimized version for web search.
    This node takes the original question from the state, passes it through
    the question rewriter chain, and updates the state with the rewritten question.

    Args:
        state: Current graph state containing the original question.
    Returns:
        Updated graph state with the rewritten question.
    """
    print("\n🔄 --- REWRITING QUERY FOR WEB SEARCH ---")
    question = state["question"]
    rewritten_question = question_rewriter.invoke({"question": question})
    print(f"   Rewritten Question: {rewritten_question}")
    return {"question": rewritten_question}
def web_search(state: GraphState) -> GraphState:
    """
    Performs a web search using the rewritten question.
    This node takes the rewritten question from the state, invokes the web search tool,
    and updates the state with the search results.

    Args:
        state: Current graph state containing the rewritten question.   
    returns:
        Updated graph state with web search results.
    """
    print("\n🌐 --- WEB SEARCH ---")
    question = state["question"]
    documents = state["documents"]

    tv_search = get_web_search_tool()
    docs = tv_search.invoke(question)

    web_results = "\n\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)
    documents.append(web_results_doc)

    print(f"   Found {len(docs)} web results")
    print(f"   Total context docs: {len(documents)}")

    return {"documents": documents, "question": question}

def generate_answer(state: GraphState) -> GraphState:
    """
    Generates an answer to the user's question using the retrieved context.
    This node takes the question and context documents from the state, invokes
    the RAG chain, and updates the state with the generated answer.

    Args:
        state: Current graph state containing the question and context documents.
    Returns:
        Updated graph state with the generated answer.
    """
    print("\n📝 --- GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])
    answer = qa_rag_chain.invoke({"question": question, "context": context})

    print(f"   Generated Answer: {answer}")
    return {"generation": answer, "question": question}
def decide_to_generate(state: GraphState) -> str:
    """
    Conditional routing function for the LangGraph.

    Checks the web_search_needed flag and decides the next node:
      - "Yes" → route to rewrite_query (then web_search → generate_answer)
      - "No"  → route directly to generate_answer

    Args:
        state: Current graph state with web_search_needed flag.

    Returns:
        String name of the next node to execute.
    """
    print("\n🔀 --- ASSESS GRADED DOCUMENTS ---")
    web_search_needed = state["web_search_needed"]

    if web_search_needed == "Yes":
        print("   DECISION: Documents insufficient → REWRITE QUERY → WEB SEARCH")
        return "rewrite_query"
    else:
        print("   DECISION: All docs relevant → GENERATE RESPONSE")
        return "generate_answer"