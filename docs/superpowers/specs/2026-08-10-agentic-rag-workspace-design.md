# Agentic RAG Workspace Design

## Goal

Create a deployment-ready Streamlit portfolio application where a user selects one knowledge source, indexes it, and chats with an agentic RAG workflow. The application must support uploads as a first-class source while retaining the project's existing demo, Wikipedia, CSV, and JSON/JSONL sources.

## User experience

The page is a polished single-page workspace.

- A branded sidebar contains the active knowledge-source selector, source-specific controls, indexed-document count, and a reset control.
- The main panel introduces the product and presents the document-ingestion state before an index exists.
- Once indexed, the main panel becomes a chat interface. Answers render as assistant messages and retain their history for the browser session.
- Each answer has a compact, expandable evidence panel with retrieved-source excerpts, whether web research was used, and user-facing agent activity messages. It never exposes hidden chain-of-thought.
- Custom CSS in the Streamlit entry point creates a distinctive, accessible dark-on-light visual system without external assets or fragile HTML layouts.

## Knowledge sources

Users select exactly one source per active index:

1. File upload: TXT, PDF, CSV, JSON, and JSONL. File parsing validates format, ignores empty text, records a useful file title, and reports malformed files without crashing the app.
2. Built-in demo data: the existing synthetic set.
3. Wikipedia API: the existing article loader.
4. Local text folder, CSV file, or JSON/JSONL file: retained for local development.

Ingested `Document` objects are stored in Streamlit session state. A separate Build/Update Knowledge Base action chunks and indexes the selected documents. Reruns caused by widgets do not discard the selected documents, chat history, index metadata, or current retriever.

## Agentic workflow

The LangGraph workflow keeps the existing corrective-RAG backbone and makes each transition robust:

1. Retrieve the strongest local chunks for the user's question.
2. Grade context relevance and retain only relevant chunks.
3. If local evidence is insufficient, rewrite the query and run Tavily web research.
4. Generate a grounded response from local and, where required, web evidence.
5. Validate the generated response for useful evidence coverage. A failed validation performs one bounded regeneration using the available evidence; it never creates an unbounded loop.

The graph returns structured activity and source metadata for the UI. The UI emits short status labels such as "Searching your knowledge base" and "Researching the web" rather than raw prompts, hidden reasoning, or model internals.

## Reliability and error handling

- Fix the vector-store module import failure and create a fresh, source-scoped Chroma collection for each re-index so old documents are not silently mixed with the selected source.
- Validate empty uploads, empty extracted PDF text, missing required CSV columns, malformed JSON/JSONL records, missing API credentials, and unavailable external services with actionable user-facing messages.
- Permit local retrieval and answer generation whenever Groq is configured. Tavily is required only when the workflow needs web research; absence or failure of Tavily produces an explicit fallback response rather than an application crash.
- Preserve the original question separately from a rewritten web-search query so displayed chat messages remain faithful to the user's wording.
- Avoid persistent cross-user data in a Streamlit deployment: the default index operates in session memory / a session-scoped temporary collection and reset clears its session state.

## Verification

- Add focused tests for upload parsing and agent-routing helpers using local fakes instead of real API calls.
- Run syntax checks, test suite, and a Streamlit startup smoke test.
- Manually verify the synthetic-data path and a text-file upload path through indexing and a chat response. Web-search behavior is verified with valid Groq and Tavily deployment secrets.

## Deployment

The repository will include a Streamlit Community Cloud-ready entry point and dependency list. README instructions will state required secrets (`GROQ_API_KEY`, `TAVILY_API_KEY`), local launch command, supported file types, and the distinction between local retrieval and the optional web-research fallback.
