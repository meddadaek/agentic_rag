import os
from getpass import getpass

from dotenv import load_dotenv

load_dotenv()


def load_api_keys():
    """Load API keys from env variables or prompt the user for input if not found."""

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        groq_key = getpass("Enter your Groq API key: ")
        os.environ["GROQ_API_KEY"] = groq_key

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        tavily_key = getpass("Enter your Tavily API key: ")
        os.environ["TAVILY_API_KEY"] = tavily_key

    return groq_key, tavily_key


def validate_api_keys() -> None:
    """Ensure required API keys are present in environment variables."""
    missing = [
        name for name in ("GROQ_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )


def validate_groq_api_key() -> None:
    """Ensure answer-generation credentials are available."""
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "Missing GROQ_API_KEY. Add it to .env locally or Streamlit secrets when deployed."
        )


def tavily_is_configured() -> bool:
    """Return whether the optional web-research credential is available."""
    return bool(os.getenv("TAVILY_API_KEY"))


def api_keys_status() -> dict:
    """Return the presence status of required API keys."""
    return {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
    }


# Groq does not provide embeddings — use a local HuggingFace model instead
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.0

# Vector DB configuration
COLLECTION_NAME = "agentic_rag"
PERSIST_DIRECTORY = "wikipedia_db"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

RETRIEVER_K = 3
SCORE_THRESHOLD = 0.3

WEB_SEARCH_MAX_RESULTS = 3
WEB_SEARCH_DEPTH = "advanced"
WEB_SEARCH_MAX_TOKENS = 10000
