"""External tools and API connections for the Agentic RAG system."""
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_DEPTH, WEB_SEARCH_MAX_TOKENS


def get_web_search_tool ()-> TavilySearchResults:
    """
    Initialize and return the Tavily web search tool.
    Tavily is an ai-opyimizd search engine that returns structured
    search results with content snippets, urls, and metadata.ideal for RAG
    Returns:
        TavilySearchResults: Configured web search tool instance.
    """
    return TavilySearchResults(
        max_results=WEB_SEARCH_MAX_RESULTS,
        depth=WEB_SEARCH_DEPTH,
        max_tokens=WEB_SEARCH_MAX_TOKENS
    )