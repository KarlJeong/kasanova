from typing import Any, Dict

from langchain_core.tools import tool
from tavily import TavilyClient

from app.core.config import get_settings


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information about the given query."""
    settings = get_settings()
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return client.search(query)
