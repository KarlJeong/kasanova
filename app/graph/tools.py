from typing import Any, Dict

from langchain_core.tools import BaseTool, tool
from tavily import TavilyClient

from app.core.config import get_settings


def create_web_search_tool() -> BaseTool:
    """TavilyClient를 사용하는 웹 검색 Tool을 생성한다."""

    @tool
    def web_search(query: str) -> Dict[str, Any]:
        """Search the web for factual information, news, or knowledge that cannot be answered from general reasoning alone.
          Do NOT use for math, calculations, or simple logical questions."""
        settings = get_settings()
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        return client.search(query)

    return web_search
