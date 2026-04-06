from typing import Any, Dict

from langchain_core.tools import BaseTool, tool
from tavily import TavilyClient

from app.core.config import get_settings
from app.services.dabs_service import DabsService


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


def create_dabs_list_tool(dabs_service: DabsService) -> BaseTool:
    """DABS 목록 조회 Tool을 생성한다."""

    @tool
    async def dabs_list() -> str:
        """Retrieve the list of all DABS (Digital Asset-Backed Securities) \
managed by KASA.
Use this tool when the user asks about a specific DABS, building, \
or real estate asset to identify the correct dabs_code.
Returns a summary list containing code, name, address, \
and building info for each DABS."""
        summaries = await dabs_service.get_dabs_summary_list()
        lines = []
        for d in summaries:
            lines.append(
                f"[{d['code']}] {d['name']}"
                f" | {d.get('address', '')}"
                f" | {d.get('buildingSubtitle', '')}"
            )
        return "\n".join(lines)

    return dabs_list
