from langchain_core.tools import BaseTool, tool

from app.rag.searcher import HybridSearcher


def create_retrieval_tool(searcher: HybridSearcher) -> BaseTool:
    """HybridSearcher를 바인딩한 LangGraph Tool을 생성한다."""

    @tool
    async def retrieval_tool(query: str) -> str:
        """Search internal company documents using hybrid search.
        Use this tool when the user asks about internal company
        knowledge, policies, or documents.
        Do NOT use for general web information or real-time news."""
        try:
            results = await searcher.search(query, top_k=10)
        except Exception as e:
            return f"사내 문서 검색 중 오류가 발생했습니다: {e}"

        if not results:
            return "검색 결과가 없습니다."

        parts: list[str] = []
        for i, doc in enumerate(results, 1):
            parts.append(
                f"[{i}] (score: {doc['score']}) "
                f"[{doc['source']}] {doc['content']}"
            )
        return "\n\n".join(parts)

    return retrieval_tool
