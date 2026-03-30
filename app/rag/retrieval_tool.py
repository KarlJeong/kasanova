import logging
from enum import Enum

from langchain_core.tools import BaseTool, tool

from app.rag.searcher import HybridSearcher


class DocCategory(str, Enum):
    """사내 문서 카테고리."""
    hr = "hr"
    ops = "ops"

logger = logging.getLogger(__name__)

_MIN_SCORE = 0.45


def create_retrieval_tool(searcher: HybridSearcher) -> BaseTool:
    """HybridSearcher를 바인딩한 LangGraph Tool을 생성한다."""

    @tool
    async def retrieval_tool(
        query: str, category: DocCategory | None = None
    ) -> str:
        """Search internal company documents using hybrid search.
        Use this tool when the user asks about internal company
        knowledge, policies, or documents.
        Do NOT use for general web information or real-time news.

        category option (only specify when clearly applicable,
        use None when uncertain):
        - "hr": 인사, 연차, 복리후생, 채용, 사내 규정, 임직원 매매 관련
        - "ops": 서비스 운영, 운영 지침, 프로덕트 운영 관련
        - None: 카테고리 불명확 시 전체 검색
        """
        cat_value = category.value if category else None
        logger.info(
            "[retrieval_tool] query=%r, category=%s",
            query,
            cat_value,
        )
        try:
            results = await searcher.search(
                query,
                top_k=5,
                category=cat_value,
            )
        except Exception as e:
            return f"사내 문서 검색 중 오류가 발생했습니다: {e}"

        if not results:
            logger.info("[retrieval_tool] 검색 결과 없음")
            return "검색 결과가 없습니다."

        for i, doc in enumerate(results, 1):
            preview = doc["content"][:80].replace(
                "\n", " "
            )
            logger.info(
                "[retrieval_tool] [%d/%d] score=%.4f"
                " len=%d [%s] '%s...'",
                i,
                len(results),
                doc["score"],
                len(doc["content"]),
                doc["source"],
                preview,
            )

        filtered = [
            r for r in results if r["score"] >= _MIN_SCORE
        ]
        logger.info(
            "[retrieval_tool] %d/%d건 (score >= %.1f)",
            len(filtered),
            len(results),
            _MIN_SCORE,
        )

        if not filtered:
            return "검색 결과가 없습니다."

        parts: list[str] = []
        for i, doc in enumerate(filtered, 1):
            parts.append(
                f"[{i}] (score: {doc['score']}) "
                f"[{doc['source']}] {doc['content']}"
            )
        return "\n\n".join(parts)

    return retrieval_tool
