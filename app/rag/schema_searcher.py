import logging
from typing import Any

from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)


class SchemaSearcher:
    """kasanova_schema 인덱스용 얇은 어댑터.

    HybridSearcher에 top_k=10 고정 호출을 위임하고, 결과를
    {table_name, schema} 형식으로 변환해 반환한다.
    """

    def __init__(self, hybrid: HybridSearcher) -> None:
        self.hybrid = hybrid

    async def search(
        self, query: str
    ) -> list[dict[str, Any]]:
        logger.info("[schema_searcher] query=%r", query)
        hits = await self.hybrid.search(query, top_k=10)

        if not hits:
            logger.info(
                "[schema_searcher] 검색 결과 없음"
            )
            return []

        logger.info(
            "[schema_searcher] %d hits: %s",
            len(hits),
            [h["doc_id"] for h in hits],
        )
        for i, hit in enumerate(hits, 1):
            preview = hit["content"][:80].replace("\n", " ")
            logger.info(
                "[schema_searcher] [%d/%d] score=%.4f"
                " [%s] '%s...'",
                i,
                len(hits),
                hit["score"],
                hit["doc_id"],
                preview,
            )

        return [
            {
                "table_name": hit["doc_id"],
                "schema": hit["content"],
            }
            for hit in hits
        ]
