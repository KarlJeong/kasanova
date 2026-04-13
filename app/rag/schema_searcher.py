from typing import Any

from app.rag.searcher import HybridSearcher


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
        hits = await self.hybrid.search(query, top_k=10)
        return [
            {
                "table_name": hit["doc_id"],
                "schema": hit["content"],
            }
            for hit in hits
        ]
