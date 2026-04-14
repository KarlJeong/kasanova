import logging
from typing import Any

from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)


class SchemaSearcher:
    """kasanova_schema 인덱스용 얇은 어댑터.

    HybridSearcher에 top_k=10 고정 호출을 위임하고, 결과를
    {table_name, schema} 형식으로 변환해 반환한다.
    `pinned_doc_ids`에 지정된 테이블은 검색 결과에 없어도
    직접 조회해 항상 앞쪽에 포함시킨다.
    """

    def __init__(
        self,
        hybrid: HybridSearcher,
        pinned_doc_ids: list[str] | None = None,
        excluded_doc_ids: list[str] | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.pinned_doc_ids = pinned_doc_ids or []
        self.excluded_doc_ids = set(excluded_doc_ids or [])

    async def search(
        self, query: str
    ) -> list[dict[str, Any]]:
        logger.info("[schema_searcher] query=%r", query)
        raw_hits = await self.hybrid.search(query, top_k=20)

        # 같은 doc_id의 여러 청크가 상위를 도배하지 않도록 dedup.
        # 첫 등장(=최고 점수)만 남긴다.
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for h in raw_hits:
            if h["doc_id"] in seen:
                continue
            if h["doc_id"] in self.excluded_doc_ids:
                logger.info(
                    "[schema_searcher] 제외 테이블 무시: %s",
                    h["doc_id"],
                )
                continue
            seen.add(h["doc_id"])
            hits.append(h)

        pinned_hits: list[dict[str, Any]] = []
        for doc_id in self.pinned_doc_ids:
            if doc_id in seen:
                continue
            chunks = await self.hybrid.fetch_by_doc_id(doc_id)
            if not chunks:
                logger.warning(
                    "[schema_searcher] 고정 테이블 조회 실패: %s",
                    doc_id,
                )
                continue
            # 여러 청크를 chunk_index 순으로 하나의 엔트리로 합쳐 주입.
            merged_content = "\n".join(
                c["content"] for c in chunks
            )
            first = chunks[0]
            pinned_hits.append(
                {
                    "content": merged_content,
                    "doc_id": doc_id,
                    "source": first["source"],
                    "chunk_index": 0,
                    "score": 0.0,
                }
            )
            seen.add(doc_id)
            logger.info(
                "[schema_searcher] 고정 테이블 주입: %s"
                " (청크 %d개 병합)",
                doc_id,
                len(chunks),
            )

        hits = pinned_hits + hits

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
