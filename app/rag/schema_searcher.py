from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.rag.searcher import HybridSearcher

if TYPE_CHECKING:
    from app.rag.reranker import Reranker

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
        reranker: Reranker | None = None,
        rerank_top_k: int = 10,
    ) -> None:
        self.hybrid = hybrid
        self.pinned_doc_ids = pinned_doc_ids or []
        self.excluded_doc_ids = set(excluded_doc_ids or [])
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    async def search(
        self, query: str, log_prefix: str = "",
    ) -> list[dict[str, Any]]:
        pfx = f"[{log_prefix}][schema_searcher]" if log_prefix else "[schema_searcher]"
        logger.info("%s query=%r", pfx, query)
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
                    "%s 제외 테이블 무시: %s",
                    pfx, h["doc_id"],
                )
                continue
            seen.add(h["doc_id"])
            hits.append(h)

        # ── 하이브리드 검색 결과 로그 ──
        if hits:
            logger.info(
                "%s 하이브리드 검색 %d건:",
                pfx, len(hits),
            )
            for i, h in enumerate(hits, 1):
                logger.info(
                    "%s   [%d/%d] hybrid=%.4f [%s]",
                    pfx, i, len(hits),
                    h["score"], h["doc_id"],
                )

        # ── Rerank ──
        if self.reranker and hits:
            # hybrid 순위를 기록해 둔다
            hybrid_rank = {
                h["doc_id"]: rank
                for rank, h in enumerate(hits, 1)
            }
            passages = [h["content"] for h in hits]
            rerank_scores = await self.reranker.arank(
                query, passages
            )
            for h, rs in zip(hits, rerank_scores):
                h["rerank_score"] = rs
            hits.sort(
                key=lambda h: h["rerank_score"],
                reverse=True,
            )
            logger.info(
                "%s rerank 결과 %d건:",
                pfx, len(hits),
            )
            for i, h in enumerate(hits, 1):
                marker = " ✔" if i <= self.rerank_top_k else ""
                h_rank = hybrid_rank[h["doc_id"]]
                logger.info(
                    "%s   [%d→%d/%d] rerank=%.4f"
                    " hybrid=%.4f [%s]%s",
                    pfx, h_rank, i, len(hits),
                    h["rerank_score"],
                    h["score"],
                    h["doc_id"],
                    marker,
                )
            hits = hits[: self.rerank_top_k]
            logger.info(
                "%s 상위 %d개만 LLM에 전달",
                pfx, len(hits),
            )

        pinned_hits: list[dict[str, Any]] = []
        for doc_id in self.pinned_doc_ids:
            if doc_id in seen:
                continue
            chunks = await self.hybrid.fetch_by_doc_id(doc_id)
            if not chunks:
                logger.warning(
                    "%s 고정 테이블 조회 실패: %s",
                    pfx, doc_id,
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
                "%s 고정 테이블 주입: %s"
                " (청크 %d개 병합)",
                pfx, doc_id,
                len(chunks),
            )

        hits = pinned_hits + hits

        if not hits:
            logger.info("%s 검색 결과 없음", pfx)
            return []

        logger.info(
            "%s 최종 전달 %d건: %s",
            pfx,
            len(hits),
            [h["doc_id"] for h in hits],
        )

        return [
            {
                "table_name": hit["doc_id"],
                "schema": hit["content"],
            }
            for hit in hits
        ]

    async def fetch_schemas_by_names(
        self, table_names: list[str], log_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """테이블명 리스트를 받아, 각각의 스키마 문서 전체를
        `chunk_index` 순으로 병합해 반환한다.

        스키마 인덱스에서 `doc_id`는 곧 테이블명이다. 정적
        `pinned_doc_ids` 주입 로직을 다른 호출자(예: KB 기반
        동적 pinning)가 재사용할 수 있도록 분리했다.

        존재하지 않는 테이블명이나 제외 대상은 조용히 스킵한다.
        """
        pfx = f"[{log_prefix}][schema_searcher]" if log_prefix else "[schema_searcher]"
        results: list[dict[str, Any]] = []
        for table_name in table_names:
            if table_name in self.excluded_doc_ids:
                logger.info(
                    "%s 이름 조회 제외: %s",
                    pfx, table_name,
                )
                continue
            chunks = await self.hybrid.fetch_by_doc_id(
                table_name
            )
            if not chunks:
                logger.warning(
                    "%s 이름 조회 실패: %s",
                    pfx, table_name,
                )
                continue
            merged_content = "\n".join(
                c["content"] for c in chunks
            )
            results.append(
                {
                    "table_name": table_name,
                    "schema": merged_content,
                }
            )
            logger.info(
                "%s 이름 조회 성공:"
                " %s (청크 %d개)",
                pfx, table_name,
                len(chunks),
            )
        return results
