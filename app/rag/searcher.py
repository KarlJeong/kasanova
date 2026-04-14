import asyncio
import logging
from typing import Any

from app.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    BM25 + k-NN 하이브리드 검색 → Weighted Mean 병합.
    curl -X PUT "http://localhost:9200/_search/pipeline/weighted-mean-pipeline" \
      -H 'Content-Type: application/json' \
      -d '{
        "description": "Post processor for hybrid search with Weighted Mean",
        "phase_results_processors": [
          {
            "normalization-processor": {
              "normalization": { "technique": "min_max" },
              "combination": {
                "technique": "arithmetic_mean",
                "parameters": { "weights": [0.3, 0.7] }
              }
            }
          }
        ]
      }'
    """

    def __init__(
        self,
        os_client: Any,
        embedder: Embedder,
        index_name: str,
        search_pipeline: str = "weighted-mean-pipeline",
    ) -> None:
        self.os_client = os_client
        self.embedder = embedder
        self.index_name = index_name
        self.search_pipeline = search_pipeline

    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """하이브리드 검색을 수행하고 상위 결과를 반환한다."""
        query_vector = await asyncio.to_thread(
            self.embedder.encode, [query]
        )

        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"content": query}},
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector[0],
                                    "k": 20,
                                }
                            }
                        },
                    ]
                }
            },
        }

        if category is not None:
            body["post_filter"] = {
                "term": {"metadata.category": category}
            }

        response = await self.os_client.search(
            index=self.index_name,
            body=body,
            params={"search_pipeline": self.search_pipeline},
        )

        results: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                {
                    "content": source["content"],
                    "doc_id": source["doc_id"],
                    "source": source["metadata"]["filename"],
                    "chunk_index": source["chunk_index"],
                    "score": hit["_score"],
                }
            )

        return results

    async def fetch_best_docs(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 10,
        top_n: int = 3,
        gap_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """청크 단위 하이브리드 검색으로 관련 있는 문서 **여러 개**를
        찾은 뒤, 각 문서의 모든 청크를 `chunk_index` 순으로 병합해
        반환한다.

        멀티 도메인 쿼리("A를 청약하고 보유 중이며 배당금 수령한…")
        에서는 정답 문서가 여러 개일 수 있어, 단일 top-1만 반환하면
        일부 도메인이 완전히 누락된다. 대신 다음 규칙으로 여러 건
        반환한다:

        - 청크 hit을 doc_id 단위로 dedup(= 최고 점수만 유지)
        - top-1 점수를 기준으로 `gap_threshold` 상대 컷오프
          (0.7이면 top-1의 70% 이상 점수인 문서만 통과)
        - 단, 최대 `top_n`개로 상한

        단일 도메인 쿼리는 top-1이 압도적이라 컷오프를 통과하는 게
        거의 top-1뿐이므로 기존 동작과 유사하게 유지된다.

        결과가 없으면 빈 리스트 반환.
        """
        logger.info(
            "[fetch_best_docs] query=%r category=%s top_k=%d"
            " top_n=%d gap_threshold=%.2f index=%s",
            query,
            category,
            top_k,
            top_n,
            gap_threshold,
            self.index_name,
        )
        hits = await self.search(
            query, top_k=top_k, category=category
        )
        if not hits:
            logger.info(
                "[fetch_best_docs] hybrid search 결과 0건 → []"
            )
            return []

        # doc_id별 dedup (최고 점수 청크만 유지)
        seen_doc_ids: set[str] = set()
        unique_hits: list[dict[str, Any]] = []
        for h in hits:
            if h["doc_id"] in seen_doc_ids:
                continue
            seen_doc_ids.add(h["doc_id"])
            unique_hits.append(h)

        logger.info(
            "[fetch_best_docs] unique 문서 %d개 (상위 5): %s",
            len(unique_hits),
            [
                (h["doc_id"], round(h["score"], 4))
                for h in unique_hits[:5]
            ],
        )

        # gap-based cutoff
        top_score = unique_hits[0]["score"]
        cutoff = top_score * gap_threshold
        selected = [
            h for h in unique_hits if h["score"] >= cutoff
        ][:top_n]
        logger.info(
            "[fetch_best_docs] gap 컷오프(top=%.4f"
            " × %.2f = %.4f) + top_n=%d 통과: %d개",
            top_score,
            gap_threshold,
            cutoff,
            top_n,
            len(selected),
        )

        results: list[dict[str, Any]] = []
        for sel in selected:
            doc_id = sel["doc_id"]
            chunks = await self.fetch_by_doc_id(doc_id)
            if not chunks:
                logger.warning(
                    "[fetch_best_docs] fetch_by_doc_id(%r)"
                    " 결과 0건. 스킵. 인덱스=%s",
                    doc_id,
                    self.index_name,
                )
                continue
            merged_content = "\n".join(
                c["content"] for c in chunks
            )
            results.append(
                {
                    "doc_id": doc_id,
                    "content": merged_content,
                    "source": chunks[0]["source"],
                    "score": sel["score"],
                    "chunk_count": len(chunks),
                }
            )
            logger.info(
                "[fetch_best_docs] 포함: %s"
                " (score=%.4f, 청크 %d개)",
                doc_id,
                sel["score"],
                len(chunks),
            )

        return results

    async def fetch_by_doc_id(
        self, doc_id: str
    ) -> list[dict[str, Any]]:
        """doc_id의 모든 청크를 chunk_index 순으로 반환.

        인덱스마다 `doc_id`의 매핑이 plain `keyword`인 경우도 있고
        `text` + `.keyword` 서브필드 조합인 경우도 있어, 두 경로를
        모두 시도한다.
        """
        response = await self.os_client.search(
            index=self.index_name,
            body={
                "size": 100,
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"doc_id": doc_id}},
                            {"term": {"doc_id.keyword": doc_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "sort": [{"chunk_index": "asc"}],
            },
        )
        results: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                {
                    "content": source["content"],
                    "doc_id": source["doc_id"],
                    "source": source["metadata"]["filename"],
                    "chunk_index": source["chunk_index"],
                    "score": 0.0,
                }
            )
        return results
