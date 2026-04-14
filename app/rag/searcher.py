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

    async def fetch_best_doc(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any] | None:
        """청크 단위 하이브리드 검색으로 가장 관련 있는 문서 **하나**를
        찾은 뒤, 해당 문서의 모든 청크를 `chunk_index` 순으로 병합해
        단일 블록으로 반환한다.

        여러 도메인 지식 문서가 섞여 들어오는 문제를 피하기 위해
        사용한다. 결과가 없으면 None.
        """
        logger.info(
            "[fetch_best_doc] query=%r category=%s top_k=%d"
            " index=%s",
            query,
            category,
            top_k,
            self.index_name,
        )
        hits = await self.search(
            query, top_k=top_k, category=category
        )
        if not hits:
            logger.info(
                "[fetch_best_doc] hybrid search 결과 0건 → None"
            )
            return None

        logger.info(
            "[fetch_best_doc] hybrid 상위 %d건: %s",
            len(hits),
            [
                (h["doc_id"], round(h["score"], 4))
                for h in hits[:5]
            ],
        )

        best = hits[0]
        best_doc_id = best["doc_id"]
        best_score = best["score"]
        best_source = best.get("source")
        logger.info(
            "[fetch_best_doc] top-1 선정: doc_id=%r"
            " source=%r score=%.4f",
            best_doc_id,
            best_source,
            best_score,
        )

        chunks = await self.fetch_by_doc_id(best_doc_id)
        if not chunks:
            logger.warning(
                "[fetch_best_doc] fetch_by_doc_id(%r) 결과 0건."
                " 이는 보통 doc_id 필드 매핑 문제(keyword"
                " subfield 없음) 또는 인덱스 간 값 불일치를"
                " 의미한다. 인덱스=%s",
                best_doc_id,
                self.index_name,
            )
            return None

        logger.info(
            "[fetch_best_doc] doc_id=%r → 청크 %d개 병합",
            best_doc_id,
            len(chunks),
        )

        merged_content = "\n".join(c["content"] for c in chunks)
        return {
            "doc_id": best_doc_id,
            "content": merged_content,
            "source": chunks[0]["source"],
            "score": best_score,
            "chunk_count": len(chunks),
        }

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
