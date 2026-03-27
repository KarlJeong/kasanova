import asyncio
from typing import Any

from app.rag.embedder import Embedder


class HybridSearcher:
    """BM25 + k-NN 하이브리드 검색 → RRF 병합."""

    def __init__(
        self,
        os_client: Any,
        embedder: Embedder,
        index_name: str,
    ) -> None:
        self.os_client = os_client
        self.embedder = embedder
        self.index_name = index_name

    async def search(
        self, query: str, top_k: int = 5
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

        response = await self.os_client.search(
            index=self.index_name, body=body
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
