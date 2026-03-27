import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opensearchpy.helpers import async_bulk

from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.parsers import parse_by_extension


class DocumentNotFoundError(Exception):
    """삭제 대상 문서가 존재하지 않을 때 발생한다."""


class DocumentIndexer:
    """문서를 파싱 → 청크 → 임베딩 → OpenSearch 저장하는 인덱서."""

    def __init__(
        self,
        os_client: Any,
        embedder: Embedder,
        index_name: str,
    ) -> None:
        self.os_client = os_client
        self.embedder = embedder
        self.index_name = index_name

    def _generate_doc_id(
        self, filename: str, content: bytes
    ) -> str:
        """파일명(확장자 제외) + SHA256 해시 앞 8자리로 doc_id를 생성한다."""
        stem = Path(filename).stem
        file_hash = hashlib.sha256(content).hexdigest()[:8]
        return f"{stem}_{file_hash}"

    async def index_document(
        self,
        filename: str,
        file_type: str,
        content: bytes,
    ) -> dict[str, Any]:
        """문서를 인덱싱한다."""
        ext = Path(filename).suffix
        text = await asyncio.to_thread(
            parse_by_extension, ext, content
        )

        chunks = chunk_text(text)

        texts = [c.content for c in chunks]
        embeddings = await asyncio.to_thread(
            self.embedder.encode, texts
        )

        doc_id = self._generate_doc_id(filename, content)
        indexed_at = datetime.now(timezone.utc).isoformat()

        await self.os_client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"doc_id": doc_id}}},
        )

        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "doc_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": embeddings[i],
                    "metadata": {
                        "filename": filename,
                        "file_type": file_type,
                        "indexed_at": indexed_at,
                    },
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        await async_bulk(self.os_client, actions)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "status": "indexed",
        }

    async def delete_document(self, doc_id: str) -> None:
        """doc_id에 해당하는 모든 청크를 삭제한다."""
        result = await self.os_client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"doc_id": doc_id}}},
        )
        if result.get("deleted", 0) == 0:
            raise DocumentNotFoundError(
                f"문서를 찾을 수 없습니다: {doc_id}"
            )

    async def list_documents(self) -> list[dict[str, Any]]:
        """인덱싱된 문서 목록을 조회한다."""
        result = await self.os_client.search(
            index=self.index_name,
            body={
                "size": 0,
                "aggs": {
                    "docs": {
                        "terms": {
                            "field": "doc_id",
                            "size": 10000,
                        },
                        "aggs": {
                            "latest": {
                                "top_hits": {
                                    "_source": [
                                        "metadata.filename",
                                        "metadata.indexed_at",
                                    ],
                                    "size": 1,
                                }
                            }
                        },
                    }
                },
            },
        )

        documents: list[dict[str, Any]] = []
        for bucket in result["aggregations"]["docs"]["buckets"]:
            hit = bucket["latest"]["hits"]["hits"][0]["_source"]
            documents.append(
                {
                    "doc_id": bucket["key"],
                    "filename": hit["metadata"]["filename"],
                    "chunks": bucket["doc_count"],
                    "indexed_at": hit["metadata"]["indexed_at"],
                }
            )

        return documents
