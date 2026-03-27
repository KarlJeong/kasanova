import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opensearchpy import AsyncOpenSearch

from app.core.config import settings
from rag.api.index_router import router as index_router
from rag.embedder import Embedder
from rag.indexer import DocumentIndexer

logger = logging.getLogger(__name__)

INDEX_SCHEMA = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "content": {
                "type": "text",
                "analyzer": "nori",
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                },
            },
            "metadata": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                }
            },
        }
    },
}


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    logger.info("RAG 인덱서 시작 중...")

    embedder = Embedder()
    logger.info("KURE-v1 임베딩 모델 로드 완료")

    os_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        use_ssl=False,
        verify_certs=False,
    )

    index_name = settings.opensearch_index
    if not await os_client.indices.exists(index=index_name):
        await os_client.indices.create(
            index=index_name, body=INDEX_SCHEMA
        )
        logger.info(f"OpenSearch 인덱스 생성: {index_name}")

    app.state.indexer = DocumentIndexer(
        os_client=os_client,
        embedder=embedder,
        index_name=index_name,
    )
    logger.info("RAG 인덱서 준비 완료")

    yield

    await os_client.close()
    logger.info("RAG 인덱서 종료")


app = FastAPI(
    title="KasaNova RAG Indexer",
    lifespan=lifespan,
)
app.include_router(index_router)
