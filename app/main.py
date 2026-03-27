import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opensearchpy import AsyncOpenSearch
from sqlalchemy import text

from app.api.index_router import router as index_router
from app.api.slack import router as slack_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.llm import get_llm
from app.graph.workflow import build_workflow
from app.rag.embedder import Embedder
from app.rag.indexer import DocumentIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
settings = get_settings()

INDEX_SCHEMA = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "content": {"type": "text", "analyzer": "nori"},
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
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: verify DB connection
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    # LLM + Checkpointer + Workflow
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_db_url
    ) as checkpointer:
        await checkpointer.setup()
        llm = get_llm()
        app.state.workflow = build_workflow(checkpointer, llm)

        # RAG: Embedder + OpenSearch + Indexer
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
                index=index_name,
                body=INDEX_SCHEMA,
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

    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(slack_router)
app.include_router(index_router)


@app.get("/health")
async def health() -> JSONResponse:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "service": settings.app_name},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)},
        )
