import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

os.environ.setdefault(
    "HF_HOME", os.path.expanduser("~/.cache/huggingface")
)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opensearchpy import AsyncOpenSearch
from sqlalchemy import text

from redis.asyncio import Redis

from app.api.index_router import router as index_router
from app.api.slack import router as slack_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.llm import get_llm
from app.db.mysql_client import MySQLClient
from app.graph.workflow import build_workflow
from app.rag.embedder import Embedder
from app.rag.indexer import DocumentIndexer
from app.rag.schema_searcher import SchemaSearcher
from app.rag.searcher import HybridSearcher
from app.services.dabs_service import DabsService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
settings = get_settings()

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
        # RAG: Embedder + OpenSearch
        embedder = Embedder()
        logger.info("KURE-v1 임베딩 모델 로드 완료")

        os_client = AsyncOpenSearch(
            hosts=[settings.opensearch_url],
            use_ssl=False,
            verify_certs=False,
        )

        async def _require_index(name: str) -> None:
            if not await os_client.indices.exists(index=name):
                raise RuntimeError(
                    f"OpenSearch 인덱스가 존재하지 않습니다: {name}. "
                    f"운영자가 사전에 생성해야 합니다."
                )

        index_name = settings.opensearch_index
        schema_index_name = settings.opensearch_schema_index
        await _require_index(index_name)
        await _require_index(schema_index_name)

        app.state.indexer = DocumentIndexer(
            os_client=os_client,
            embedder=embedder,
            index_name=index_name,
        )
        app.state.schema_indexer = DocumentIndexer(
            os_client=os_client,
            embedder=embedder,
            index_name=schema_index_name,
        )
        logger.info("RAG 인덱서 준비 완료")

        searcher = HybridSearcher(
            os_client=os_client,
            embedder=embedder,
            index_name=index_name,
           search_pipeline="weighted-mean-pipeline",
        )
        schema_hybrid = HybridSearcher(
            os_client=os_client,
            embedder=embedder,
            index_name=schema_index_name,
            search_pipeline="weighted-mean-pipeline",
        )
        schema_searcher = SchemaSearcher(
            schema_hybrid,
            pinned_doc_ids=[
                "kasa_ledger_dabs",
                "kasa_member"
            ],
            excluded_doc_ids=[],
        )
        logger.info("하이브리드 검색기 준비 완료")

        mysql_client = MySQLClient(
            host=settings.mysql_host,
            port=settings.mysql_port,
            db=settings.mysql_db,
            user=settings.mysql_user,
            password=settings.mysql_password,
        )
        await mysql_client.connect()
        logger.info("MySQL 풀 준비 완료")

        redis = Redis.from_url(settings.redis_url)
        dabs_service = DabsService(redis)
        logger.info("Redis 연결 완료")

        llm = get_llm()
        app.state.workflow = build_workflow(
            checkpointer,
            llm,
            searcher,
            schema_searcher,
            mysql_client,
            dabs_service,
        )

        yield

        await redis.aclose()
        await mysql_client.close()
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
