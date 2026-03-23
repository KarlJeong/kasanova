import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import text

from app.api.slack import router as slack_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.llm import get_llm
from app.graph.workflow import build_workflow

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
        llm = get_llm()
        app.state.workflow = build_workflow(checkpointer, llm)

        yield

    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(slack_router)


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
