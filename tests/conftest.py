from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.database import Base
from app.main import app
from app.models.slack_thread import SlackThread


# DB URL의 마지막 path 부분(DB명)만 교체
_base, _, _params = settings.database_url.partition("?")
_base = _base.rsplit("/", 1)[0] + "/kasanova_test"
TEST_DATABASE_URL = _base + ("?" + _params if _params else "")


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    app.state.workflow = AsyncMock()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(
            text("DROP TABLE IF EXISTS slack_queries CASCADE")
        )
        await conn.execute(
            text("DROP TABLE IF EXISTS slack_threads CASCADE")
        )
        await conn.execute(
            text("DROP TYPE IF EXISTS query_status_enum CASCADE")
        )
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(
            text("DROP TYPE IF EXISTS query_status_enum CASCADE")
        )
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest.fixture
async def slack_thread(db_session: AsyncSession) -> SlackThread:
    thread = SlackThread(
        slack_channel_id="C_TEST_CHAN",
        slack_thread_ts="1111111111.000001",
        slack_user_id="U_TEST_USER",
    )
    db_session.add(thread)
    await db_session.commit()
    await db_session.refresh(thread)
    return thread
