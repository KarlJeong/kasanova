import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.database import Base
from app.models.slack_query import QueryStatusEnum, SlackQuery
from app.models.slack_thread import SlackThread
from app.services.query_service import QueryService


@pytest.fixture
async def db_engine():
    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS user_queries CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS slack_queries CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS slack_threads CASCADE"))
        await conn.execute(text(
            "DROP TYPE IF EXISTS query_status_enum CASCADE"
        ))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text(
            "DROP TYPE IF EXISTS query_status_enum CASCADE"
        ))
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


class TestQueryService:
    async def test_create_query(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        query = await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="오늘 증시 현황 알려줘",
        )

        assert query.status == QueryStatusEnum.pending
        assert query.query_text == "오늘 증시 현황 알려줘"
        assert query.slack_user_id == "U_TEST_USER"
        assert query.slack_thread_id == slack_thread.id
        assert isinstance(query.id, uuid.UUID)
        assert query.created_at is not None
        assert query.response_text is None

    async def test_has_active_queries_false(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        result = await service.has_active_queries("U_TEST_USER", limit=2)
        assert result is False

    async def test_has_active_queries_true_when_at_limit(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="질문1",
        )
        await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="질문2",
        )

        result = await service.has_active_queries("U_TEST_USER", limit=2)
        assert result is True

    async def test_completed_queries_not_counted_as_active(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        q = await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="완료된 질문",
        )
        await service.update_status(
            q.id, QueryStatusEnum.completed, answer="답변"
        )

        result = await service.has_active_queries("U_TEST_USER", limit=2)
        assert result is False

    async def test_update_status_to_processing(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        q = await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="처리중 질문",
        )
        updated = await service.update_status(
            q.id, QueryStatusEnum.processing
        )
        assert updated.status == QueryStatusEnum.processing

    async def test_update_status_to_completed(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        q = await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="완료 질문",
        )
        updated = await service.update_status(
            q.id, QueryStatusEnum.completed, answer="최종 답변"
        )
        assert updated.status == QueryStatusEnum.completed
        assert updated.response_text == "최종 답변"

    async def test_update_status_to_failed(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        service = QueryService(db_session)
        q = await service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="실패 질문",
        )
        updated = await service.update_status(
            q.id, QueryStatusEnum.failed
        )
        assert updated.status == QueryStatusEnum.failed
