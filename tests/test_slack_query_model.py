import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.database import Base
from app.models.slack_query import QueryStatusEnum, SlackQuery
from app.models.slack_thread import SlackThread


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


class TestSlackQueryModel:
    async def test_create_and_read(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        query = SlackQuery(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="테스트 질문입니다",
        )
        db_session.add(query)
        await db_session.commit()
        await db_session.refresh(query)

        stmt = select(SlackQuery).where(SlackQuery.id == query.id)
        result = await db_session.execute(stmt)
        fetched = result.scalar_one()

        assert fetched.id == query.id
        assert fetched.query_text == "테스트 질문입니다"
        assert fetched.slack_thread_id == slack_thread.id
        assert isinstance(fetched.id, uuid.UUID)
        assert fetched.created_at is not None

    async def test_status_default_is_pending(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        query = SlackQuery(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="기본 상태 확인",
        )
        db_session.add(query)
        await db_session.commit()
        await db_session.refresh(query)

        assert query.status == QueryStatusEnum.pending

    async def test_updated_at_changes_on_status_update(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        query = SlackQuery(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="updated_at 갱신 확인",
            status=QueryStatusEnum.processing,
        )
        db_session.add(query)
        await db_session.commit()
        await db_session.refresh(query)
        original_updated_at = query.updated_at

        query.status = QueryStatusEnum.completed
        query.response_text = "완료 답변"
        await db_session.commit()
        await db_session.refresh(query)

        assert query.updated_at >= original_updated_at

    async def test_query_text_and_response_text_columns(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        query = SlackQuery(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="컬럼명 확인 질문",
            response_text="컬럼명 확인 답변",
        )
        db_session.add(query)
        await db_session.commit()
        await db_session.refresh(query)

        stmt = select(SlackQuery).where(SlackQuery.id == query.id)
        result = await db_session.execute(stmt)
        fetched = result.scalar_one()

        assert fetched.query_text == "컬럼명 확인 질문"
        assert fetched.response_text == "컬럼명 확인 답변"

    async def test_slack_ts_columns_nullable(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        query = SlackQuery(
            slack_thread_id=slack_thread.id,
            slack_user_id="U_TEST_USER",
            query_text="ts 컬럼 null 확인",
        )
        db_session.add(query)
        await db_session.commit()
        await db_session.refresh(query)

        assert query.slack_message_ts is None
        assert query.slack_response_ts is None

        query.slack_message_ts = "1234567890.000001"
        query.slack_response_ts = "1234567890.000002"
        await db_session.commit()
        await db_session.refresh(query)

        assert query.slack_message_ts == "1234567890.000001"
        assert query.slack_response_ts == "1234567890.000002"

    async def test_fk_constraint_without_thread(
        self, db_session: AsyncSession
    ) -> None:
        query = SlackQuery(
            slack_thread_id=uuid.uuid4(),
            slack_user_id="U_TEST_USER",
            query_text="FK 제약 확인",
        )
        db_session.add(query)
        with pytest.raises(IntegrityError):
            await db_session.commit()

    async def test_count_active_queries(
        self, db_session: AsyncSession, slack_thread: SlackThread
    ) -> None:
        for i in range(2):
            q = SlackQuery(
                slack_thread_id=slack_thread.id,
                slack_user_id="U_TEST_USER",
                query_text=f"처리중 질문 {i}",
                status=QueryStatusEnum.processing,
            )
            db_session.add(q)
        await db_session.commit()

        stmt = select(func.count()).select_from(SlackQuery).where(
            SlackQuery.slack_user_id == "U_TEST_USER",
            SlackQuery.status.in_([
                QueryStatusEnum.pending,
                QueryStatusEnum.processing,
            ]),
        )
        result = await db_session.execute(stmt)
        count = result.scalar_one()

        assert count == 2
