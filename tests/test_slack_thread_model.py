import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.slack_thread import SlackThread


class TestSlackThreadModel:
    async def test_create_and_read(self, db_session: AsyncSession) -> None:
        thread = SlackThread(
            slack_channel_id="C1234567890",
            slack_thread_ts="1234567890.123456",
            slack_user_id="U1234567890",
        )
        db_session.add(thread)
        await db_session.commit()

        result = await db_session.execute(
            select(SlackThread).where(SlackThread.id == thread.id)
        )
        saved = result.scalar_one()

        assert saved.slack_channel_id == "C1234567890"
        assert saved.slack_thread_ts == "1234567890.123456"
        assert saved.slack_user_id == "U1234567890"
        assert saved.langgraph_thread_id == "C1234567890_1234567890.123456"
        assert isinstance(saved.id, uuid.UUID)

    async def test_duplicate_channel_thread_raises_integrity_error(
        self, db_engine
    ) -> None:
        session_factory = async_sessionmaker(
            db_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with session_factory() as session:
            thread1 = SlackThread(
                slack_channel_id="C_DUP_CHANNEL",
                slack_thread_ts="1111111111.111111",
                slack_user_id="U0000000001",
            )
            session.add(thread1)
            await session.commit()

        async with session_factory() as session:
            thread2 = SlackThread(
                slack_channel_id="C_DUP_CHANNEL",
                slack_thread_ts="1111111111.111111",
                slack_user_id="U0000000002",
            )
            session.add(thread2)
            with pytest.raises(IntegrityError):
                await session.commit()

    async def test_duplicate_langgraph_thread_id_raises_integrity_error(
        self, db_engine
    ) -> None:
        session_factory = async_sessionmaker(
            db_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with session_factory() as session:
            thread1 = SlackThread(
                slack_channel_id="C_LG_DUP",
                slack_thread_ts="2222222222.222222",
                slack_user_id="U0000000001",
            )
            session.add(thread1)
            await session.commit()

        async with session_factory() as session:
            thread2 = SlackThread(
                slack_channel_id="C_LG_DUP",
                slack_thread_ts="2222222222.222222",
                slack_user_id="U0000000003",
            )
            session.add(thread2)
            with pytest.raises(IntegrityError):
                await session.commit()

    async def test_created_at_auto_set(
        self, db_session: AsyncSession
    ) -> None:
        before = datetime.now(timezone.utc)

        thread = SlackThread(
            slack_channel_id="C_AUTO_TS",
            slack_thread_ts="3333333333.333333",
            slack_user_id="U0000000001",
        )
        db_session.add(thread)
        await db_session.commit()

        after = datetime.now(timezone.utc)

        await db_session.refresh(thread)
        assert thread.created_at is not None
        created = thread.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        assert before <= created <= after
