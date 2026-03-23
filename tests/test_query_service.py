import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.slack_query import QueryStatusEnum, SlackQuery
from app.models.slack_thread import SlackThread
from app.services.query_service import QueryService


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
