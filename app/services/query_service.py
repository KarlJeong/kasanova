import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.slack_query import QueryStatusEnum, SlackQuery


class QueryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_query(
        self,
        slack_thread_id: uuid.UUID,
        slack_user_id: str,
        query_text: str,
    ) -> SlackQuery:
        slack_query = SlackQuery(
            slack_thread_id=slack_thread_id,
            slack_user_id=slack_user_id,
            query_text=query_text,
            status=QueryStatusEnum.pending,
        )
        self.db.add(slack_query)
        await self.db.commit()
        await self.db.refresh(slack_query)
        return slack_query

    async def has_active_queries(
        self, slack_user_id: str, limit: int = 2
    ) -> bool:
        stmt = select(func.count()).select_from(SlackQuery).where(
            SlackQuery.slack_user_id == slack_user_id,
            SlackQuery.status.in_([
                QueryStatusEnum.pending,
                QueryStatusEnum.processing,
            ]),
        )
        result = await self.db.execute(stmt)
        count = result.scalar_one()
        return count >= limit

    async def update_status(
        self,
        query_id: uuid.UUID,
        status: QueryStatusEnum,
        answer: str | None = None,
    ) -> SlackQuery:
        stmt = select(SlackQuery).where(SlackQuery.id == query_id)
        result = await self.db.execute(stmt)
        slack_query = result.scalar_one()

        slack_query.status = status
        if answer is not None:
            slack_query.response_text = answer

        await self.db.commit()
        await self.db.refresh(slack_query)
        return slack_query
