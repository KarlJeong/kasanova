import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user_query import UserQuery


class QueryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_query(
        self,
        slack_thread_id: uuid.UUID,
        slack_user_id: str,
        query_text: str,
    ) -> UserQuery:
        user_query = UserQuery(
            slack_thread_id=slack_thread_id,
            slack_user_id=slack_user_id,
            query=query_text,
            status="pending",
        )
        self.db.add(user_query)
        await self.db.commit()
        await self.db.refresh(user_query)
        return user_query

    async def has_active_queries(
        self, slack_user_id: str, limit: int = 2
    ) -> bool:
        stmt = select(func.count()).where(
            UserQuery.slack_user_id == slack_user_id,
            UserQuery.status.in_(["pending", "processing"]),
        )
        result = await self.db.execute(stmt)
        count = result.scalar_one()
        return count >= limit

    async def update_status(
        self,
        query_id: uuid.UUID,
        status: str,
        answer: str | None = None,
    ) -> UserQuery:
        stmt = select(UserQuery).where(UserQuery.id == query_id)
        result = await self.db.execute(stmt)
        user_query = result.scalar_one()

        user_query.status = status
        if answer is not None:
            user_query.answer = answer
        if status in ("completed", "failed"):
            user_query.completed_at = datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(user_query)
        return user_query
