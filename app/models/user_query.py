import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class UserQuery(Base):
    __tablename__ = "user_queries"
    __table_args__ = (
        Index("idx_user_queries_slack_thread_id", "slack_thread_id"),
        Index(
            "idx_user_queries_user_status", "slack_user_id", "status"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    slack_thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("slack_threads.id"), nullable=False
    )
    slack_user_id: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
