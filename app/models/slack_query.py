import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class QueryStatusEnum(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class SlackQuery(Base):
    __tablename__ = "slack_queries"
    __table_args__ = (
        Index("idx_slack_queries_thread_id", "slack_thread_id"),
        Index(
            "idx_slack_queries_user_status", "slack_user_id", "status"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    slack_thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("slack_threads.id"), nullable=False
    )
    thread: Mapped["SlackThread"] = relationship(
        back_populates="queries",
    )
    slack_user_id: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    status: Mapped[QueryStatusEnum] = mapped_column(
        SAEnum(QueryStatusEnum, name="query_status_enum"),
        nullable=False,
        default=QueryStatusEnum.pending,
    )
    slack_message_ts: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    slack_response_ts: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
