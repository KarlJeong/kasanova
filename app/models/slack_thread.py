import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SlackThread(Base):
    __tablename__ = "slack_threads"
    __table_args__ = (
        UniqueConstraint(
            "slack_channel_id", "slack_thread_ts", name="uq_channel_thread"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    slack_channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_thread_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    langgraph_thread_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __init__(self, **kwargs):
        if "langgraph_thread_id" not in kwargs:
            channel = kwargs.get("slack_channel_id", "")
            thread_ts = kwargs.get("slack_thread_ts", "")
            kwargs["langgraph_thread_id"] = f"{channel}_{thread_ts}"
        super().__init__(**kwargs)
