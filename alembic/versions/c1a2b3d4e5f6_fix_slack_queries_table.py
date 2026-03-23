"""fix user_queries -> slack_queries table

Revision ID: c1a2b3d4e5f6
Revises: b69168299f32
Create Date: 2026-03-23 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1a2b3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "b69168299f32"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. QueryStatusEnum PostgreSQL ENUM 타입 생성
    query_status_enum = sa.Enum(
        "pending", "processing", "completed", "failed",
        name="query_status_enum",
    )
    query_status_enum.create(op.get_bind(), checkfirst=True)

    # 2. 테이블 rename: user_queries → slack_queries
    op.rename_table("user_queries", "slack_queries")

    # 3. 컬럼 rename: query → query_text
    op.alter_column(
        "slack_queries", "query",
        new_column_name="query_text",
    )

    # 4. 컬럼 rename: answer → response_text
    op.alter_column(
        "slack_queries", "answer",
        new_column_name="response_text",
    )

    # 5. status 타입 변경: VARCHAR(16) → query_status_enum
    op.execute(
        "ALTER TABLE slack_queries"
        " ALTER COLUMN status TYPE query_status_enum"
        " USING status::query_status_enum"
    )

    # 6. completed_at 컬럼 삭제
    op.drop_column("slack_queries", "completed_at")

    # 7. updated_at 컬럼 추가
    op.add_column(
        "slack_queries",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # 8. slack_message_ts 컬럼 추가
    op.add_column(
        "slack_queries",
        sa.Column("slack_message_ts", sa.String(64), nullable=True),
    )

    # 9. slack_response_ts 컬럼 추가
    op.add_column(
        "slack_queries",
        sa.Column("slack_response_ts", sa.String(64), nullable=True),
    )

    # 10. 인덱스 삭제 후 재생성
    op.drop_index(
        "idx_user_queries_slack_thread_id",
        table_name="slack_queries",
    )
    op.drop_index(
        "idx_user_queries_user_status",
        table_name="slack_queries",
    )
    op.create_index(
        "idx_slack_queries_thread_id",
        "slack_queries",
        ["slack_thread_id"],
    )
    op.create_index(
        "idx_slack_queries_user_status",
        "slack_queries",
        ["slack_user_id", "status"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    # 10. 인덱스 삭제 후 재생성 (역순)
    op.drop_index(
        "idx_slack_queries_user_status",
        table_name="slack_queries",
    )
    op.drop_index(
        "idx_slack_queries_thread_id",
        table_name="slack_queries",
    )
    op.create_index(
        "idx_user_queries_user_status",
        "slack_queries",
        ["slack_user_id", "status"],
    )
    op.create_index(
        "idx_user_queries_slack_thread_id",
        "slack_queries",
        ["slack_thread_id"],
    )

    # 9. slack_response_ts 컬럼 삭제
    op.drop_column("slack_queries", "slack_response_ts")

    # 8. slack_message_ts 컬럼 삭제
    op.drop_column("slack_queries", "slack_message_ts")

    # 7. updated_at 컬럼 삭제
    op.drop_column("slack_queries", "updated_at")

    # 6. completed_at 컬럼 추가
    op.add_column(
        "slack_queries",
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # 5. status 타입 변경: query_status_enum → VARCHAR(16)
    op.execute(
        "ALTER TABLE slack_queries"
        " ALTER COLUMN status TYPE varchar(16)"
        " USING status::text"
    )

    # 4. 컬럼 rename: response_text → answer
    op.alter_column(
        "slack_queries", "response_text",
        new_column_name="answer",
    )

    # 3. 컬럼 rename: query_text → query
    op.alter_column(
        "slack_queries", "query_text",
        new_column_name="query",
    )

    # 2. 테이블 rename: slack_queries → user_queries
    op.rename_table("slack_queries", "user_queries")

    # 1. QueryStatusEnum 타입 삭제
    sa.Enum(name="query_status_enum").drop(
        op.get_bind(), checkfirst=True
    )
