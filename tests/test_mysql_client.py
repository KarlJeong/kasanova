from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_pool(
    cursor: MagicMock,
) -> MagicMock:
    """aiomysql 커넥션/풀/커서 체인을 Mock으로 구성한다."""

    @asynccontextmanager
    async def acquire_cm():
        conn = MagicMock()

        @asynccontextmanager
        async def cursor_cm(*args, **kwargs):
            yield cursor

        conn.cursor = cursor_cm
        yield conn

    pool = MagicMock()
    pool.acquire = acquire_cm
    pool.close = MagicMock()
    pool.wait_closed = AsyncMock()
    return pool


@pytest.fixture
def mock_pool_factory():
    rows = [
        {"id": i, "name": f"user{i}"} for i in range(150)
    ]
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=rows)
    cursor.description = [("id",), ("name",)]
    pool = _make_pool(cursor)
    return pool, cursor


class TestMySQLClient:
    async def test_execute_select_caps_at_100_rows(
        self, mock_pool_factory
    ) -> None:
        pool, cursor = mock_pool_factory
        with patch(
            "app.db.mysql_client.aiomysql.create_pool",
            AsyncMock(return_value=pool),
        ):
            from app.db.mysql_client import MySQLClient

            client = MySQLClient(
                host="h",
                port=3306,
                db="d",
                user="u",
                password="p",
            )
            await client.connect()
            rows = await client.execute_select(
                "SELECT * FROM t"
            )

        assert rows is not None
        assert len(rows) == 100

    async def test_empty_result_returns_empty_list(
        self, mock_pool_factory
    ) -> None:
        pool, cursor = mock_pool_factory
        cursor.fetchall = AsyncMock(return_value=[])

        with patch(
            "app.db.mysql_client.aiomysql.create_pool",
            AsyncMock(return_value=pool),
        ):
            from app.db.mysql_client import MySQLClient

            client = MySQLClient(
                host="h",
                port=3306,
                db="d",
                user="u",
                password="p",
            )
            await client.connect()
            rows = await client.execute_select(
                "SELECT * FROM t"
            )

        assert rows == []

    async def test_execute_failure_returns_none(
        self, mock_pool_factory
    ) -> None:
        pool, cursor = mock_pool_factory
        cursor.execute = AsyncMock(
            side_effect=RuntimeError("db down")
        )

        with patch(
            "app.db.mysql_client.aiomysql.create_pool",
            AsyncMock(return_value=pool),
        ):
            from app.db.mysql_client import MySQLClient

            client = MySQLClient(
                host="h",
                port=3306,
                db="d",
                user="u",
                password="p",
            )
            await client.connect()
            rows = await client.execute_select(
                "SELECT * FROM t"
            )

        assert rows is None

    async def test_close_releases_pool(
        self, mock_pool_factory
    ) -> None:
        pool, _cursor = mock_pool_factory
        with patch(
            "app.db.mysql_client.aiomysql.create_pool",
            AsyncMock(return_value=pool),
        ):
            from app.db.mysql_client import MySQLClient

            client = MySQLClient(
                host="h",
                port=3306,
                db="d",
                user="u",
                password="p",
            )
            await client.connect()
            await client.close()

        pool.close.assert_called_once()
        pool.wait_closed.assert_awaited_once()
