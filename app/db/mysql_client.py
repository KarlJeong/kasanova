import logging
from typing import Any

import aiomysql

logger = logging.getLogger(__name__)

_MAX_ROWS = 100


class MySQLClient:
    """aiomysql 풀 기반 비동기 SELECT 실행기."""

    def __init__(
        self,
        host: str,
        port: int,
        db: str,
        user: str,
        password: str,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.user = user
        self.password = password
        self._pool: Any | None = None

    async def connect(self) -> None:
        self._pool = await aiomysql.create_pool(
            host=self.host,
            port=self.port,
            db=self.db,
            user=self.user,
            password=self.password,
            autocommit=False,
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None

    async def execute_select(
        self, sql: str
    ) -> list[dict[str, Any]] | None:
        """SELECT 쿼리를 실행하고 상위 100행까지 반환한다.

        실패 시 예외 없이 None 반환.
        """
        if self._pool is None:
            logger.error(
                "[mysql_client] 풀이 초기화되지 않음"
            )
            return None
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor(
                    aiomysql.DictCursor
                ) as cursor:
                    await cursor.execute(sql)
                    rows = await cursor.fetchall()
            return list(rows[:_MAX_ROWS])
        except Exception:
            logger.exception(
                "[mysql_client] 쿼리 실행 실패: %s", sql
            )
            return None
