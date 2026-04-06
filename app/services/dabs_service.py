import json
import logging
from typing import Any

import httpx
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

DABS_API_URL = "https://api.kr.kasa.exchange/dabs"
CACHE_KEY = "dabs:list"
CACHE_TTL = 60 * 60 * 24  # 24시간


class DabsService:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def get_dabs_summary_list(self) -> list[dict[str, Any]]:
        """DABS 목록을 가져온다. Redis 캐시 우선, 없으면 API 호출."""
        cached = await self.redis.get(CACHE_KEY)
        if cached:
            logger.info("DABS 목록 캐시 히트")
            return json.loads(cached)

        logger.info("DABS 목록 캐시 미스 — API 호출")
        summaries = await self._fetch_and_summarize()
        await self.redis.set(
            CACHE_KEY, json.dumps(summaries, ensure_ascii=False), ex=CACHE_TTL
        )
        return summaries

    async def _fetch_and_summarize(self) -> list[dict[str, Any]]:
        """kasa-api에서 DABS 목록을 조회하고 식별용 요약으로 변환한다."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(DABS_API_URL)
            resp.raise_for_status()

        data: list[dict[str, Any]] = resp.json()["data"]
        return [self._to_summary(item) for item in data]

    @staticmethod
    def _to_summary(item: dict[str, Any]) -> dict[str, Any]:
        building = item.get("building") or {}
        address = building.get("address") or {}
        return {
            "code": item.get("code"),
            "name": item.get("name"),
            "status": item.get("status"),
            "ksdDabsNameKr": item.get("ksdDabsNameKr"),
            "ksdDabsNameEn": item.get("ksdDabsNameEn"),
            "buildingName": building.get("name"),
            "buildingSubtitle": building.get("subtitle"),
            "address": address.get("street"),
        }
