import hashlib
import hmac
import time
from typing import Any

from slack_sdk.web.async_client import AsyncWebClient

from app.core.config import settings


def verify_signature(
    body: bytes, timestamp: str, signature: str
) -> bool:
    if abs(time.time() - float(timestamp)) > 60 * 5:
        raise TimestampExpiredError

    sig_basestring = f"v0:{timestamp}:{body.decode()}"
    expected = (
        "v0="
        + hmac.new(
            settings.slack_signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    return hmac.compare_digest(expected, signature)


class TimestampExpiredError(Exception):
    pass


class SlackService:
    def __init__(self) -> None:
        self.client = AsyncWebClient(token=settings.slack_bot_token)

    async def send_loading_message(
        self, channel: str, thread_ts: str
    ) -> dict[str, Any]:
        response = await self.client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="답변을 준비 중입니다... :hourglass_flowing_sand:",
        )
        return response.data

    async def update_message(
        self, channel: str, ts: str, text: str
    ) -> dict[str, Any]:
        response = await self.client.chat_update(
            channel=channel,
            ts=ts,
            text=text,
        )
        return response.data

    async def send_message(
        self, channel: str, thread_ts: str, text: str
    ) -> dict[str, Any]:
        response = await self.client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=text,
        )
        return response.data
