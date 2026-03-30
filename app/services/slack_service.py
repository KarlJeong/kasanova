import hashlib
import hmac
import re
import time
from typing import Any

from slack_sdk.web.async_client import AsyncWebClient

from app.core.config import settings


def md_to_slack(text: str) -> str:
    """Markdown → Slack mrkdwn 변환."""
    # **bold** → *bold*  (코드블록 내부는 제외)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # ## heading → *heading*
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # --- 수평선 제거
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    return text


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
        self, channel: str, thread_ts: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": "답변을 준비 중입니다... :hourglass_flowing_sand:",
        }
        if thread_ts is not None:
            kwargs["thread_ts"] = thread_ts
        response = await self.client.chat_postMessage(**kwargs)
        return response.data

    async def update_message(
        self, channel: str, ts: str, text: str
    ) -> dict[str, Any]:
        response = await self.client.chat_update(
            channel=channel,
            ts=ts,
            text=md_to_slack(text),
        )
        return response.data

    async def send_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": md_to_slack(text),
        }
        if thread_ts is not None:
            kwargs["thread_ts"] = thread_ts
        response = await self.client.chat_postMessage(**kwargs)
        return response.data
