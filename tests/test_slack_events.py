import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.core.config import settings


def _sign(body: str, timestamp: str, secret: str) -> str:
    sig_basestring = f"v0:{timestamp}:{body}"
    signature = hmac.new(
        secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    return f"v0={signature}"


def _slack_headers(body: str, secret: str) -> dict:
    timestamp = str(int(time.time()))
    signature = _sign(body, timestamp, secret)
    return {
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": signature,
        "Content-Type": "application/json",
    }


class TestSlackSignatureVerification:
    async def test_invalid_signature_returns_403(
        self, client: AsyncClient
    ) -> None:
        body = json.dumps({"type": "event_callback", "event": {}})
        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "v0=invalidsignature",
            "Content-Type": "application/json",
        }
        response = await client.post(
            "/slack/events", content=body, headers=headers
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "Invalid signature"

    async def test_old_timestamp_returns_403(
        self, client: AsyncClient
    ) -> None:
        old_timestamp = str(int(time.time()) - 60 * 6)
        body = json.dumps({"type": "event_callback", "event": {}})
        signature = _sign(
            body, old_timestamp, settings.slack_signing_secret
        )
        headers = {
            "X-Slack-Request-Timestamp": old_timestamp,
            "X-Slack-Signature": signature,
            "Content-Type": "application/json",
        }
        response = await client.post(
            "/slack/events", content=body, headers=headers
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "Request too old"


class TestUrlVerification:
    async def test_url_verification(self, client: AsyncClient) -> None:
        body = json.dumps({
            "type": "url_verification",
            "challenge": "test_challenge_token",
        })
        headers = _slack_headers(body, settings.slack_signing_secret)
        response = await client.post(
            "/slack/events", content=body, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["challenge"] == "test_challenge_token"


class TestEventCallback:
    async def test_bot_event_ignored(self, client: AsyncClient) -> None:
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "user": "U12345",
                "bot_id": "B12345",
                "text": "bot message",
                "channel": "C12345",
                "ts": "1234567890.000001",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)
        response = await client.post(
            "/slack/events", content=body, headers=headers
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_subtype_event_ignored(
        self, client: AsyncClient
    ) -> None:
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "subtype": "message_changed",
                "user": "U12345",
                "text": "edited message",
                "channel": "C12345",
                "ts": "1234567890.000001",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)
        response = await client.post(
            "/slack/events", content=body, headers=headers
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_app_mention_returns_200(
        self, client: AsyncClient
    ) -> None:
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "user": "U12345",
                "text": "<@BOTID> 오늘 증시 현황 요약해줘",
                "channel": "C12345",
                "ts": "1234567890.000001",
                "thread_ts": "1234567890.000001",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)

        with patch(
            "app.api.slack.handle_message_event", new_callable=AsyncMock
        ):
            response = await client.post(
                "/slack/events", content=body, headers=headers
            )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_dm_message_returns_200(
        self, client: AsyncClient
    ) -> None:
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel_type": "im",
                "user": "U12345",
                "text": "DM 질문",
                "channel": "D12345",
                "ts": "1234567890.000002",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)

        with patch(
            "app.api.slack.handle_message_event", new_callable=AsyncMock
        ):
            response = await client.post(
                "/slack/events", content=body, headers=headers
            )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_non_dm_message_ignored(
        self, client: AsyncClient
    ) -> None:
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel_type": "channel",
                "user": "U12345",
                "text": "일반 채널 메시지",
                "channel": "C12345",
                "ts": "1234567890.000003",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)
        response = await client.post(
            "/slack/events", content=body, headers=headers
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_thread_ts_fallback_to_ts(
        self, client: AsyncClient
    ) -> None:
        """thread_ts가 없으면 ts를 사용해야 한다."""
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "user": "U12345",
                "text": "<@BOTID> 질문",
                "channel": "C12345",
                "ts": "9999999999.000001",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)

        with patch(
            "app.api.slack.handle_message_event", new_callable=AsyncMock
        ) as mock_handle:
            response = await client.post(
                "/slack/events", content=body, headers=headers
            )

        assert response.status_code == 200
        call_args = mock_handle.call_args
        event_arg = call_args[1].get("event") or call_args[0][0]
        assert event_arg.get("thread_ts") or event_arg["ts"] == "9999999999.000001"
