import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

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

    async def test_channel_thread_message_triggers_handler(
        self, client: AsyncClient
    ) -> None:
        """채널 스레드 안에서 멘션 없이 메시지 → 핸들러 호출."""
        body = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel_type": "channel",
                "user": "U12345",
                "text": "스레드 내 후속 질문",
                "channel": "C12345",
                "ts": "1234567890.000010",
                "thread_ts": "1234567890.000001",
            },
        })
        headers = _slack_headers(body, settings.slack_signing_secret)

        with patch(
            "app.api.slack.handle_message_event",
            new_callable=AsyncMock,
        ) as mock_handle:
            response = await client.post(
                "/slack/events", content=body, headers=headers
            )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        mock_handle.assert_called_once()

    async def test_channel_message_without_thread_ignored(
        self, client: AsyncClient
    ) -> None:
        """채널 일반 메시지(thread_ts 없음)는 무시."""
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

        with patch(
            "app.api.slack.handle_message_event",
            new_callable=AsyncMock,
        ) as mock_handle:
            response = await client.post(
                "/slack/events", content=body, headers=headers
            )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        mock_handle.assert_not_called()


class TestHandleMessageEvent:
    """handle_message_event 내부 동작 테스트."""

    async def test_channel_thread_without_bot_participation_ignored(
        self,
    ) -> None:
        """봇이 참여하지 않은 스레드 메시지는 무시."""
        from app.api.slack import handle_message_event

        event = {
            "type": "message",
            "channel_type": "channel",
            "user": "U12345",
            "text": "봇 없는 스레드 메시지",
            "channel": "C12345",
            "ts": "1234567890.000010",
            "thread_ts": "1234567890.000001",
        }

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "app.api.slack.AsyncSessionLocal",
                return_value=mock_session_ctx,
            ),
            patch(
                "app.api.slack.SlackService"
            ) as mock_slack_cls,
        ):
            mock_slack = AsyncMock()
            mock_slack_cls.return_value = mock_slack

            await handle_message_event(
                event=event, workflow=AsyncMock()
            )

            mock_slack.send_loading_message.assert_not_called()

    async def test_dm_response_without_thread_ts(self) -> None:
        """DM 응답 시 thread_ts 없이 메시지를 보내야 한다."""
        from app.api.slack import handle_message_event

        event = {
            "type": "message",
            "channel_type": "im",
            "user": "U12345",
            "text": "DM 질문",
            "channel": "D12345",
            "ts": "1234567890.000002",
        }

        mock_thread = AsyncMock()
        mock_thread.id = "thread-uuid"

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_query_service = AsyncMock()
        mock_query_service.has_active_queries.return_value = False
        mock_query_service.create_query.return_value = AsyncMock(
            id="query-uuid"
        )

        with (
            patch(
                "app.api.slack.AsyncSessionLocal",
                return_value=mock_session_ctx,
            ),
            patch(
                "app.api.slack.SlackService"
            ) as mock_slack_cls,
            patch(
                "app.api.slack.QueryService",
                return_value=mock_query_service,
            ),
            patch(
                "app.api.slack._upsert_slack_thread",
                return_value=mock_thread,
            ),
        ):
            mock_slack = AsyncMock()
            mock_slack.send_loading_message.return_value = {
                "ts": "loading_ts"
            }
            mock_slack_cls.return_value = mock_slack

            mock_workflow = AsyncMock()
            mock_workflow.ainvoke.return_value = {
                "messages": [
                    MagicMock(content="mocked answer")
                ]
            }
            await handle_message_event(
                event=event, workflow=mock_workflow
            )

            # send_loading_message에 thread_ts가 None이어야 함
            mock_slack.send_loading_message.assert_called_once()
            call_kwargs = (
                mock_slack.send_loading_message.call_args[1]
            )
            assert call_kwargs.get("thread_ts") is None

    async def test_channel_thread_with_bot_participation_responds(
        self,
    ) -> None:
        """봇이 참여한 스레드 메시지에는 응답."""
        from app.api.slack import handle_message_event

        event = {
            "type": "message",
            "channel_type": "channel",
            "user": "U12345",
            "text": "후속 질문",
            "channel": "C12345",
            "ts": "1234567890.000010",
            "thread_ts": "1234567890.000001",
        }

        mock_thread = MagicMock()
        mock_thread.id = "thread-uuid"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        # 봇이 참여한 스레드 → SlackThread 존재
        mock_result.scalar_one_or_none.return_value = mock_thread
        mock_db.execute.return_value = mock_result

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_query_service = AsyncMock()
        mock_query_service.has_active_queries.return_value = False
        mock_query_service.create_query.return_value = AsyncMock(
            id="query-uuid"
        )

        with (
            patch(
                "app.api.slack.AsyncSessionLocal",
                return_value=mock_session_ctx,
            ),
            patch(
                "app.api.slack.SlackService"
            ) as mock_slack_cls,
            patch(
                "app.api.slack.QueryService",
                return_value=mock_query_service,
            ),
            patch(
                "app.api.slack._upsert_slack_thread",
                return_value=mock_thread,
            ),
        ):
            mock_slack = AsyncMock()
            mock_slack.send_loading_message.return_value = {
                "ts": "loading_ts"
            }
            mock_slack_cls.return_value = mock_slack

            mock_workflow = AsyncMock()
            mock_workflow.ainvoke.return_value = {
                "messages": [
                    MagicMock(content="mocked answer")
                ]
            }
            await handle_message_event(
                event=event, workflow=mock_workflow
            )

            mock_slack.send_loading_message.assert_called_once()
