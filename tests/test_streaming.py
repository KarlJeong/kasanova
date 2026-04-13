import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage


async def _make_stream_events(events):
    """테스트용 async generator: astream_events 모킹."""
    for e in events:
        yield e


class TestStreamResponse:
    """_stream_response() 단위 테스트."""

    async def test_tool_start_retrieval_updates_status(self) -> None:
        """on_tool_start: retrieval_tool → 🔍 내부 문서 검색 중..."""
        from app.api.slack import _stream_response

        events = [
            {"event": "on_tool_start", "name": "retrieval_tool", "data": {}},
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        mock_slack.update_message.assert_any_call(
            channel="C123", ts="ts123", text="🔍 내부 문서 검색 중...",
        )

    async def test_tool_start_web_search_updates_status(self) -> None:
        """on_tool_start: web_search → 🌐 웹 검색 중..."""
        from app.api.slack import _stream_response

        events = [
            {"event": "on_tool_start", "name": "web_search", "data": {}},
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        mock_slack.update_message.assert_any_call(
            channel="C123", ts="ts123", text="🌐 웹 검색 중...",
        )

    async def test_tokens_accumulated_in_buffer(self) -> None:
        """on_chat_model_stream 토큰이 올바르게 누적된다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="안녕")},
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="하세요")},
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        answer, _ = await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        assert answer == "안녕하세요"

    async def test_update_throttled_to_3_seconds(self) -> None:
        """3초 미만 간격에서는 chat.update가 호출되지 않는다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="a")},
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="b")},
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="c")},
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        # time.monotonic()이 항상 같은 값을 반환 → 3초 미경과
        with patch("app.api.slack.time") as mock_time:
            mock_time.monotonic.return_value = 100.0
            await _stream_response(
                workflow=mock_workflow,
                input_messages={"messages": [HumanMessage(content="test")]},
                config={"configurable": {"thread_id": "t1"}},
                slack_service=mock_slack,
                channel="C123",
                loading_ts="ts123",
            )

        # 3초 미경과이므로 on_chat_model_stream에 의한 update는 없어야 함
        mock_slack.update_message.assert_not_called()

    async def test_update_called_after_3_seconds(self) -> None:
        """3초 경과 후에는 chat.update가 호출된다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="a")},
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="b")},
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        # 초기화: 100.0, 첫 토큰 체크: 104.0(3초 초과→업데이트),
        # last_update 갱신 없음(monotonic 재호출 없음),
        # 두 번째 토큰 체크: 104.0(0초 경과→스킵)
        with patch("app.api.slack.time") as mock_time:
            mock_time.monotonic.side_effect = [100.0, 104.0, 104.0]
            await _stream_response(
                workflow=mock_workflow,
                input_messages={"messages": [HumanMessage(content="test")]},
                config={"configurable": {"thread_id": "t1"}},
                slack_service=mock_slack,
                channel="C123",
                loading_ts="ts123",
            )

        # 첫 토큰 시점에 3초 경과 → "a ▌" 업데이트
        mock_slack.update_message.assert_any_call(
            channel="C123", ts="ts123", text="a ▌",
        )
        # 총 1회만 호출 (두 번째 토큰은 3초 미경과)
        assert mock_slack.update_message.call_count == 1

    async def test_final_answer_no_cursor(self) -> None:
        """최종 답변에 커서(▌)가 포함되지 않는다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="최종 답변")},
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        answer, _ = await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        assert "▌" not in answer

    async def test_empty_buffer_returns_fallback(self) -> None:
        """토큰이 없으면 폴백 메시지를 반환한다."""
        from app.api.slack import _stream_response

        events = [
            {"event": "on_graph_start", "data": {}},
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        answer, _ = await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        assert answer == "응답을 생성하지 못했습니다."

    async def test_tool_end_collects_tool_messages(self) -> None:
        """on_tool_end에서 ToolMessage를 수집한다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_tool_end",
                "name": "retrieval_tool",
                "data": {
                    "output": "[1] (score: 0.85) [guide.pdf] 내용",
                },
            },
            {
                "event": "on_tool_end",
                "name": "web_search",
                "data": {
                    "output": "{'results': [{'title': 'T', 'url': 'http://x'}]}",
                },
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        _, tool_messages = await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        assert len(tool_messages) == 2
        assert tool_messages[0].name == "retrieval_tool"
        assert tool_messages[1].name == "web_search"

    async def test_inner_tool_llm_stream_is_filtered(self) -> None:
        """tools 노드 내부 LLM 스트림(SQL 생성 등)은 버퍼에 쌓이지 않는다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "tools"},
                "data": {
                    "chunk": AIMessageChunk(
                        content="SELECT COUNT(*) FROM kasa_member;"
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {
                    "chunk": AIMessageChunk(
                        content="전체 회원은 42명입니다."
                    )
                },
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        answer, _ = await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        assert "SELECT" not in answer
        assert answer == "전체 회원은 42명입니다."

    async def test_empty_token_chunks_ignored(self) -> None:
        """빈 토큰 청크는 버퍼에 추가되지 않는다."""
        from app.api.slack import _stream_response

        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="")},
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="답변")},
            },
        ]
        mock_workflow = MagicMock()
        mock_workflow.astream_events = lambda *a, **kw: _make_stream_events(
            events
        )
        mock_slack = AsyncMock()

        answer, _ = await _stream_response(
            workflow=mock_workflow,
            input_messages={"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "t1"}},
            slack_service=mock_slack,
            channel="C123",
            loading_ts="ts123",
        )

        assert answer == "답변"


class TestHandleMessageEventStreaming:
    """handle_message_event 스트리밍 통합 테스트."""

    async def test_happy_path_streams_and_appends_references(
        self,
    ) -> None:
        """정상 흐름: 스트리밍 → 참조 추출 → 최종 업데이트."""
        from app.api.slack import handle_message_event

        event = {
            "type": "message",
            "channel_type": "im",
            "user": "U12345",
            "text": "질문",
            "channel": "D12345",
            "ts": "1234567890.000002",
        }

        mock_thread = MagicMock()
        mock_thread.id = "thread-uuid"

        mock_db = AsyncMock()
        mock_result = MagicMock()
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

        stream_events = [
            {
                "event": "on_tool_start",
                "name": "retrieval_tool",
                "data": {},
            },
            {
                "event": "on_tool_end",
                "name": "retrieval_tool",
                "data": {
                    "output": "[1] (score: 0.90) [manual.pdf] 내용",
                },
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "call_llm"},
                "data": {"chunk": AIMessageChunk(content="답변입니다")},
            },
        ]

        mock_workflow = MagicMock()
        mock_workflow.astream_events = (
            lambda *a, **kw: _make_stream_events(stream_events)
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

            await handle_message_event(
                event=event, workflow=mock_workflow
            )

            # 최종 업데이트에 참조가 포함되어야 함
            final_call = mock_slack.update_message.call_args_list[-1]
            final_text = final_call[1]["text"]
            assert "답변입니다" in final_text
            assert "manual.pdf" in final_text

            # 쿼리 상태가 completed로 변경
            mock_query_service.update_status.assert_any_call(
                "query-uuid",
                pytest.importorskip(
                    "app.models.slack_query"
                ).QueryStatusEnum.completed,
                answer=final_text,
            )

    async def test_exception_shows_error_message(self) -> None:
        """예외 발생 시 오류 메시지로 업데이트한다."""
        from app.api.slack import handle_message_event

        event = {
            "type": "message",
            "channel_type": "im",
            "user": "U12345",
            "text": "질문",
            "channel": "D12345",
            "ts": "1234567890.000002",
        }

        mock_thread = MagicMock()
        mock_thread.id = "thread-uuid"

        mock_db = AsyncMock()
        mock_result = MagicMock()
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

        async def _failing_stream(*a, **kw):
            raise RuntimeError("LLM 오류")
            yield  # noqa: unreachable — async generator 문법 필요

        mock_workflow = MagicMock()
        mock_workflow.astream_events = _failing_stream

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

            await handle_message_event(
                event=event, workflow=mock_workflow
            )

            mock_slack.update_message.assert_called_with(
                channel="D12345",
                ts="loading_ts",
                text="❌ 오류가 발생했습니다. 다시 시도해 주세요.",
            )
