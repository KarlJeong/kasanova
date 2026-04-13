import logging
import re
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from langchain_core.messages import HumanMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.slack_query import QueryStatusEnum
from app.models.slack_thread import SlackThread
from app.services.query_service import QueryService
from app.services.slack_service import (
    SlackService,
    TimestampExpiredError,
    verify_signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_RAG_SOURCE_RE = re.compile(
    r"\[(\d+)\] \(score: ([\d.]+)\) \[(.+?)\]"
)

_FENCED_BLOCK_RE = re.compile(
    r"```[^\n`]*\n.*?\n?```",
    re.DOTALL,
)
_LOOSE_SQL_LINE_RE = re.compile(
    r"^\s*(SELECT|WITH|FROM|WHERE|GROUP BY|ORDER BY|HAVING|JOIN|"
    r"LEFT JOIN|RIGHT JOIN|INNER JOIN|UNION|LIMIT)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_SQL_KEYWORD_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|FROM|WHERE|JOIN)\b",
    re.IGNORECASE,
)


def _strip_sql_from_answer(text: str) -> str:
    """최종 답변에서 LLM이 누출시킨 SQL/코드펜스를 제거한다.

    ReAct 상위 LLM이 시스템 프롬프트의 "SQL 금지" 규칙을 무시하고
    답변 앞뒤에 SQL 코드 펜스를 그대로 출력하는 경우가 있다.
    사용자가 Slack에서 보는 최종 응답에는 자연어 답변만 남기기 위해
    송출 직전에 다음을 수행한다.

    1. 모든 펜스 블록(```...```) 제거 (언어 태그 무관)
    2. 반쪽만 열린 채 남은 ``` 토큰 정리
    3. SQL 키워드로 시작하는 잔여 라인 제거
    """
    stripped = _FENCED_BLOCK_RE.sub("", text)
    # 반쪽 펜스 정리
    stripped = stripped.replace("```sql", "").replace("```SQL", "")
    stripped = stripped.replace("```", "")
    # 남은 SQL 라인 제거 (키워드가 줄 선두에 있는 경우만)
    if _SQL_KEYWORD_RE.search(stripped):
        stripped = _LOOSE_SQL_LINE_RE.sub("", stripped)
    # 공백/빈 줄 정리
    lines = [line.rstrip() for line in stripped.splitlines()]
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if not line.strip():
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _extract_references(
    messages: list[Any],
) -> str:
    """현재 턴의 ToolMessage에서 참고 자료 출처를 추출한다."""
    # 마지막 HumanMessage 이후의 메시지만 대상으로 한다
    last_human_idx = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = i
    current_turn = messages[last_human_idx + 1:]

    seen_files: set[str] = set()
    rag_sources: list[tuple[str, str]] = []
    web_urls: list[tuple[str, str]] = []

    for msg in current_turn:
        if not isinstance(msg, ToolMessage):
            continue

        content = (
            msg.content
            if isinstance(msg.content, str)
            else str(msg.content)
        )

        if msg.name == "retrieval_tool":
            for m in _RAG_SOURCE_RE.finditer(content):
                filename = m.group(3)
                score = m.group(2)
                if filename not in seen_files:
                    seen_files.add(filename)
                    rag_sources.append((filename, score))

        elif msg.name == "web_search":
            try:
                import ast
                data = ast.literal_eval(content)
                for r in data.get("results", []):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    if url and (title, url) not in web_urls:
                        web_urls.append((title, url))
            except Exception:
                pass

    if not rag_sources and not web_urls:
        return ""

    parts: list[str] = ["\n\n---\n**참고 자료**"]
    for filename, score in rag_sources:
        parts.append(f"- 📎 {filename} (score: {score})")
    for title, url in web_urls:
        parts.append(f"- 🔗 [{title}]({url})")
    return "\n".join(parts)


@router.post("/slack/events")
async def slack_events(
    request: Request, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    try:
        if not verify_signature(body, timestamp, signature):
            raise HTTPException(
                status_code=403, detail="Invalid signature"
            )
    except TimestampExpiredError:
        raise HTTPException(
            status_code=403, detail="Request too old"
        )

    payload = await request.json()

    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})

    if event.get("bot_id"):
        return {"ok": True}
    if event.get("subtype"):
        return {"ok": True}

    event_type = event.get("type")
    is_app_mention = event_type == "app_mention"
    is_dm = (
        event_type == "message"
        and event.get("channel_type") == "im"
    )
    is_channel_thread = (
        event_type == "message"
        and event.get("channel_type") != "im"
        and event.get("thread_ts") is not None
    )

    if not is_app_mention and not is_dm and not is_channel_thread:
        return {"ok": True}

    background_tasks.add_task(
        handle_message_event,
        event=event,
        workflow=request.app.state.workflow,
    )
    return {"ok": True}


_TOOL_STATUS: dict[str, str] = {
    "retrieval_tool": "🔍 내부 문서 검색 중...",
    "web_search": "🌐 웹 검색 중...",
}
_UPDATE_INTERVAL = 3.0


async def _stream_response(
    workflow: Any,
    input_messages: dict[str, Any],
    config: dict[str, Any],
    slack_service: SlackService,
    channel: str,
    loading_ts: str,
) -> tuple[str, list[ToolMessage]]:
    """LLM 스트림을 소비하며 Slack 메시지를 점진적으로 업데이트한다."""
    buffer = ""
    tool_messages: list[ToolMessage] = []
    last_update = time.monotonic()

    async for event in workflow.astream_events(
        input_messages, config=config, version="v2"
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            status_text = _TOOL_STATUS.get(event["name"])
            if status_text:
                await slack_service.update_message(
                    channel=channel,
                    ts=loading_ts,
                    text=status_text,
                )
                last_update = time.monotonic()

        elif kind == "on_tool_end":
            if event["name"] in _TOOL_STATUS:
                output = event["data"].get("output", "")
                tool_messages.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id="",
                        name=event["name"],
                    )
                )

        elif kind == "on_chat_model_stream":
            # text_to_sql_tool 등 tools 노드 내부에서 돌아가는
            # inner LLM 스트림은 버퍼에 담으면 SQL이 그대로 섞인다.
            # 최상위 call_llm 노드에서 발생한 토큰만 사용한다.
            node = event.get("metadata", {}).get("langgraph_node")
            if node != "call_llm":
                continue
            chunk = event["data"]["chunk"]
            raw = (
                chunk.content
                if hasattr(chunk, "content")
                else ""
            )
            if isinstance(raw, list):
                token = "".join(
                    block.get("text", "")
                    if isinstance(block, dict)
                    else str(block)
                    for block in raw
                )
            else:
                token = raw or ""
            if token:
                buffer += token
                now = time.monotonic()
                if now - last_update >= _UPDATE_INTERVAL:
                    await slack_service.update_message(
                        channel=channel,
                        ts=loading_ts,
                        text=buffer + " ▌",
                    )
                    last_update = now

    answer = buffer or "응답을 생성하지 못했습니다."
    return answer, tool_messages


async def handle_message_event(
    event: dict[str, Any], workflow: Any = None
) -> None:
    channel = event["channel"]
    user = event["user"]
    text = event.get("text", "")
    is_dm = event.get("channel_type") == "im"
    is_channel_thread = (
        event.get("type") == "message"
        and not is_dm
        and event.get("thread_ts") is not None
    )

    if is_dm:
        thread_ts = "dm"  # DM은 채널당 하나의 스레드로 취급
    else:
        thread_ts = event.get("thread_ts") or event["ts"]
    # DM에서는 thread_ts 없이 일반 메시지로 응답
    reply_thread_ts: str | None = None if is_dm else thread_ts

    slack_service = SlackService()

    async with AsyncSessionLocal() as db:
        # 채널 스레드 메시지: 봇이 참여한 스레드인지 확인
        if is_channel_thread:
            stmt = select(SlackThread).where(
                SlackThread.slack_channel_id == channel,
                SlackThread.slack_thread_ts == event["thread_ts"],
            )
            result = await db.execute(stmt)
            if result.scalar_one_or_none() is None:
                return

        query_service = QueryService(db)

        if await query_service.has_active_queries(user, limit=2):
            await slack_service.send_message(
                channel=channel,
                thread_ts=reply_thread_ts,
                text="현재 처리 중인 요청이 있습니다."
                " 잠시 후 다시 질문해주세요.",
            )
            return

        slack_thread = await _upsert_slack_thread(
            db, channel, thread_ts, user
        )

        user_query = await query_service.create_query(
            slack_thread_id=slack_thread.id,
            slack_user_id=user,
            query_text=text,
        )

        loading_response = await slack_service.send_loading_message(
            channel=channel, thread_ts=reply_thread_ts
        )
        loading_ts = loading_response["ts"]

        await query_service.update_status(
            user_query.id, QueryStatusEnum.processing
        )

        try:
            config = {
                "configurable": {
                    "thread_id": str(slack_thread.id)
                }
            }
            answer, tool_msgs = await _stream_response(
                workflow=workflow,
                input_messages={
                    "messages": [HumanMessage(content=text)]
                },
                config=config,
                slack_service=slack_service,
                channel=channel,
                loading_ts=loading_ts,
            )

            sanitized = _strip_sql_from_answer(answer)
            if sanitized != answer:
                logger.info(
                    "[Slack] SQL/code-fence 제거됨 (원본 %d자 → %d자)",
                    len(answer),
                    len(sanitized),
                )
            answer = sanitized or "응답을 생성하지 못했습니다."

            ref_messages: list[Any] = [
                HumanMessage(content=""),
                *tool_msgs,
            ]
            answer += _extract_references(ref_messages)

            await query_service.update_status(
                user_query.id, QueryStatusEnum.completed,
                answer=answer,
            )
            await slack_service.update_message(
                channel=channel, ts=loading_ts, text=answer
            )
        except Exception:
            logger.exception("LangGraph 처리 중 오류 발생")
            await query_service.update_status(
                user_query.id, QueryStatusEnum.failed
            )
            await slack_service.update_message(
                channel=channel,
                ts=loading_ts,
                text="❌ 오류가 발생했습니다."
                " 다시 시도해 주세요.",
            )


async def _upsert_slack_thread(
    db: AsyncSession,
    channel: str,
    thread_ts: str,
    user: str,
) -> SlackThread:
    stmt = select(SlackThread).where(
        SlackThread.slack_channel_id == channel,
        SlackThread.slack_thread_ts == thread_ts,
    )
    result = await db.execute(stmt)
    thread = result.scalar_one_or_none()

    if thread is None:
        thread = SlackThread(
            slack_channel_id=channel,
            slack_thread_ts=thread_ts,
            slack_user_id=user,
        )
        db.add(thread)
        await db.commit()
        await db.refresh(thread)

    return thread
