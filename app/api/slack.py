import logging
import re
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
            result = await workflow.ainvoke(
                {"messages": [HumanMessage(content=text)]},
                config=config,
            )
            answer_message: AIMessage = result["messages"][-1]
            answer: str = (
                answer_message.content
                or "응답을 생성하지 못했습니다."
            )
            answer += _extract_references(
                result["messages"]
            )

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
                text="처리 중 오류가 발생했습니다."
                " 잠시 후 다시 시도해주세요.",
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
