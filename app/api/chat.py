"""Slack 없이 HTTP로 직접 질의할 수 있는 테스트용 채팅 엔드포인트."""

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.slack import _extract_references, _strip_sql_from_answer
from app.core.database import AsyncSessionLocal
from app.models.slack_thread import SlackThread

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_TOOL_STATUS: dict[str, str] = {
    "retrieval_tool": "🔍 내부 문서 검색 중...",
    "web_search": "🌐 웹 검색 중...",
    "text_to_sql_tool": "🗄️ SQL 생성 중...",
    "dabs_summary_list": "📋 DABS 목록 조회 중...",
}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    references: str


@router.get("/", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    """테스트용 채팅 UI 페이지를 반환한다."""
    html = _CHAT_HTML
    return HTMLResponse(content=html)


@router.post("/send")
async def chat_send(
    body: ChatRequest, request: Request
) -> ChatResponse:
    """메시지를 워크플로우에 전달하고 응답을 반환한다."""
    workflow = request.app.state.workflow
    session_id = body.session_id or str(uuid.uuid4())

    async with AsyncSessionLocal() as db:
        slack_thread = await _upsert_chat_thread(
            db, session_id
        )
        config = {
            "configurable": {
                "thread_id": str(slack_thread.id)
            }
        }

    answer, tool_msgs = await _run_workflow(
        workflow, body.message, config
    )

    sanitized = _strip_sql_from_answer(answer)
    answer = sanitized or "응답을 생성하지 못했습니다."

    ref_messages: list[Any] = [
        HumanMessage(content=""),
        *tool_msgs,
    ]
    references = _extract_references(ref_messages)
    answer += references

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        references=references,
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest, request: Request
) -> StreamingResponse:
    """SSE 스트리밍으로 응답을 반환한다."""
    workflow = request.app.state.workflow
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        "[Chat] 요청 수신 (session=%s): %s",
        session_id, body.message[:100],
    )

    async with AsyncSessionLocal() as db:
        slack_thread = await _upsert_chat_thread(
            db, session_id
        )
        config = {
            "configurable": {
                "thread_id": str(slack_thread.id)
            }
        }

    async def event_generator():
        yield f"data: {{\"type\":\"session\",\"session_id\":\"{session_id}\"}}\n\n"

        buffer = ""
        tool_msgs: list[ToolMessage] = []

        try:
            async for event in workflow.astream_events(
                {"messages": [HumanMessage(content=body.message)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_tool_start":
                    logger.info(
                        "[Chat] 도구 실행 시작: %s",
                        event["name"],
                    )
                    status = _TOOL_STATUS.get(event["name"])
                    if status:
                        yield (
                            f"data: {json.dumps({'type': 'status', 'text': status}, ensure_ascii=False)}\n\n"
                        )

                elif kind == "on_tool_end":
                    logger.info(
                        "[Chat] 도구 실행 완료: %s",
                        event["name"],
                    )
                    if event["name"] in _TOOL_STATUS:
                        output = event["data"].get("output", "")
                        tool_msgs.append(
                            ToolMessage(
                                content=str(output),
                                tool_call_id="",
                                name=event["name"],
                            )
                        )

                elif kind == "on_chat_model_stream":
                    node = event.get(
                        "metadata", {}
                    ).get("langgraph_node")
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
                        yield (
                            f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                        )

            answer = buffer or "응답을 생성하지 못했습니다."
            sanitized = _strip_sql_from_answer(answer)
            if sanitized != answer:
                logger.info(
                    "[Chat] SQL/code-fence 제거됨"
                    " (원본 %d자 → %d자)",
                    len(answer), len(sanitized),
                )
            answer = sanitized or answer

            ref_messages: list[Any] = [
                HumanMessage(content=""),
                *tool_msgs,
            ]
            references = _extract_references(ref_messages)

            logger.info(
                "[Chat] 응답 완료 (session=%s, %d자)",
                session_id, len(answer),
            )

            yield (
                f"data: {json.dumps({'type': 'done', 'answer': answer, 'references': references}, ensure_ascii=False)}\n\n"
            )
        except Exception:
            logger.exception(
                "[Chat] LangGraph 처리 중 오류 발생"
                " (session=%s)", session_id,
            )
            yield (
                f"data: {json.dumps({'type': 'done', 'answer': '오류가 발생했습니다. 다시 시도해 주세요.', 'references': ''}, ensure_ascii=False)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


async def _run_workflow(
    workflow: Any,
    text: str,
    config: dict[str, Any],
) -> tuple[str, list[ToolMessage]]:
    """워크플로우를 실행하고 최종 응답을 반환한다."""
    buffer = ""
    tool_messages: list[ToolMessage] = []

    async for event in workflow.astream_events(
        {"messages": [HumanMessage(content=text)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_end":
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
            node = event.get(
                "metadata", {}
            ).get("langgraph_node")
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

    answer = buffer or "응답을 생성하지 못했습니다."
    return answer, tool_messages


async def _upsert_chat_thread(
    db: AsyncSession, session_id: str
) -> SlackThread:
    """채팅 세션용 SlackThread를 조회하거나 생성한다."""
    stmt = select(SlackThread).where(
        SlackThread.slack_channel_id == "web-chat",
        SlackThread.slack_thread_ts == session_id,
    )
    result = await db.execute(stmt)
    thread = result.scalar_one_or_none()

    if thread is None:
        thread = SlackThread(
            slack_channel_id="web-chat",
            slack_thread_ts=session_id,
            slack_user_id="web-user",
        )
        db.add(thread)
        await db.commit()
        await db.refresh(thread)

    return thread


_CHAT_HTML = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KasaNova Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont,
      'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #1a1a2e;
    color: #fff;
    padding: 16px 24px;
    font-size: 18px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  header span { font-size: 14px; color: #aaa; font-weight: 400; }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: #1a1a2e;
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.bot {
    align-self: flex-start;
    background: #fff;
    color: #333;
    border: 1px solid #e0e0e0;
    border-bottom-left-radius: 4px;
  }
  .msg.status {
    align-self: flex-start;
    background: transparent;
    color: #888;
    font-size: 13px;
    padding: 4px 0;
    border: none;
  }
  #input-area {
    display: flex;
    gap: 8px;
    padding: 16px 24px;
    background: #fff;
    border-top: 1px solid #e0e0e0;
  }
  #input-area textarea {
    flex: 1;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    outline: none;
    min-height: 44px;
    max-height: 120px;
  }
  #input-area textarea:focus { border-color: #1a1a2e; }
  #input-area button {
    padding: 0 24px;
    background: #1a1a2e;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    white-space: nowrap;
  }
  #input-area button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  #input-area button:hover:not(:disabled) { background: #16213e; }
  .session-bar {
    display: flex;
    justify-content: flex-end;
    padding: 8px 24px 0;
    gap: 8px;
  }
  .session-bar button {
    font-size: 12px;
    padding: 4px 12px;
    background: #eee;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    color: #555;
  }
  .session-bar button:hover { background: #ddd; }
</style>
</head>
<body>
  <header>
    KasaNova <span>테스트 채팅</span>
  </header>
  <div class="session-bar">
    <button onclick="newSession()">새 세션</button>
  </div>
  <div id="messages"></div>
  <div id="input-area">
    <textarea id="input" placeholder="질문을 입력하세요..."
      rows="1"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"
      oninput="autoResize(this)"></textarea>
    <button id="send-btn" onclick="send()">전송</button>
  </div>
<script>
let sessionId = null;

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  document.getElementById('messages').appendChild(d);
  d.scrollIntoView({ behavior: 'smooth' });
  return d;
}

function newSession() {
  sessionId = null;
  document.getElementById('messages').innerHTML = '';
}

async function send() {
  const input = document.getElementById('input');
  const btn = document.getElementById('send-btn');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  btn.disabled = true;

  addMsg(text, 'user');
  const statusEl = addMsg('...', 'status');
  const botEl = addMsg('', 'bot');

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.type === 'session') {
          sessionId = data.session_id;
        } else if (data.type === 'status') {
          statusEl.textContent = data.text;
        } else if (data.type === 'token') {
          botEl.textContent += data.text;
          botEl.scrollIntoView({ behavior: 'smooth' });
        } else if (data.type === 'done') {
          botEl.textContent = data.answer;
          if (data.references) {
            botEl.textContent += data.references;
          }
          statusEl.remove();
        }
      }
    }
  } catch (e) {
    botEl.textContent = '오류가 발생했습니다: ' + e.message;
    statusEl.remove();
  }

  btn.disabled = false;
  input.focus();
}
</script>
</body>
</html>
"""
