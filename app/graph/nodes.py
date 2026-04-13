import logging
from datetime import datetime

import pytz
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.core.config import get_settings
from app.graph.state import KasaNovaState

logger = logging.getLogger(__name__)
_settings = get_settings()

_KST = pytz.timezone("Asia/Seoul")

_SUMMARY_TRIGGER_TOKENS = 3000
_RECENT_MESSAGES_BUDGET = 2000

_SYSTEM_PROMPT_TEMPLATE = (
    "Current date and time: {now} (KST).\n\n"
    "You are KasaNova, an AI assistant that helps employees"
    " at Kasa Korea.\n"
    "Provide accurate, concise, and helpful answers.\n"
    "If you do not know something, say you do not know."
    " Do not guess or fabricate information.\n"
    "All responses must be written in Korean.\n\n"
    "## Tool Usage Guidelines\n"
    "- If the question is about internal company documents,"
    " policies, guides, or process knowledge"
    " → use retrieval_tool.\n"
    "- If the question asks about specific records, counts,"
    " rankings, statistics, or lists from internal"
    " operational data (e.g., users, posts, comments,"
    " 픽앤톡, transactions, orders, DABS trades, ledger)"
    " → use text_to_sql_tool.\n"
    "- If the question is about identifying a specific DABS,"
    " 댑스, building, 빌딩, 건물, or real estate asset"
    " by name or code → use dabs_summary_list"
    " (often combined with text_to_sql_tool).\n"
    "- If the question requires real-time, external,"
    " or up-to-date information from outside the company"
    " → use web_search.\n"
    "- If the question is a greeting, small talk,"
    " or can be answered from general knowledge"
    " → respond directly without tools.\n"
    "- If the query is genuinely ambiguous and no tool"
    " could reasonably resolve it, ask a clarifying"
    " question. Do NOT give up with 'I don't have access'"
    " before attempting the relevant tool.\n\n"
    "## Tool Calling Rules\n"
    "- When calling a tool, return ONLY valid JSON.\n"
    "- Do NOT include any additional text.\n"
    "- Choose the most appropriate tool for the task.\n"
    "- Do not call multiple tools unless absolutely"
    " necessary.\n"
    "- Do not call the same tool repeatedly with"
    " similar queries. Use the results from the first"
    " 1-2 calls to compose your answer.\n"
    "- If no tool is needed, respond normally in Korean.\n\n"
    "## Tool Execution Handling\n"
    "- If a tool fails or returns no useful result,"
    " explain the situation and respond without guessing.\n\n"
    "## Response Style\n"
    "- Keep answers concise and relevant.\n"
    "- Avoid unnecessary explanations unless requested.\n"
    "- Never echo SQL statements or fenced code blocks"
    " (```...```) in the final response to the user.\n"
    "  Tool results may contain raw data rows; translate"
    " them into natural Korean sentences.\n"
)


def _estimate_tokens(text: str) -> int:
    """한/영 혼합 텍스트의 토큰 수를 근사한다 (1토큰 ≈ 3자)."""
    return max(1, len(text) // 3)


def _estimate_messages_tokens(
    messages: list[BaseMessage],
) -> int:
    """메시지 리스트의 총 토큰 수를 추정한다."""
    total = 0
    for msg in messages:
        content = (
            msg.content
            if isinstance(msg.content, str)
            else str(msg.content)
        )
        total += _estimate_tokens(content) + 4
    return total


def _build_system_prompt(summary: str = "") -> str:
    now = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S")
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(now=now)
    if summary:
        prompt += (
            "\n## Previous Conversation Summary\n"
            f"{summary}\n"
        )
    return prompt


async def _maybe_summarize(
    state: KasaNovaState, llm_base: BaseChatModel
) -> tuple[str, list[BaseMessage]]:
    """토큰 임계치 초과 시 오래된 메시지를 요약한다."""
    messages = state["messages"]
    existing_summary = state.get("summary") or ""

    # 도구 호출 사이클 중에는 요약 스킵
    if messages and isinstance(messages[-1], ToolMessage):
        return existing_summary, messages

    total_tokens = _estimate_messages_tokens(messages)
    if total_tokens <= _SUMMARY_TRIGGER_TOKENS:
        return existing_summary, messages

    logger.info(
        "[Summary] 토큰 임계치 초과 (%d > %d), 요약 시작",
        total_tokens,
        _SUMMARY_TRIGGER_TOKENS,
    )

    # 최근 메시지를 예산 내에서 확보
    recent: list[BaseMessage] = []
    recent_tokens = 0
    for msg in reversed(messages):
        content = (
            msg.content
            if isinstance(msg.content, str)
            else str(msg.content)
        )
        msg_tokens = _estimate_tokens(content) + 4
        if (
            recent_tokens + msg_tokens
            > _RECENT_MESSAGES_BUDGET
            and recent
        ):
            break
        recent.insert(0, msg)
        recent_tokens += msg_tokens

    # HumanMessage로 시작하도록 조정
    while recent and not isinstance(recent[0], HumanMessage):
        recent.pop(0)

    # 최소 1개 메시지는 유지
    if not recent:
        recent = messages[-1:]

    num_to_summarize = len(messages) - len(recent)
    if num_to_summarize <= 0:
        return existing_summary, messages

    messages_to_summarize = messages[:num_to_summarize]

    # 요약 프롬프트 구성
    parts: list[str] = []
    if existing_summary:
        parts.append(
            "기존 대화 요약:\n"
            f"{existing_summary}\n\n"
            "아래 새로운 메시지를 기존 요약에 통합하세요:"
        )
    else:
        parts.append(
            "아래 대화를 한국어로 간결하게 요약하세요. "
            "핵심 주제, 결정사항, 사용자가 물어본 정보를"
            " 포함하세요:"
        )

    for msg in messages_to_summarize:
        content = (
            msg.content
            if isinstance(msg.content, str)
            else str(msg.content)
        )
        if isinstance(msg, HumanMessage):
            parts.append(f"사용자: {content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"AI: {content[:300]}")
        else:
            parts.append(f"도구 결과: {content[:200]}")

    parts.append(
        "\n200자 이내로 통합 요약을 작성하세요."
    )

    response = await llm_base.ainvoke(
        [HumanMessage(content="\n".join(parts))]
    )
    new_summary = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    logger.info(
        "[Summary] 요약 완료 (%d자, 최근 메시지 %d개 유지)",
        len(new_summary),
        len(recent),
    )
    return new_summary, recent


_TOOL_MSG_PLACEHOLDER = "(이전 도구 호출 결과 - 답변에 반영됨)"


def _compact_old_tool_messages(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """답변 완료된 이전 ToolMessage를 플레이스홀더로 대체."""
    # 현재 진행 중인 도구 호출의 tool_call_id 수집
    # (마지막 AIMessage가 tool_calls를 가진 경우)
    active_tool_ids: set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    active_tool_ids.add(tc["id"])
            break

    result: list[BaseMessage] = []
    for msg in messages:
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id not in active_tool_ids
        ):
            result.append(
                ToolMessage(
                    content=_TOOL_MSG_PLACEHOLDER,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        else:
            result.append(msg)
    return result


async def call_llm(
    state: KasaNovaState,
    llm_with_tools: BaseChatModel,
    llm_base: BaseChatModel,
) -> dict:
    summary, recent_messages = await _maybe_summarize(
        state, llm_base
    )

    messages = [
        SystemMessage(
            content=_build_system_prompt(summary)
        )
    ] + recent_messages

    # 이전 턴의 ToolMessage를 플레이스홀더로 대체
    messages = _compact_old_tool_messages(messages)

    # 프롬프트 크기 로깅
    total_chars = sum(
        len(m.content) if isinstance(m.content, str)
        else len(str(m.content))
        for m in messages
    )
    total_tokens_est = _estimate_messages_tokens(messages)
    tool_msg_info = ""
    for m in messages:
        if isinstance(m, ToolMessage):
            content = (
                m.content
                if isinstance(m.content, str)
                else str(m.content)
            )
            tool_msg_info += (
                f", {m.name}={len(content)}자"
            )

    logger.info(
        "[LLM] 호출 시작 (%s/%s, 메시지 %d개,"
        " 전체 %d개, 요약 %s,"
        " 프롬프트 %d자/~%d토큰%s)",
        _settings.llm_provider,
        _settings.llm_model_name,
        len(messages),
        len(state["messages"]) + 1,
        "있음" if summary else "없음",
        total_chars,
        total_tokens_est,
        tool_msg_info,
    )
    response: AIMessage = await llm_with_tools.ainvoke(
        messages
    )
    logger.info(
        "[LLM] 응답 원본: content=%r, tool_calls=%r, "
        "additional_kwargs=%r, response_metadata=%r",
        response.content,
        response.tool_calls,
        response.additional_kwargs,
        response.response_metadata,
    )
    if response.tool_calls:
        logger.info(
            "[LLM] tool_calls 반환 (%d건)",
            len(response.tool_calls),
        )
        for i, tc in enumerate(response.tool_calls, 1):
            logger.info(
                "[LLM]   └ [%d/%d] %s(%s)",
                i,
                len(response.tool_calls),
                tc["name"],
                tc["args"],
            )
    else:
        logger.info(
            "[LLM] 최종 응답 반환: %s",
            response.content[:200]
            if response.content
            else "(empty)",
        )
    return {"messages": [response], "summary": summary}
