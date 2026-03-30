import logging
from datetime import datetime

import pytz
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
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
    "- If the question is about internal company knowledge,"
    " policies, or documents → use retrieval_tool.\n"
    "- If the question requires real-time, external,"
    " or up-to-date information → use web_search.\n"
    "- If the question is a greeting, small talk,"
    " or can be answered from general knowledge"
    " → respond directly without tools.\n"
    "- If the query is unclear, ask a clarifying question"
    " instead of using a tool.\n\n"
    "## Tool Calling Rules\n"
    "- When calling a tool, return ONLY valid JSON.\n"
    "- Do NOT include any additional text.\n"
    "- Choose the most appropriate tool for the task.\n"
    "- Do not call multiple tools unless absolutely"
    " necessary.\n"
    "- If no tool is needed, respond normally in Korean.\n\n"
    "## Tool Execution Handling\n"
    "- If a tool fails or returns no useful result,"
    " explain the situation and respond without guessing.\n\n"
    "## Response Style\n"
    "- Keep answers concise and relevant.\n"
    "- Avoid unnecessary explanations unless requested.\n"
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
        if recent_tokens + msg_tokens > _RECENT_MESSAGES_BUDGET:
            break
        recent.insert(0, msg)
        recent_tokens += msg_tokens

    # HumanMessage로 시작하도록 조정
    while recent and not isinstance(recent[0], HumanMessage):
        recent.pop(0)

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
        [SystemMessage(content="\n".join(parts))]
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

    logger.info(
        "[LLM] 호출 시작 (%s/%s, 메시지 %d개,"
        " 전체 %d개, 요약 %s)",
        _settings.llm_provider,
        _settings.llm_model_name,
        len(messages),
        len(state["messages"]) + 1,
        "있음" if summary else "없음",
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
