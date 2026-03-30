import logging
from datetime import datetime

import pytz
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    trim_messages,
)

from app.core.config import get_settings
from app.graph.state import KasaNovaState

logger = logging.getLogger(__name__)
_settings = get_settings()

_KST = pytz.timezone("Asia/Seoul")

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


def _build_system_prompt() -> str:
    now = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S")
    return _SYSTEM_PROMPT_TEMPLATE.format(now=now)


async def call_llm(
    state: KasaNovaState, llm: BaseChatModel
) -> dict:
    trimmed = trim_messages(
        state["messages"],
        max_tokens=20,
        token_counter=lambda msgs: len(msgs),
        strategy="last",
        allow_partial=False,
        start_on="human",
    )
    messages = [
        SystemMessage(content=_build_system_prompt())
    ] + trimmed
    logger.info(
        "[LLM] 호출 시작 (%s/%s, 메시지 %d개, 전체 %d개)",
        _settings.llm_provider,
        _settings.llm_model_name,
        len(messages),
        len(state["messages"]) + 1,
    )
    response: AIMessage = await llm.ainvoke(messages)
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
    return {"messages": [response]}
