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
from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)
_settings = get_settings()

_KST = pytz.timezone("Asia/Seoul")

_SYSTEM_PROMPT_TEMPLATE = (
    "Current date and time: {now} (KST).\n\n"
    "You are KasaNova, an AI assistant that helps employees"
    " search internal company knowledge.\n"
    "Provide accurate, concise, and helpful answers"
    " to work-related questions.\n"
    "If you do not know something, say you do not know."
    " Do not guess or fabricate information.\n"
    "All responses must be written in Korean.\n\n"
    "## Tool Usage Guidelines\n"
    "- Prefer answering directly using your own knowledge"
    " when possible.\n"
    "- Use the web_search tool ONLY when real-time or"
    " external information is required.\n"
    "- If you are uncertain about factual information,"
    " use web_search instead of guessing.\n"
    "- Do NOT use tools for simple inputs such as"
    " greetings, tests, or basic questions.\n"
    "- Do NOT use tools if the answer can be reasonably"
    " inferred from general knowledge.\n"
    "- If the query is unclear, ask a clarifying question"
    " instead of using a tool.\n\n"
    "## Tool Calling Rules\n"
    "- When calling a tool, return ONLY valid JSON.\n"
    "- Do NOT include any additional text.\n"
    "- Ensure the JSON is properly formatted.\n"
    "- Do not wrap JSON in quotes.\n"
    "- If no tool is needed, respond normally in Korean.\n\n"
    "## Tool Execution Handling\n"
    "- If a tool fails or returns no useful result,"
    " explain the situation and respond without guessing.\n"
    "- Choose the most appropriate tool for the task.\n"
    "- Do not call multiple tools unless absolutely"
    " necessary.\n\n"
    "## Response Style\n"
    "- Keep answers concise and relevant.\n"
    "- Avoid unnecessary explanations unless requested.\n"
)


def _build_system_prompt(
    retrieved_docs: list[dict] | None = None,
) -> str:
    now = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S")
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(now=now)

    if retrieved_docs:
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(
                f"[{doc['source']}] {doc['content']}"
            )
        context = "\n\n".join(context_parts)
        prompt += (
            "\n## Retrieved Context\n"
            "Answer based on the following retrieved"
            " documents. If the answer is not in the"
            " context, say so.\n\n"
            f"{context}\n"
        )

    return prompt


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
    retrieved_docs = state.get("retrieved_docs")
    messages = [
        SystemMessage(
            content=_build_system_prompt(retrieved_docs)
        )
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
            "[LLM] tool_calls 반환 (%d건)", len(response.tool_calls)
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
            response.content[:200] if response.content else "(empty)",
        )
    return {"messages": [response]}


async def retrieve(
    state: KasaNovaState, searcher: HybridSearcher
) -> dict:
    """마지막 사용자 메시지로 하이브리드 검색을 수행한다."""
    query = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            query = msg.content
            break

    if not query:
        logger.warning("[Retrieve] 사용자 메시지를 찾을 수 없음")
        return {"retrieved_docs": []}

    logger.info("[Retrieve] 검색 시작: %s", query[:100])
    docs = await searcher.search(query)
    logger.info("[Retrieve] %d건 검색 완료", len(docs))
    return {"retrieved_docs": docs}
