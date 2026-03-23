import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from app.graph.state import KasaNovaState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "당신은 KasaNova입니다. 사내 지식 검색을 돕는 AI 어시스턴트입니다.\n"
    "직원들의 업무 관련 질문에 정확하고 간결하게 답변합니다.\n"
    "모르는 내용은 모른다고 솔직하게 말하고, 추측으로 답변하지 않습니다.\n"
    "답변은 한국어로 작성합니다."
)


async def call_llm(
    state: KasaNovaState, llm: BaseChatModel
) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    logger.info("[LLM] 호출 시작 (메시지 %d개)", len(messages))
    response: AIMessage = await llm.ainvoke(messages)
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
